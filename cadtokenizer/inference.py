from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import LogitsProcessor

from .common import device_for, file_hash, load_checkpoint, require_container, write_json
from .instruction_data import instruction_tokens
from .representation import analyze, split_pairs, stringify
from .vq import PairDataset, collate_pairs, load_vq, to_device


class NativeCadGrammar(LogitsProcessor):
    """Framing at LLM level; detailed SEM constraints are applied by the VQ decoder."""
    def __init__(self, native_ids, boundaries, prompt_length, max_primitives, max_new_tokens, max_pairs=10, trailer_ids=None):
        self.native_ids = native_ids
        self.trailer_ids = list(trailer_ids) if trailer_ids else []
        self._trailer_set = set(self.trailer_ids)
        self.boundaries = boundaries
        self.prompt_length = prompt_length
        self.max_primitives = max_primitives
        self.max_new_tokens = max_new_tokens
        self.max_pairs = max_pairs

    def __call__(self, input_ids, scores):
        for row in range(len(input_ids)):
            generated = input_ids[row, self.prompt_length:].tolist()
            pairs = generated.count(self.boundaries['pair_end'])
            last_boundary = max((i for i, t in enumerate(generated) if t == self.boundaries['pair_end']), default=-1)
            current = len(generated) - last_boundary - 1
            remaining = self.max_new_tokens - len(generated)
            allowed = []
            if self.trailer_ids:
                segment = generated[last_boundary + 1:]
                trailer = sum(t in self._trailer_set for t in segment)
                codes = len(segment) - trailer
                # A pair is codes, then exactly three trailer bins, then the pair boundary.
                if trailer:
                    allowed.extend(self.trailer_ids if trailer < 3 else [self.boundaries['pair_end']])
                elif codes:
                    if codes < self.max_primitives and remaining > 5:
                        allowed.extend(self.native_ids)
                    allowed.extend(self.trailer_ids)
                else:
                    if pairs:
                        allowed.append(self.boundaries['end'])
                    if not pairs or (pairs < self.max_pairs and remaining >= 6):
                        allowed.extend(self.native_ids)
                mask = torch.full_like(scores[row], -torch.inf)
                mask[allowed] = 0
                scores[row] += mask
                continue
            if current:
                if current < self.max_primitives and remaining > 2:
                    allowed.extend(self.native_ids)
                allowed.append(self.boundaries['pair_end'])
            elif pairs:
                allowed.append(self.boundaries['end'])
                if pairs < self.max_pairs and remaining >= 3:
                    allowed.extend(self.native_ids)
            else:
                allowed.extend(self.native_ids)
            mask = torch.full_like(scores[row], -torch.inf)
            mask[allowed] = 0
            scores[row] += mask
        return scores


@torch.inference_mode()
def decode_native_ids(vq, alignment, generated, decoder='alignment'):
    """Decode the same native IDs using learned vectors or exact codebook lookup."""
    if decoder not in ('alignment', 'codebook'):
        raise ValueError('cad_decoder must be alignment or codebook')
    boundaries = alignment['boundaries']
    native_to_code = {token: code for code, token in enumerate(alignment['native_ids'].tolist())}
    trailer_to_bin = {token: value for value, token in enumerate(alignment.get('trailer_ids') or [])}
    if set(trailer_to_bin) & set(native_to_code):
        raise ValueError('Trailer IDs overlap code IDs')
    if len(native_to_code) != len(alignment['native_ids']):
        raise ValueError('Native-ID mapping is not injective')
    if not generated or generated[-1] != boundaries['end']:
        raise ValueError('LLM did not terminate the CAD output')
    code_pairs, trailers, current, bins = [], [], [], []
    for token in generated[:-1]:
        if token == boundaries['pair_end']:
            if not current:
                raise ValueError('Empty generated pair')
            if trailer_to_bin and len(bins) != 3:
                raise ValueError('Generated pair is missing its scale/origin trailer')
            code_pairs.append(current)
            trailers.append(bins)
            current, bins = [], []
        elif token in trailer_to_bin:
            if not current or len(bins) >= 3:
                raise ValueError('Misplaced scale/origin trailer token')
            bins.append(trailer_to_bin[token])
        else:
            if token not in native_to_code:
                raise ValueError(f'Unknown native CAD token: {token}')
            if bins:
                raise ValueError('Code token after scale/origin trailer')
            current.append(native_to_code[token])
    if current or bins or not code_pairs:
        raise ValueError('Incomplete generated pair boundary')
    tokens = []
    device = vq.codebook.weight.device
    for index, codes in enumerate(code_pairs):
        if decoder == 'codebook':
            memory = vq.codebook.weight[torch.tensor(codes, device=device, dtype=torch.long)]
        else:
            memory = alignment['reconstructed_code_vectors'][codes].to(device)
        pair = vq.reconstruct(memory, force_add=(index == 0))
        if trailer_to_bin:
            from .scale_origin_codec import merge_pair
            pair = merge_pair(pair, trailers[index])
        tokens.extend(pair)
    split_pairs(tokens)
    return stringify(tokens)


class CadGenerator:
    """Load the LLM/tokenizer once and reuse them for all benchmark examples."""
    def __init__(self, config):
        require_container()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = config
        self.cad_decoder = config.get('inference', {}).get('cad_decoder', 'alignment')
        if self.cad_decoder not in ('alignment', 'codebook'):
            raise ValueError('cad_decoder must be alignment or codebook')
        self.device = device_for(config)
        align_path = Path(config['alignment']['output']) / 'alignment.pt'
        vq_path = Path(config['vq']['output']) / 'best.pt'
        self.alignment = load_checkpoint(align_path)
        if self.alignment['vq_sha256'] != file_hash(vq_path):
            raise ValueError('Alignment/VQ checkpoint mismatch')
        self.run = json.loads((Path(config['sft']['output']) / 'run.json').read_text())
        if self.run['alignment_sha256'] != file_hash(align_path) or self.run['vq_sha256'] != file_hash(vq_path):
            raise ValueError('SFT, alignment and VQ checkpoints are incompatible')
        self.vq, _ = load_vq(vq_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(Path(config['alignment']['output']) / 'llm_tokenizer')
        dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        backbone = AutoModelForCausalLM.from_pretrained(self.alignment['model_name'], revision=self.alignment['model_revision'],
                                                       torch_dtype=dtype, attn_implementation='sdpa')
        self.model = PeftModel.from_pretrained(backbone, Path(config['sft']['output']) / 'final_adapter').to(self.device).eval()
        self.model.config.use_cache = True

    def _prompt(self, instruction, source_text, max_new_tokens):
        """Token IDs for one request. Accepts NO target/reference CAD: heldout ground truth cannot
        enter the model input through the benchmark API."""
        if max_new_tokens < 3:
            raise ValueError('Need at least three output tokens for a code, pair boundary, and end')
        alignment, device = self.alignment, self.device
        source_pairs = None
        if source_text is not None:
            # Read grouping from the checkpoint so runtime config cannot silently
            # reinterpret its learned code semantics.
            representation = self.vq.config['data']
            source_text, pairs = analyze(source_text, **{k: representation[k] for k in ('max_pair_tokens', 'max_primitives', 'extrusion_partition')},
                                         sketch_partition=representation.get('sketch_partition', 'curve'))
            batch = to_device(collate_pairs(pairs), device)
            with torch.inference_mode():
                _, codes, mask = self.vq.encode(batch['tokens'], batch['groups'])
            from .encode import trailer_native
            trailer_ids = alignment.get('trailer_ids')
            source_pairs = [alignment['native_ids'][codes[i][mask[i]].cpu()].tolist() + trailer_native(pairs[i]['tokens'], trailer_ids)
                            for i in range(len(pairs))]
        prompt = instruction_tokens(self.tokenizer, alignment['boundaries'], instruction, source_pairs)
        if len(prompt) + max_new_tokens > self.config['sft']['context_length']:
            raise ValueError('Prompt plus requested output budget exceeds context_length')
        return prompt, source_text

    def _processor(self, prompt_length, max_new_tokens):
        return NativeCadGrammar(self.alignment['native_ids'].tolist(), self.alignment['boundaries'], prompt_length,
                                self.config['data']['max_primitives'], max_new_tokens,
                                trailer_ids=self.alignment.get('trailer_ids'))

    def _report(self, instruction, source_text, generated):
        sequence = decode_native_ids(self.vq, self.alignment, generated, self.cad_decoder)
        return {'instruction': instruction, 'source_sequence': source_text, 'generated_native_ids': generated,
                'generated_sequence': sequence, 'status': 'syntax_valid', 'cad_decoder': self.cad_decoder,
                'geometry_validity': 'not_checked; run a CAD kernel to verify solid geometry'}

    def predict(self, instruction, source_text=None, max_new_tokens=256):
        """Reference single-prompt decoding."""
        prompt, source_text = self._prompt(instruction, source_text, max_new_tokens)
        ids = torch.tensor([prompt], device=self.device)
        with torch.inference_mode():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                         max_new_tokens=max_new_tokens, logits_processor=[self._processor(len(prompt), max_new_tokens)],
                                         eos_token_id=self.alignment['boundaries']['end'], pad_token_id=self.tokenizer.pad_token_id)
        return self._report(instruction, source_text, result[0, len(prompt):].tolist())

    def predict_candidates(self, instruction, source_text=None, max_new_tokens=256, k=8, temperature=1.0):
        """Sample k independent candidates for one prompt with the same grammar and VQ decoder.

        Used only by the best-of-k variant; the frozen benchmark stays greedy (`predict`).
        """
        if k < 1:
            raise ValueError('k must be at least one')
        prompt, source_text = self._prompt(instruction, source_text, max_new_tokens)
        end, pad = self.alignment['boundaries']['end'], self.tokenizer.pad_token_id
        ids = torch.tensor([prompt], device=self.device).repeat(k, 1)
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                      temperature=temperature, top_k=0, top_p=1.0,
                                      max_new_tokens=max_new_tokens, logits_processor=[self._processor(len(prompt), max_new_tokens)],
                                      eos_token_id=end, pad_token_id=pad)
        reports = []
        for row in range(k):
            generated = out[row, len(prompt):].tolist()
            if end in generated:
                generated = generated[:generated.index(end) + 1]
            try:
                reports.append(self._report(instruction, source_text, generated))
            except (ValueError, KeyError) as error:
                reports.append({'instruction': instruction, 'source_sequence': source_text, 'generated_native_ids': generated,
                                'status': 'inference_error', 'error': str(error)})
        return reports

    def prompt_length(self, instruction, max_new_tokens=256):
        """Token length of the prompt for one instruction, used to batch prompts without padding."""
        return len(self._prompt(instruction, None, max_new_tokens)[0])

    def predict_candidates_batch(self, instructions, max_new_tokens=256, k=8, temperature=1.0):
        """Sample k candidates for each instruction in one batched `generate` call (prompts left-padded).

        Returns one list of k reports per instruction, in order. Sampling variant only; see predict_candidates.
        """
        end, pad = self.alignment['boundaries']['end'], self.tokenizer.pad_token_id
        prompts = [self._prompt(instruction, None, max_new_tokens) for instruction in instructions]
        width = max(len(prompt) for prompt, _ in prompts)
        ids = torch.full((len(prompts) * k, width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, (prompt, _) in enumerate(prompts):
            for copy in range(k):
                row = index * k + copy
                ids[row, width - len(prompt):] = torch.tensor(prompt)
                mask[row, width - len(prompt):] = 1
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), do_sample=True,
                                      temperature=temperature, top_k=0, top_p=1.0,
                                      max_new_tokens=max_new_tokens, logits_processor=[self._processor(width, max_new_tokens)],
                                      eos_token_id=end, pad_token_id=pad)
        results = []
        for index, (_, canonical) in enumerate(prompts):
            reports = []
            for copy in range(k):
                generated = out[index * k + copy, width:].tolist()
                if end in generated:
                    generated = generated[:generated.index(end) + 1]
                try:
                    reports.append(self._report(instructions[index], canonical, generated))
                except (ValueError, KeyError) as error:
                    reports.append({'instruction': instructions[index], 'source_sequence': canonical, 'generated_native_ids': generated,
                                    'status': 'inference_error', 'error': str(error)})
            results.append(reports)
        return results

    def predict_batch(self, requests, max_new_tokens=256, batch_size=16):
        """Batched greedy decoding of [(instruction, source_text)] with the same grammar and decoder.

        Prompts are left-padded and sorted by length to limit padding; results come back in request
        order as a report dict or the ValueError/KeyError a single-prompt call would have raised.
        Padded bf16 batches can differ numerically from unpadded single-prompt decoding, so validate
        against `predict` before relying on it (see `benchmark-inference-profile`).
        """
        end, pad = self.alignment['boundaries']['end'], self.tokenizer.pad_token_id
        results, prepared = [None] * len(requests), []
        for index, (instruction, source_text) in enumerate(requests):
            try:
                prompt, canonical = self._prompt(instruction, source_text, max_new_tokens)
                prepared.append((index, prompt, canonical))
            except (ValueError, KeyError) as error:
                results[index] = error
        prepared.sort(key=lambda item: (len(item[1]), item[0]))
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            width = max(len(prompt) for _, prompt, _ in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, (_, prompt, _) in enumerate(chunk):
                ids[row, width - len(prompt):] = torch.tensor(prompt)
                mask[row, width - len(prompt):] = 1
            with torch.inference_mode():
                out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), do_sample=False,
                                          max_new_tokens=max_new_tokens, logits_processor=[self._processor(width, max_new_tokens)],
                                          eos_token_id=end, pad_token_id=pad)
            for row, (index, _, canonical) in enumerate(chunk):
                generated = out[row, width:].tolist()
                if end in generated:
                    generated = generated[:generated.index(end) + 1]
                try:
                    results[index] = self._report(requests[index][0], canonical, generated)
                except (ValueError, KeyError) as error:
                    results[index] = error
        return results


def generate(config, instruction, source_path=None, output='artifacts/generated.json', max_new_tokens=256):
    require_container()
    generator = CadGenerator(config)
    source_text = Path(source_path).read_text().strip() if source_path else None
    try:
        report = generator.predict(instruction, source_text, max_new_tokens)
    except (ValueError, KeyError) as e:
        write_json(output, {'instruction': instruction, 'status': 'invalid', 'error': str(e)})
        raise
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def evaluate_vq(config, max_pairs=100, aligned=False, constrained=True):
    require_container()
    device = device_for(config)
    path = Path(config['vq']['output']) / 'best.pt'
    vq, _ = load_vq(path, device)
    alignment = load_checkpoint(Path(config['alignment']['output']) / 'alignment.pt') if aligned else None
    if aligned and alignment['vq_sha256'] != file_hash(path):
        raise ValueError('Alignment/VQ checkpoint mismatch')
    dataset = PairDataset(config['data']['database'], 'val', max_pairs)
    exact, valid, total, correct, failures = 0, 0, 0, 0, []
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = dataset[i]
            batch = to_device(collate_pairs([item]), device)
            vectors, indices, mask = vq.encode(batch['tokens'], batch['groups'])
            memory = vectors[0][mask[0]]
            if aligned:
                memory = alignment['reconstructed_code_vectors'][indices[0][mask[0]].cpu()].to(device)
            try:
                predicted = vq.reconstruct(memory, constrained=constrained)
                split_pairs(predicted)
                valid += 1
                exact += int(predicted == item['tokens'])
                correct += sum(a == b for a, b in zip(predicted, item['tokens']))
                total += max(len(predicted), len(item['tokens']))
            except ValueError as e:
                total += len(item['tokens'])
                failures.append({'geometry_id': item['geometry_id'], 'ordinal': item['ordinal'], 'error': str(e)})
    report = {'pairs': len(dataset), 'exact_reconstruction': exact / len(dataset), 'syntax_validity': valid / len(dataset),
              'position_token_accuracy': correct / total, 'aligned': aligned, 'constrained': constrained,
              'failures': failures, 'scope': 'SEM reconstruction only; no 3D-kernel validity, Chamfer distance, or instruction score'}
    output = Path(config['vq']['output']) / f'evaluation_{"aligned" if aligned else "raw"}_{"fsa" if constrained else "unconstrained"}.json'
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report
