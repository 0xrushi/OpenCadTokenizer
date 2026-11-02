from __future__ import annotations

import copy
import gc
from pathlib import Path

from .common import require_container, write_json


def smoke(base_config, directory='artifacts/smoke'):
    require_container()
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from .prepare import prepare
    from .vq import train_vq
    from .alignment import train_alignment
    from .encode import encode_geometry
    from .sft import train_sft
    from .inference import evaluate_vq, generate
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f'{root} already exists; choose another --directory for a fresh smoke run')
    root.mkdir(parents=True)
    config = copy.deepcopy(base_config)
    config['device'] = 'cpu'
    config['data']['database'] = str(root / 'prepared.sqlite')
    config['data']['validation_fraction'] = 0.1
    config['vq'].update(output=str(root / 'vq'), width=32, heads=4, encoder_layers=1,
                        decoder_layers=1, feedforward=64, dropout=0.0, codebook_size=16,
                        reconstruction_loss='cross_entropy', epochs=1, batch_size=8, num_workers=0)
    config['alignment'].update(output=str(root / 'alignment'), model_name=str(root / 'tiny_llm'),
                              epochs=1, batch_size=8, refinement_epochs=1, candidate_count=128,
                              collision_policy='unique_topk')
    config['sft'].update(output=str(root / 'sft'), encoded_database=str(root / 'encoded.sqlite'),
                        context_length=512, editing_per_generation=1, batch_size=2,
                        gradient_accumulation_steps=1, epochs=1, max_steps=2,
                        gradient_checkpointing=False, lora_rank=4, lora_alpha=4, logging_steps=1)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(config['seed'])
    names = ['<unk>', '<s>', '</s>', '<pad>'] + [f'<|reserved_special_token_{i}|>' for i in range(3)]
    names += ['Task', 'CAD', 'generation', 'editing', 'Instruction', 'Original', 'Output', ':']
    names += [f'word_{i}' for i in range(512 - len(names))]
    raw_tokenizer = Tokenizer(WordLevel({name: i for i, name in enumerate(names)}, unk_token='<unk>'))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, unk_token='<unk>', bos_token='<s>',
                                        eos_token='</s>', pad_token='<pad>', additional_special_tokens=names[4:7])
    tokenizer.save_pretrained(root / 'tiny_llm')
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=2))
    tiny.save_pretrained(root / 'tiny_llm')
    del tiny
    write_json(root / 'config.json', config)
    prepare(config, limit=24)
    train_vq(config)
    gc.collect()
    train_alignment(config)
    gc.collect()
    encode_geometry(config)
    train_sft(config)
    gc.collect()
    evaluate_vq(config, max_pairs=3, aligned=True)
    generate(config, 'Create a rectangular prism.', output=str(root / 'generated.json'), max_new_tokens=16)
    write_json(root / 'smoke_result.json', {'status': 'passed', 'scope': 'CPU wiring test with random tiny LLM; not a trained CAD model or quality benchmark'})
