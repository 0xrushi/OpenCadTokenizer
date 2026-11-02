"""Prepare a lossless codec, exact round-trip audit and full prompt-length audit."""
import collections
import json
import os
from pathlib import Path
import random
import sqlite3

from .common import file_hash, read_config, require_container, write_json
from .instruction_data import instruction_tokens, native_sequence
from .lossless_sem import SCHEMA, LosslessSemCodec, install_codec
from .representation import stringify, tokenize


def length_summary(values, limit):
    ordered = sorted(values)
    if not ordered:
        return {'count':0}
    def percentile(q):
        return ordered[int((len(ordered)-1)*q)]
    return {'count':len(ordered),'minimum':ordered[0],'median':percentile(.5),
            'p95':percentile(.95),'p99':percentile(.99),'maximum':ordered[-1],
            'within_context':sum(value<=limit for value in ordered),
            'context_overflows':sum(value>limit for value in ordered)}


def lossless_sample(tokenizer, codec, instruction, source_pairs, target_pairs):
    prompt = instruction_tokens(tokenizer,codec.boundaries,instruction,source_pairs)
    target = native_sequence(target_pairs,codec.boundaries)+[codec.boundaries['end'],tokenizer.eos_token_id]
    return prompt+target,len(prompt)


def prepare_lossless(settings):
    require_container()
    from transformers import AutoTokenizer
    root = Path(settings['output'])
    root.mkdir(parents=True,exist_ok=True)
    signature = {'schema':SCHEMA,'settings':settings,'database_sha256':file_hash(settings['database']),
                 'base_tokenizer_sha256':{name:file_hash(Path(settings['tokenizer_path'])/name)
                                         for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')},
                 'code_sha256':{name:file_hash(Path('cadtokenizer')/name) for name in
                                ('lossless_sem.py','lossless_prepare.py','representation.py','instruction_data.py')},
                 'image_id':os.environ.get('CADTOKENIZER_IMAGE_ID')}
    path = root/'manifest.json'
    if path.exists() and read_config(path)!=signature:
        raise ValueError('Lossless audit provenance changed; choose a fresh output')
    write_json(path,signature)
    if (root/'state.json').exists() and read_config(root/'state.json').get('status')=='complete':
        return read_config(root/'results.json')
    try:
        write_json(root/'state.json',{'status':'running','phase':'tokenizer'})
        tokenizer = AutoTokenizer.from_pretrained(settings['tokenizer_path'],local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('Cached language tokenizer has no EOS')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        old_vocabulary_size = len(tokenizer)
        codec = install_codec(tokenizer)
        tokenizer.save_pretrained(root/'tokenizer')
        write_json(root/'codec.json',codec.to_dict())
        # A fresh load must preserve both token IDs and instruction tokenization.
        reloaded = AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
        if reloaded.encode('Task: CAD editing\n',add_special_tokens=False)!=tokenizer.encode('Task: CAD editing\n',add_special_tokens=False):
            raise ValueError('Saved tokenizer changed text tokenization')
        if install_codec(reloaded).to_dict()!=codec.to_dict():
            raise ValueError('Saved codec ID mapping changed')
        temporary = root/'tokens.sqlite.tmp'
        temporary.unlink(missing_ok=True)
        dest = sqlite3.connect(temporary)
        dest.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE geometry(id TEXT PRIMARY KEY,split TEXT,native_pairs TEXT);
            CREATE TABLE samples(id INTEGER PRIMARY KEY,example_id TEXT UNIQUE,split TEXT,task TEXT,
                                 input_ids TEXT,prompt_length INTEGER,context_overflow INTEGER);
            CREATE INDEX samples_split ON samples(split,task);
        ''')
        db = sqlite3.connect(settings['database'])
        geometries = db.execute("SELECT id,sequence,split FROM geometry WHERE split IN ('train','val') ORDER BY id").fetchall()
        encoded, splits = {},{}
        exact = collections.Counter()
        noncanonical = []
        write_json(root/'state.json',{'status':'running','phase':'round_trip','geometries':len(geometries)})
        for index,(key,sequence,split) in enumerate(geometries,1):
            pairs = codec.encode_pairs(sequence)
            decoded = codec.decode(codec.encode(sequence))
            if tokenize(decoded)!=tokenize(sequence):
                raise ValueError(f'Lossless semantic round-trip failed: {key}')
            if decoded!=sequence:
                noncanonical.append(key)
            if decoded!=stringify(tokenize(sequence)):
                raise ValueError('Decoded program is not canonical SEM')
            encoded[key],splits[key] = pairs,split
            exact[split]+=1
            dest.execute('INSERT INTO geometry VALUES(?,?,?)',(key,split,json.dumps(pairs)))
            if index%10000==0:
                print(json.dumps({'phase':'round_trip','completed':index,'total':len(geometries)}),flush=True)
        examples = db.execute("SELECT id,split,task,instruction,source_id,target_id FROM examples WHERE split IN ('train','val') ORDER BY id").fetchall()
        generations = [key for key,split,task,*_ in examples if split=='train' and task=='generation']
        edits = [key for key,split,task,*_ in examples if split=='train' and task=='editing']
        if not generations or not edits:
            raise ValueError('Need both tasks in training')
        random.Random(settings['seed']).shuffle(edits)
        selected = set(generations+edits[:int(len(generations)*settings['editing_per_generation'])])
        lengths,prompts,output_lengths = collections.defaultdict(list),collections.defaultdict(list),collections.defaultdict(list)
        selected_lengths = collections.defaultdict(list)
        dropped,empty_instructions = collections.Counter(),collections.Counter()
        overflow_records = []
        write_json(root/'state.json',{'status':'running','phase':'full_prompt_lengths','examples':len(examples)})
        for index,(key,split,task,instruction,source_id,target_id) in enumerate(examples,1):
            if task not in ('generation','editing') or (task=='editing' and source_id is None):
                raise ValueError(f'Invalid task/source: {key}')
            endpoint_ids = [target_id]+([source_id] if source_id else [])
            if any(endpoint not in encoded or splits[endpoint]!=split for endpoint in endpoint_ids):
                raise ValueError(f'Example endpoints do not belong to the same split: {key}')
            source_pairs = encoded[source_id] if source_id else None
            ids,prefix = lossless_sample(tokenizer,codec,instruction,source_pairs,encoded[target_id])
            bucket = f'{split}:{task}'
            lengths[bucket].append(len(ids));prompts[bucket].append(prefix);output_lengths[bucket].append(len(ids)-prefix)
            if not instruction.strip():
                empty_instructions[bucket]+=1
            overflow = len(ids)>settings['context_length']
            if overflow:
                overflow_records.append({'id':key,'split':split,'task':task,'total_tokens':len(ids),
                                         'prompt_tokens':prefix,'output_tokens':len(ids)-prefix})
            if split=='train' and key not in selected:
                dropped[bucket+':mixture_not_selected']+=1
            else:
                # Keep overflows explicitly flagged; never truncate geometry.
                dest.execute('INSERT INTO samples(example_id,split,task,input_ids,prompt_length,context_overflow) VALUES(?,?,?,?,?,?)',
                             (key,split,task,json.dumps(ids),prefix,int(overflow)))
                selected_lengths[bucket].append(len(ids))
            if index%10000==0:
                print(json.dumps({'phase':'full_prompt_lengths','completed':index,'total':len(examples)}),flush=True)
        db.close()
        limit = settings['context_length']
        result = {'schema':SCHEMA,'round_trip':{'exact_semantic_geometry_by_split':dict(exact),
                  'noncanonical_input_strings':len(noncanonical),'test_cases_accessed':0},
                  'vocabulary':{'base_size':old_vocabulary_size,'size_with_codec':len(tokenizer),
                                'new_tokens':len(tokenizer)-old_vocabulary_size,'semantic_symbols':len(codec.native_ids),
                                'frames':len(codec.boundaries)},
                  'context_length':limit,
                  'all_examples':{key:length_summary(values,limit) for key,values in lengths.items()},
                  'prompt_only':{key:length_summary(values,limit) for key,values in prompts.items()},
                  'completion_only':{key:length_summary(values,limit) for key,values in output_lengths.items()},
                  'selected_training_mixture_and_validation':{key:length_summary(values,limit) for key,values in selected_lengths.items()},
                  'mixture_exclusions':dict(dropped),'empty_instructions':dict(empty_instructions),
                  'prompt_format':'Existing instruction_tokens: BOS, task/instruction text, optional framed source, output header and CAD-start. Full length includes target pairs, CAD-end and LM EOS. No chat template is used by the existing base-model pipeline.',
                  'limits':['Exactness preserves already-discretized SEM, not pre-quantization geometry.',
                            'Round-trip fidelity is not an end-to-end generation result. No model weights were loaded.',
                            'The prepared pool has upstream filtering; counts do not cover rejected raw examples.',
                            'New token embeddings and output rows must be initialized and trained during a future SFT pilot. Existing VQ adapters cannot be reused as-is.',
                            'tokens.sqlite is a separate codec cache, not compatible with the current alignment-dependent train-sft command. Context overflows are retained and flagged.']}
        metadata = {'signature':signature,'codec':codec.to_dict(),'context_length':limit,
                    'prompt_format':result['prompt_format'],'training_mixture_seed':settings['seed']}
        dest.executemany('INSERT INTO metadata VALUES(?,?)',[(key,json.dumps(value)) for key,value in metadata.items()])
        dest.commit();dest.close()
        temporary.replace(root/'tokens.sqlite')
        write_json(root/'results.json',result)
        write_json(root/'context_overflows.json',overflow_records)
        write_json(root/'noncanonical_input_ids.json',noncanonical)
        lines = ['# Lossless structured SEM codec: round-trip and context audit','',
                 f'Exact semantic round-trip: {sum(exact.values())} train/validation geometries. No test cases, CAD kernel, VQ model or LLM weights were used.', '',
                 f'{len(codec.native_ids)} semantic symbols + 3 frames; vocabulary {old_vocabulary_size} → {len(tokenizer)}. Each CAD symbol is atomic in the saved tokenizer.', '',
                 '| Split/task | All examples | Median complete tokens | p95 | p99 | Maximum | Over context |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for key,row in result['all_examples'].items():
            lines.append(f"| {key} | {row['count']} | {row['median']} | {row['p95']} | {row['p99']} | {row['maximum']} | {row['context_overflows']} |")
        lines += ['',result['prompt_format'],'',f'Context limit: {limit}. These are exact tokenizer lengths; no truncation or estimates. All examples are audited before 5:1 training-mixture selection. Cache mixture counts and any exclusions are in results.json.', '',*result['limits']]
        (root/'findings.md').write_text('\n'.join(lines)+'\n')
        write_json(root/'state.json',{'status':'complete','results':str(root/'results.json')})
        print(json.dumps(result),flush=True)
        return result
    except Exception as exc:
        write_json(root/'state.json',{'status':'failed','error':repr(exc)})
        raise
