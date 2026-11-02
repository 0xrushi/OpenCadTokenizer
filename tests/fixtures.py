import json
import pickle
from pathlib import Path


def skexgen_item(offset):
    # Four-point circle layout includes a separate curve terminator coordinate.
    coords = [[offset, 10], [offset + 2, 12], [offset + 4, 10], [offset + 2, 8]]
    xy = [[x + 4, y + 4] for x, y in coords] + [[0, 0]] * 4
    # B is in slot 14; SEM serialization moves it to the front.
    ext = [1, 64, 32, 32, 32] + [3, 2, 2, 2, 3, 2, 2, 2, 3] + [1, 4, 31, 31, 0]
    return {'name': f'geometry-{offset}', 'num_se': 1, 'se_xy': [xy],
            'se_cmd': [[5, 2, 1, 0]], 'se_ext': [ext]}


def config_for(root):
    config = json.loads(Path('configs/train.json').read_text())
    config['device'] = 'cpu'
    config['data'].update(root=str(root / 'raw'), database=str(root / 'prepared.sqlite'),
                          validation_fraction=0, tokenizer_fraction=1)
    config['alignment']['output'] = str(root / 'alignment')
    config['sft'].update(output=str(root / 'sft'), encoded_database=str(root / 'encoded.sqlite'),
                         context_length=1024)
    config['vq'].update(output=str(root / 'vq'), width=16, heads=4, feedforward=32,
                        encoder_layers=1, decoder_layers=1, codebook_size=8, dropout=0.0)
    return config


def make_sources(config):
    from cadtokenizer.representation import skexgen_to_sem
    root = Path(config['data']['root'])
    skex = root / 'skexgen/cad_data'
    skex.mkdir(parents=True)
    for filename, data in [('train_deduplicate_s.pkl', [skexgen_item(i) for i in range(20)]),
                           ('val.pkl', [skexgen_item(30)]), ('test.pkl', [skexgen_item(40)]),
                           ('train_invalid.pkl', [])]:
        with (skex / filename).open('wb') as f:
            pickle.dump(data, f)
    captions = root / 'cadfusion'
    captions.mkdir()
    for split, indices in [('train', [0, 3, 5]), ('val', [1]), ('test', [2])]:
        (captions / f'{split}.json').write_text(json.dumps([{'serial_num': i, 'description': f'A circle at {i}.'} for i in indices]))
    editor = root / 'cad_editor/dataset'
    editor.mkdir(parents=True)
    def edit(a, b):
        return {'original_sequence': skexgen_to_sem(skexgen_item(a)),
                'edited_sequence': skexgen_to_sem(skexgen_item(b)), 'instruction': 'Move the circle.'}
    (editor / 'train.json').write_text(json.dumps([edit(3, 4), edit(6, 7), edit(1, 10)]))
    (editor / 'test.json').write_text(json.dumps([edit(8, 9)]))
