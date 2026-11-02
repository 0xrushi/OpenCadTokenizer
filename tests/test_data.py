import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cadtokenizer.common import file_hash
from cadtokenizer.instruction_data import instruction_tokens, native_sequence, prepare_sft_tokens
from cadtokenizer.prepare import prepare
from tests.fixtures import config_for, make_sources


class FakeTokenizer:
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        return [7] * len(text.split())


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = config_for(self.root)
        make_sources(self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_linked_edits_do_not_cross_holdouts(self):
        with sqlite3.connect(self.config['data']['database']) as db:
            rows = db.execute('SELECT s.split,t.split,e.split FROM examples e JOIN geometry s ON s.id=e.source_id JOIN geometry t ON t.id=e.target_id').fetchall()
            self.assertTrue(rows)
            self.assertTrue(all(a == b == c for a, b, c in rows))
            linked = db.execute('SELECT split FROM examples WHERE id="cad_editor:train:2"').fetchone()[0]
            self.assertEqual(linked, 'val')
            leakage = db.execute('SELECT COUNT(*) FROM geometry WHERE split!="train" AND tokenizer_split="train"').fetchone()[0]
            self.assertEqual(leakage, 0)

    def test_caption_uses_upstream_index(self):
        with sqlite3.connect(self.config['data']['database']) as db:
            text = db.execute('SELECT g.sequence FROM examples e JOIN geometry g ON g.id=e.target_id WHERE e.id="cadfusion:train:0"').fetchone()[0]
            self.assertTrue(text.startswith('circle,0,10'))

    def test_existing_database_is_preserved(self):
        old_hash = file_hash(self.config['data']['database'])
        with self.assertRaises(FileExistsError):
            prepare(self.config)
        self.assertEqual(file_hash(self.config['data']['database']), old_hash)

    def test_native_ids_are_never_retokenized(self):
        tokenizer = FakeTokenizer()
        boundary = {'start': 90, 'pair_end': 91, 'end': 92}
        source = [[101, 102], [103]]
        ids = instruction_tokens(tokenizer, boundary, 'Edit it.', source)
        expected = [90, 101, 102, 91, 103, 91, 92]
        self.assertTrue(any(ids[i:i+len(expected)] == expected for i in range(len(ids))))
        self.assertEqual(native_sequence(source, boundary), [101, 102, 91, 103, 91])

    def test_sft_cache_keeps_framing_and_checks_fingerprints(self):
        alignment_dir = Path(self.config['alignment']['output'])
        alignment_dir.mkdir()
        (alignment_dir / 'alignment.pt').write_bytes(b'test fingerprint fixture')
        alignment = {'boundaries': {'start': 90, 'pair_end': 91, 'end': 92}}
        with sqlite3.connect(self.config['sft']['encoded_database']) as encoded:
            encoded.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE geometry(id TEXT PRIMARY KEY,native_pairs TEXT);')
            with sqlite3.connect(self.config['data']['database']) as prepared:
                for (key,) in prepared.execute('SELECT id FROM geometry'):
                    encoded.execute('INSERT INTO geometry VALUES(?,?)', (key, json.dumps([[101, 102, 103, 104]])))
            encoded.executemany('INSERT INTO metadata VALUES(?,?)', [('alignment_sha256', json.dumps(file_hash(alignment_dir / 'alignment.pt'))),
                                                                      ('database_sha256', json.dumps(file_hash(self.config['data']['database'])))])
        with contextlib.redirect_stdout(io.StringIO()):
            path = prepare_sft_tokens(self.config, FakeTokenizer(), alignment)
        with sqlite3.connect(path) as db:
            rows = db.execute('SELECT input_ids,prompt_length FROM samples').fetchall()
            self.assertTrue(rows)
            for raw, prefix in rows:
                ids = json.loads(raw)
                self.assertEqual(ids[prefix-1], 90)
                self.assertEqual(ids[prefix:], [101, 102, 103, 104, 91, 92, 2])
        (alignment_dir / 'alignment.pt').write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            prepare_sft_tokens(self.config, FakeTokenizer(), alignment)
