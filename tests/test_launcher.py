import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherTests(unittest.TestCase):
    def run_launcher(self, args, extra_env=None, eval_environment=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            shutil.copy('scripts/cad.sh', root / 'scripts/cad.sh')
            (root / 'bin').mkdir()
            if eval_environment:
                evaluation = root / '.runtime/eval-venv'
                evaluation.mkdir(parents=True)
                (evaluation / 'image-id').write_text('sha256:existing\n')
            fake = root / 'bin/docker'
            fake.write_text('''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(os.environ['DOCKER_TEST_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')
if args[:2]==['image','ls']: print('unsloth/unsloth:cached abc123')
elif args[:2]==['image','inspect']:
 print('sha256:existing' if args[3]=='{{.Id}}' else '["unsloth/unsloth:cached"]')
''')
            fake.chmod(0o755)
            env = os.environ.copy()
            env.pop('UNSLOTH_IMAGE', None)
            env.pop('CAD_NPROC', None)
            env.update(PATH=f'{root / "bin"}:{env["PATH"]}', DOCKER_TEST_LOG=str(root / 'calls.jsonl'))
            env.update(extra_env or {})
            result = subprocess.run(['bash', str(root / 'scripts/cad.sh'), *args], env=env, text=True, capture_output=True)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            return result, calls

    def test_uses_cached_image_and_does_not_pull(self):
        result, calls = self.run_launcher(['train-vq', '--config', 'path with spaces.json'], {'CAD_CPU': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        run = calls[-1]
        self.assertEqual(run[0], 'run')
        self.assertIn('--pull=never', run)
        self.assertIn('sha256:existing', run)
        self.assertNotIn('--gpus', run)
        self.assertEqual(run[-5:], ['python', '-m', 'cadtokenizer', 'train-vq', '--config', 'path with spaces.json'][-5:])

    def test_multi_gpu_uses_torchrun_in_same_image(self):
        result, calls = self.run_launcher(['train-sft'], {'CAD_NPROC': '4'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('torchrun', calls[-1])
        self.assertIn('--nproc-per-node=4', calls[-1])
        self.assertIn('--pull=never', calls[-1])

    def test_tensorboard_uses_cached_image_and_local_port_without_gpu(self):
        result, calls = self.run_launcher(['tensorboard', '--logdir', 'artifacts/sft/tensorboard'],
                                          {'CAD_TENSORBOARD_PORT': '16006'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('tensorboard.main', calls[-1])
        self.assertIn('127.0.0.1:16006:6006', calls[-1])
        self.assertNotIn('--gpus', calls[-1])
        self.assertIn('--pull=never', calls[-1])
        self.assertEqual(calls[-1][-2:], ['--logdir', 'artifacts/sft/tensorboard'])

    def test_tensorboard_rejects_invalid_port(self):
        result, calls = self.run_launcher(['tensorboard'], {'CAD_TENSORBOARD_PORT': '70000'})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'run' for call in calls))

    def test_rejects_multi_process_vq(self):
        result, calls = self.run_launcher(['train-vq'], {'CAD_NPROC': '2'})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'run' for call in calls))

    def test_geometry_environment_runs_inside_the_same_cached_image(self):
        result, calls = self.run_launcher(['benchmark-score'], {'CAD_CPU': '1'}, eval_environment=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/workspace/.runtime/eval-venv/bin/python', calls[-1])
        self.assertIn('sha256:existing', calls[-1])
        self.assertIn('--pull=never', calls[-1])
        result, calls = self.run_launcher(['benchmark-predict'], {'CAD_CPU': '1'}, eval_environment=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('/workspace/.runtime/eval-venv/bin/python', calls[-1])
