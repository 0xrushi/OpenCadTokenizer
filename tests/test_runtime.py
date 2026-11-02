import unittest
from unittest.mock import patch

from cadtokenizer.common import model_runtime_available, require_container


class RuntimeTests(unittest.TestCase):
    def test_vast_requires_explicit_runtime_and_instance_marker(self):
        with patch('cadtokenizer.common.Path.exists', return_value=True):
            with patch.dict('os.environ', {'CADTOKENIZER_RUNTIME': 'vastai'}, clear=True):
                self.assertFalse(model_runtime_available())
            with patch.dict('os.environ', {'VAST_CONTAINERLABEL': 'test'}, clear=True):
                self.assertFalse(model_runtime_available())
            with patch.dict('os.environ', {'CADTOKENIZER_RUNTIME': 'vastai', 'VAST_CONTAINERLABEL': 'test'}, clear=True):
                self.assertTrue(model_runtime_available())
                require_container()

    def test_local_host_cannot_enable_model_execution_with_only_runtime_flag(self):
        with patch('cadtokenizer.common.Path.exists', return_value=False), \
                patch.dict('os.environ', {'CADTOKENIZER_CONTAINER': 'unsloth',
                                         'CADTOKENIZER_RUNTIME': 'vastai', 'VAST_CONTAINERLABEL': 'test'}, clear=True):
            with self.assertRaises(RuntimeError):
                require_container()

    def test_cached_unsloth_runtime_still_allowed(self):
        with patch('cadtokenizer.common.Path.exists', return_value=True), \
                patch.dict('os.environ', {'CADTOKENIZER_CONTAINER': 'unsloth'}, clear=True):
            self.assertTrue(model_runtime_available())
