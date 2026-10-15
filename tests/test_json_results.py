import json
from pathlib import Path
import tempfile
import unittest

from json_results import write_json
from tools.repair_result_json import repair


def reject_constant(token):
    raise ValueError(token)


class StrictResultJsonTests(unittest.TestCase):
    def test_nested_nonfinite_values_and_literal_string(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.json'
            write_json(path, [{'loss': .125, 'gap': float('nan'), 'auc': float('inf'),
                               'negative': -float('inf'), 'literal': 'NaN', 'enabled': True}])
            result = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
            self.assertEqual(result, [{'loss': .125, 'gap': None, 'auc': None, 'negative': None,
                                       'literal': 'NaN', 'enabled': True}])

    def test_serialization_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.json'
            write_json(path, {'epoch': 1})
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                write_json(path, {'bad': object()})
            self.assertEqual(path.read_bytes(), original)

    def test_view_copy_preserves_original_and_skips_truncated_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'runs' / 'history.json'
            source.parent.mkdir()
            source.write_text('[{"loss":0.1,"gap":NaN,"text":"NaN"}]', encoding='utf-8')
            broken = root / 'history.json'
            broken.write_text('[{"epoch":', encoding='utf-8')
            original = source.read_bytes()
            result = repair(root)
            self.assertEqual(len(result['converted']), 1)
            self.assertEqual(len(result['skipped']), 1)
            self.assertEqual(source.read_bytes(), original)
            viewed = json.loads(Path(result['converted'][0]).read_text(encoding='utf-8'), parse_constant=reject_constant)
            self.assertEqual(viewed, [{'loss': .1, 'gap': None, 'text': 'NaN'}])
            self.assertEqual(len(repair(root)['converted']), 1)
