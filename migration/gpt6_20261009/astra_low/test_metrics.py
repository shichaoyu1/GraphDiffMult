"""Run with: python -B test_metrics.py (standard library only)."""
import ast
from pathlib import Path
import unittest

from metrics import masked_retrieval_metrics as measure


class MetricsTests(unittest.TestCase):
    def test_mask_before_cutoff_and_missing_positive(self):
        self.assertEqual(measure(['unknown', 'a', 'n', 'b'], {'a', 'b', 'absent'}, {'n'}, 2),
                         dict(hit_at_k=1.0, recall_at_k=1/3, reciprocal_rank=1.0))

    def test_rr_uses_full_masked_ranking(self):
        self.assertEqual(measure(['u', 'n', 'a'], {'a'}, {'n'}, 1),
                         dict(hit_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.5))

    def test_large_k(self):
        self.assertEqual(measure(['a', 'b'], ['a', 'b'], [], 99)['recall_at_k'], 1)

    def test_no_ranked_positive(self):
        for ranked in ([], ['u'], ['n']):
            self.assertEqual(measure(ranked, {'a'}, {'n'}, 1),
                             dict(hit_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.0))

    def test_no_positive(self):
        self.assertEqual(measure(['n', 'u'], [], ['n'], 1),
                         dict(hit_at_k=None, recall_at_k=None, reciprocal_rank=None))

    def test_overlap(self):
        with self.assertRaises(ValueError):
            measure([], ['a'], ['a'], 1)

    def test_duplicates_including_unknown(self):
        for ranked in (['a', 'a'], ['u', 'u']):
            with self.assertRaises(ValueError):
                measure(ranked, [], [], 1)

    def test_invalid_k_even_without_positives(self):
        for k in (0, -1, 1.0, True, False, '1', None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                measure([], [], [], k)

    def test_iterables(self):
        self.assertEqual(measure(iter(['a']), iter(['a', 'a']), iter([]), 1)['recall_at_k'], 1)


class SyntheticSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Extract only pure label-building definitions; never import training or
        # load patient data. This verifies source rules using synthetic metadata.
        path = Path(__file__).absolute().parents[3] / 'train_semantic_alignment.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        names = {'clean_value', 'canonical_field', 'anchor_source', 'anchor_type',
                 'make_anchor', 'semantic_anchors', 'target_anchor_keys'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        cls.scope = dict(PATHOLOGY_FIELDS=('Tumor Grade', 'Tumor Type'),
                         MOLECULAR_FIELDS=('IDH', 'MGMT', '1p19Q CODEL'),
                         CLINICAL_FIELDS=('Age at Histological Diagnosis', 'Gender'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), cls.scope)

    def test_region_rules(self):
        metadata = {'Tumor Grade': '4', 'Tumor Type': 'Glioblastoma', 'IDH': 'wildtype',
                    'MGMT': 'unknown', '1p19Q CODEL': 'non-codeleted'}
        g, t, i, c = ('tumor_grade::4', 'tumor_type::glioblastoma',
                      'idh::wildtype', '1p19q_codel::non-codeleted')
        expected = {'enhancing': [g, t], 'edema': [i, t, g], 'necrotic': [g, c, t]}
        f = self.scope['target_anchor_keys']
        for region, keys in expected.items():
            self.assertEqual(f(metadata, region, 'region_rules'), keys)
        self.assertEqual(f(metadata, 'enhancing', 'region_rules', include_pathology=False), [i, c])
        self.assertEqual(f(metadata, 'edema', 'region_rules', include_pathology=False), [i])
        self.assertEqual(f(metadata, 'necrotic', 'region_rules', include_pathology=False), [c])
        self.assertEqual(set(f(metadata, 'enhancing', 'all_patient_anchors')), {g, t, i, c})


if __name__ == '__main__':
    unittest.main(verbosity=2)
