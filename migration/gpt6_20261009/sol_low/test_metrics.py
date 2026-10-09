"""Run directly: python -B test_metrics.py. No patient data or third party imports."""
import unittest
from metrics import masked_retrieval_metrics as metrics


class MaskedMetricsTests(unittest.TestCase):
    def test_unknowns_removed_before_ranking(self):
        self.assertEqual(metrics(['unknown', 'n', 'p'], ['p'], ['n'], 1),
                         dict(hit_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.5))
        self.assertEqual(metrics(['unknown', 'p', 'n'], ['p'], ['n'], 1),
                         dict(hit_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0))

    def test_multiple_positives_and_missing_denominator(self):
        self.assertEqual(metrics(['p1', 'n', 'p2'], ['p1', 'p2', 'missing'], ['n'], 2),
                         dict(hit_at_k=1.0, recall_at_k=1 / 3, reciprocal_rank=1.0))
        self.assertEqual(metrics(['p1', 'n', 'p2'], ['p1', 'p2', 'missing'], ['n'], 99)['recall_at_k'], 2 / 3)

    def test_no_positive_labels(self):
        self.assertEqual(metrics(['n'], [], ['n'], 1),
                         dict(hit_at_k=None, recall_at_k=None, reciprocal_rank=None))

    def test_no_ranked_positive(self):
        for ranking in ([], ['n'], ['unknown']):
            self.assertEqual(metrics(ranking, ['p'], ['n'], 2),
                             dict(hit_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.0))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            metrics([], ['p'], ['p'], 1)

    def test_duplicate_known_and_unknown_rejected(self):
        for ranking in (['p', 'p'], ['u', 'u']):
            with self.assertRaises(ValueError):
                metrics(ranking, ['p'], [], 1)
        with self.assertRaises(ValueError):
            metrics(['u', 'u'], [], [], 1)

    def test_invalid_k_rejected(self):
        for k in (0, -1, 1.0, '1', True, False, None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                metrics([], [], [], k)

    def test_iterables_and_set_label_semantics(self):
        self.assertEqual(metrics(iter(['n', 'p']), ['p', 'p'], iter(['n']), 2),
                         dict(hit_at_k=1.0, recall_at_k=1.0, reciprocal_rank=0.5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
