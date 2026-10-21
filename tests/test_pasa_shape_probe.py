"""Synthetic checks for the frozen CPU shape probe; no clinical data in tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import run_pasa_shape_probe as probe


class ShapeProbeTests(unittest.TestCase):
    def record(self, counts=(1, 2, 0, 0, 0)):
        return {'audit': {'slab_voxel_counts_after_resize': dict(zip(probe.LABELS, counts))},
            'region_present': [n > 0 for n in counts],
            'patient_id': 'must-not-be-a-feature', 'path': '/private/never/open',
            'image_files': {'t1': {'path': '/does/not/exist', 'mtime_ns': 999}},
            'source': 'not-a-feature', 'raw_3d_volume': 1e12}

    def test_exact_eleven_features_ignore_identifiers_paths_and_intensities(self):
        record = self.record()
        expected = [np.log(4), 1 / 3, 2 / 3, 0, 0, 0, 1, 1, 0, 0, 0]
        np.testing.assert_allclose(probe.feature_row(record, 1, 2), expected)
        changed = copy.deepcopy(record)
        changed.update(patient_id='different', path=None, source=object(), image_files=None)
        changed['audit']['raw_label_counts'] = {'1': 1000000000}
        changed['intensity'] = np.nan
        np.testing.assert_array_equal(probe.feature_row(changed, 1, 2), expected)

    def test_empty_noninteger_negative_nonfinite_overfull_and_presence_rejected(self):
        for counts in ((0, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0.5, 2, 0, 0, 0),
                (np.nan, 2, 0, 0, 0), (5, 0, 0, 0, 0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                probe.feature_row(self.record(counts), 1, 2)
        record = self.record()
        record['region_present'][0] = False
        with self.assertRaisesRegex(ValueError, 'Presence'):
            probe.feature_row(record, 1, 2)
        record = self.record()
        record['audit']['slab_voxel_counts_after_resize']['6'] = 1
        with self.assertRaisesRegex(ValueError, 'five'):
            probe.feature_row(record, 1, 2)

    def vocab_and_records(self):
        vocab = [{'key': key, 'field': field}
            for field in probe.FIELDS for key in reversed(probe.CLASS_KEYS[field])]
        # Reverse candidate order deliberately: labels must follow keys, never fixed indices.
        records = {'subject_ids': ['p'] * 3, 'field_names': list(probe.FIELDS),
            'target_ids': [[1], [], [4]], 'valid_ids': [[0, 1], [], [4, 5]],
            'positive_counts': [1, 0, 1]}
        return vocab, records

    def test_recovered_labels_use_canonical_keys_and_preserve_missing(self):
        vocab, records = self.vocab_and_records()
        labels = probe.recover_labels(records, vocab, ['p'])
        self.assertEqual(labels, {'p': {'IDH': 1, 'MGMT': None, '1p19Q CODEL': 0}})
        malformed = copy.deepcopy(records)
        malformed['valid_ids'][1] = [2, 3]
        with self.assertRaisesRegex(ValueError, 'Missing'):
            probe.recover_labels(malformed, vocab, ['p'])
        malformed = copy.deepcopy(records)
        malformed['target_ids'][0] = [-1]
        with self.assertRaises(ValueError):
            probe.recover_labels(malformed, vocab, ['p'])
        with self.assertRaises(ValueError):
            probe.recover_labels(records, vocab, ['other'])
        malformed = copy.deepcopy(records)
        malformed['field_names'][1] = 'IDH'
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            probe.recover_labels(malformed, vocab, ['p'])

    def test_original_tsv_and_alignment_normalization_without_mri_discovery(self):
        from dataset import load_utsw_metadata
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / 'synthetic.tsv'
            metadata_path.write_text('Subject ID\tIDH\tMGMT\t1p19Q CODEL\n'
                'train\t  Mutated  \tunknown\tNON CO-DELETED\n'
                'unrequested\tmutant\tnew invalid class\tcodeleted\n', encoding='utf-8-sig')
            metadata = load_utsw_metadata(str(metadata_path))
            self.assertEqual(probe.labels_from_metadata(metadata, ['train']),
                {'train': {'IDH': 1, 'MGMT': None, '1p19Q CODEL': 0}})
            with self.assertRaisesRegex(ValueError, 'outside frozen'):
                probe.labels_from_metadata(metadata, ['unrequested'])
            with self.assertRaisesRegex(ValueError, 'missing'):
                probe.labels_from_metadata(metadata, ['missing'])

    def test_reference_five_hash_binding_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference'
            reference.mkdir()
            for name in probe.REFERENCE_FILES:
                (reference / name).write_text('{}', encoding='utf-8')
            binding = Path(directory) / 'binding.json'
            binding.write_text(json.dumps({'identity': {'version': 'pasa_repair_repeats_v1',
                'reference_sha256': {name: probe.digest(reference / name) for name in probe.REFERENCE_FILES}}}), encoding='utf-8')
            probe.validate_reference_binding(reference, binding)
            (reference / 'inputs.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                probe.validate_reference_binding(reference, binding)

    def test_scaler_train_only_and_missing_field_not_fitted(self):
        x = np.array([[-1.0], [1.0], [-2.0], [100.0]])
        ids = list('abcd')
        labels = {sid: {field: value for field in probe.FIELDS}
            for sid, value in zip(ids, (0, 1, 0, None))}
        scaler, models, fitted = probe.fit_field_models(x, ids, labels)
        self.assertEqual(scaler.n_samples_seen_, 4)
        np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
        for field in probe.FIELDS:
            self.assertEqual(fitted[field]['fitted_subject_ids'], list('abc'))
            self.assertEqual(fitted[field]['patients'], 3)
            self.assertEqual(fitted[field]['missing'], 1)
        saved_mean = scaler.mean_.copy()
        saved_coef = models['IDH'].coef_.copy()
        transformed_val = scaler.transform([[1e8]])
        models['IDH'].predict(transformed_val)
        np.testing.assert_array_equal(scaler.mean_, saved_mean)
        np.testing.assert_array_equal(models['IDH'].coef_, saved_coef)

    def test_default_zero_margin_predicts_negative_and_auc_ties_half(self):
        labels = {sid: {field: label for field in probe.FIELDS}
            for sid, label in (('a', 0), ('b', 1))}
        _, models, _ = probe.fit_field_models(np.array([[-1.0], [1.0]]), ['a', 'b'], labels)
        # Construct an exact-zero synthetic margin; optimizer roundoff is not a tie.
        models['IDH'].intercept_.fill(0.0)
        self.assertEqual(models['IDH'].decision_function([[0.0]])[0], 0)
        self.assertEqual(models['IDH'].predict([[0.0]])[0], 0)
        metrics = probe.classification_metrics([0, 1], [0, 0], [0, 0])
        self.assertEqual(metrics['auroc'], 0.5)
        self.assertEqual(metrics['positive_recall'], 0)
        self.assertEqual(metrics['negative_recall'], 1)
        self.assertEqual(metrics['balanced_accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
