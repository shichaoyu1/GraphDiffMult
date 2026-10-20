"""Tiny real-NIfTI smoke of the four-arm train/validation-only pilot."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from pasa_repair_pilot import METHODS, VERSION
from tools import run_pasa_repair_pilot as runner


class RepairRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_nifti_four_arm_smoke_preserves_common_inputs_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = root / 'metadata.tsv'
            metadata.write_text('synthetic metadata; discovery is mocked\n', encoding='utf-8')
            reference, audit, output = root / 'reference', root / 'audit', root / 'new_pilot'
            reference.mkdir()
            audit.mkdir()
            splits = {'train': ['train_a', 'train_b', 'empty'],
                      'val': ['val_a', 'val_b'], 'test': ['test_only']}
            runner.write_json(reference / 'splits.json', splits)
            runner.write_json(reference / 'prepared.json', {'identity': {
                'splits': splits, 'metadata_sha256': runner.digest(metadata), 'sample_seed': 17}})

            cases, audit_rows = [], []
            expected_images = {}
            # The test patient is present for split-overlap validation, but has no
            # files. Its unique anchor must never enter the training vocabulary.
            test_dir = root / 'DO_NOT_READ_TEST'
            test_case = {'subject_id': 'test_only', 'label': 0, 'patient_dir': str(test_dir),
                         'metadata': {'IDH': 'test_only_status', 'Tumor Type': 'test_only_type'}}
            rng = np.random.default_rng(98)
            for split in ('train', 'val'):
                for index, sid in enumerate(splits[split]):
                    case_dir = root / sid
                    case_dir.mkdir()
                    seg = np.zeros((20, 20, 5), dtype=np.uint8)
                    if sid != 'empty':
                        # A label-3 lesion is near the volume boundary, where the
                        # former fixed middle slab would miss it. A second raw
                        # bin makes nonempty-region pooling exercise real masks.
                        seg[3:10, 4:12, 0] = 3
                        if index == 1:
                            seg[11:16, 10:17, 0] = 5
                    seg_path = case_dir / 'rtumorseg_manual_correction.nii.gz'
                    nib.save(nib.Nifti1Image(seg, np.eye(4)), seg_path)
                    if sid != 'empty':
                        expected_images[sid] = {}
                        for modality in ('t1', 't1ce', 't2', 'flair'):
                            image_path = case_dir / ('brain_' + modality + '.nii.gz')
                            image = rng.uniform(1, 20, size=seg.shape).astype(np.float32)
                            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
                            expected_images[sid][modality] = image_path
                    cases.append({'subject_id': sid, 'label': 0, 'patient_dir': str(case_dir),
                                  'metadata': {'IDH': 'mutant' if index == 1 else 'wildtype',
                                               'Tumor Grade': '2' if index == 1 else '4'}})
                    audit_rows.append({'subject_id': sid, 'split': split,
                                       'segmentation_file': str(seg_path),
                                       'segmentation_sha256': runner.digest(seg_path)})
            cases.append(test_case)
            runner.write_json(audit / 'records.json', audit_rows)

            original_load = nib.load
            loaded = []

            def guarded_load(filename, *args, **kwargs):
                path = Path(filename)
                self.assertNotIn(test_dir, path.parents, 'Test imaging was accessed')
                loaded.append(path)
                return original_load(filename, *args, **kwargs)

            cli = ['run_pasa_repair_pilot', '--stage', 'smoke', '--reference-campaign', str(reference),
                   '--audit', str(audit), '--data-root', str(root), '--metadata', str(metadata),
                   '--out', str(output), '--cpu', '--roi-size', '16', '--z-slices', '1',
                   '--batch-size', '2', '--workers', '0', '--smoke-patients', '2']
            with patch.object(runner.base, 'discover_semantic_cases', return_value=cases), \
                    patch.object(nib, 'load', side_effect=guarded_load), patch.object(sys, 'argv', cli):
                runner.main()

            protocol = runner.read(output / 'protocol.json')
            manifest = runner.read(output / 'inputs.json')
            vocab = runner.read(output / 'anchor_vocab.json')
            self.assertEqual(protocol['version'], VERSION)
            self.assertEqual(protocol['epochs'], 1)
            self.assertEqual(protocol['splits'], {'train': ['train_a', 'train_b'], 'val': ['val_a', 'val_b']})
            self.assertEqual([item['subject_id'] for item in protocol['excluded']], ['empty'])
            self.assertEqual(set(manifest), {'train_a', 'train_b', 'val_a', 'val_b'})
            self.assertFalse(any('test_only' in anchor['key'] for anchor in vocab))
            self.assertEqual({anchor['value'] for anchor in vocab if anchor['field'] == 'IDH'},
                             {'mutant', 'wildtype'})
            self.assertEqual(protocol['metadata_sha256'], runner.digest(metadata))
            for split in ('train', 'val'):
                coverage = protocol['input_coverage'][split]
                self.assertEqual(coverage['retained'], 2)
                self.assertEqual(coverage['present_by_raw_label']['3'], 2)
                self.assertEqual(coverage['present_by_raw_label']['5'], 1)
            self.assertTrue(loaded)
            for sid, item in manifest.items():
                self.assertEqual(item['z_indices'], [0])
                self.assertTrue(item['region_present'][2])
                for modality, record in item['image_files'].items():
                    path = expected_images[sid][modality]
                    self.assertEqual(Path(record['path']), path.resolve())
                    self.assertEqual(record['bytes'], path.stat().st_size)
                    self.assertEqual(record['mtime_ns'], path.stat().st_mtime_ns)

            configs, counts = [], []
            for method in METHODS:
                with self.subTest(method=method):
                    arm = output / method
                    config = runner.read(arm / 'config.json')
                    configs.append(config['protocol'])
                    counts.append(config['parameter_count'])
                    summary = runner.read(arm / 'complete.json')
                    self.assertTrue(summary['validation_replay_equal'])
                    self.assertFalse(summary['test_evaluated'])
                    self.assertEqual(summary['training_steps'], 1)
                    self.assertEqual(summary['selected_epoch'], 1)
                    for filename, metric_name in (('best_auc.pt', 'biomarker_macro_auc'),
                                                   ('best_map.pt', 'map'), ('last.pt', 'biomarker_macro_auc')):
                        checkpoint = torch.load(arm / filename, map_location='cpu', weights_only=True)
                        self.assertEqual(checkpoint['epoch'], 1)
                        self.assertEqual(checkpoint['selection_metric_name'], metric_name)
                        self.assertEqual(checkpoint['protocol_version'], VERSION)
                    epoch = runner.read(arm / 'validation_epoch_001.json')
                    best = runner.read(arm / 'validation_best_auc.json')
                    prior = runner.read(arm / 'frequency_baseline_validation.json')
                    history = runner.read(arm / 'history.json')
                    self.assertEqual(epoch['protocol_version'], VERSION)
                    for metrics in (epoch['metrics'], best['metrics'], prior, history[0]['validation']):
                        self.assertEqual(metrics['protocol_version'], VERSION)
                        self.assertEqual(metrics['case_count'], 2)
                        self.assertTrue(np.isfinite(metrics['biomarker_macro_auc']))
                    self.assertEqual(epoch['records']['protocol_version'], VERSION)
                    self.assertEqual(set(epoch['records']['subject_ids']), {'val_a', 'val_b'})
                    self.assertEqual(history[0]['encoded_training_views'],
                                     2 if method.startswith('whole_roi') else 3)
                    # A best-checkpoint reload reproduces the stored epoch score
                    # exactly; train_arm additionally compares two fresh replays.
                    self.assertEqual(epoch['records']['scores'], best['records']['scores'])
                    self.assertEqual(prior['biomarker_macro_auc'], 0.5)
                    json.loads((arm / 'history.json').read_text(encoding='utf-8'),
                               parse_constant=lambda value: self.fail('Nonstandard JSON: ' + value))
            self.assertTrue(all(config == protocol for config in configs))
            self.assertEqual(len(set(counts)), 1)
            complete = runner.read(output / 'complete.json')
            self.assertEqual(complete, {'version': VERSION, 'stage': 'smoke', 'runs': 4,
                                        'test_evaluated': False})
            self.assertEqual(len(runner.read(output / 'summary.json')), 4)


if __name__ == '__main__':
    unittest.main()
