import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import train_semantic_alignment as training
from semantic_evaluation import binary_auc, retrieval_metrics, supervision_from_keys, validate_patient_splits
from utils.bootstrap_semantic_5seed import load_run, run_metrics_on_indices


class RevisedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.vocab = [training.make_anchor('IDH', 'wildtype'), training.make_anchor('IDH', 'mutant'),
                      training.make_anchor('MGMT', 'methylated'), training.make_anchor('MGMT', 'unmethylated'),
                      training.make_anchor('Tumor Grade', '4')]
        self.keys = {a['key']: i for i, a in enumerate(self.vocab)}
        self.args = SimpleNamespace(target_policy='all_patient_anchors', exclude_pathology_anchors=False,
                                    exclude_molecular_anchors=False, include_clinical_anchors=False,
                                    evaluation_fields='molecular', query_mode='units', node_mode='regions')

    def test_unknown_field_is_not_negative(self):
        spec = supervision_from_keys(['idh::wildtype'], self.vocab, self.keys)
        self.assertEqual(spec['valid_ids'], [0, 1])
        self.assertEqual(spec['positive_ids'], [0])

    def test_region_missing_does_not_fallback(self):
        self.assertEqual(training.target_anchor_keys({'IDH': 'wildtype'}, 'enhancing', 'region_rules'), [])

    def test_ablation_keeps_evaluation_fixed(self):
        patient = {'Tumor Grade': '4', 'IDH': 'wildtype', 'MGMT': 'unknown'}
        full = training.evaluation_specs(patient, self.vocab, self.keys, self.args)
        self.args.exclude_pathology_anchors = True
        self.assertEqual(full, training.evaluation_specs(patient, self.vocab, self.keys, self.args))
        targets = training.build_query_supervisions(['p'], ['enhancing'], {'p': {'metadata': patient}},
                                                    self.keys, self.vocab, self.args)
        self.assertEqual(targets[0]['positive_ids'], [0])

    def test_hit_recall_oov_and_unknown_ranking(self):
        metrics = retrieval_metrics([[1., 0.]], [[0]], [[1., 0.], [.9, .1], [0., 1.]],
                                    valid_ids=[[0, 2]], positive_counts=[2], ks=(1,))
        self.assertEqual(metrics['hit@1'], 1.)
        self.assertEqual(metrics['recall@1'], .5)
        self.assertEqual(metrics['map'], .5)
        self.assertEqual(metrics['unavailable_positive_count'], 1)

    def test_no_label_vs_unretrievable_label(self):
        metrics = retrieval_metrics([[1., 0.], [1., 0.]], [[], []], [[1., 0.]],
                                    valid_ids=[[], [0]], positive_counts=[0, 1], ks=(1,))
        self.assertEqual(metrics['skipped_no_positive'], 1)
        self.assertEqual(metrics['mrr'], 0)
        self.assertEqual(metrics['recall@1'], 0)

    def test_patient_field_aggregation_not_query_weighted(self):
        metrics = retrieval_metrics([[1., 0.]] * 3, [[0], [0], [1]], [[1., 0.], [0., 1.]],
                                    subject_ids=['a', 'a', 'b'], field_names=['IDH'] * 3, ks=(1,))
        self.assertEqual(metrics['hit@1'], .5)

    def test_auc_ties(self):
        self.assertEqual(binary_auc([0, 1], [1, 1]), .5)
        self.assertEqual(binary_auc([0, 1], [0, 1]), 1.)
        self.assertTrue(np.isnan(binary_auc([1, 1], [0, 1])))

    def test_splits_empty_overlap_and_duplicates(self):
        valid = {name: [{'subject_id': name}] for name in ('train', 'val', 'test')}
        validate_patient_splits(valid)
        for bad in (dict(valid, val=[]), dict(valid, test=valid['train']),
                    dict(valid, train=valid['train'] * 2)):
            with self.assertRaises(ValueError):
                validate_patient_splits(bad)

    def test_clip_empty_rows_and_unknown_gradients(self):
        queries = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        bank = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        loss = training.multi_positive_contrastive_loss(queries, [[0], []], bank, valid_ids=[[0, 1], []])
        expected = training.multi_positive_contrastive_loss(queries[:1], [[0]], bank, valid_ids=[[0, 1]])
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.equal(bank.grad[2], torch.zeros(2)))
        self.assertTrue(torch.equal(queries.grad[1], torch.zeros(2)))

    def test_medclip_empty_rows_and_bce(self):
        q = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        p = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        loss = training.medclip_multi_positive_loss(q, [[0], []], p, [[1], [0]], valid_ids=[[0, 1], []])
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.equal(q.grad[1], torch.zeros(2)))
        self.assertTrue(torch.isfinite(training.masked_multilabel_loss(q, [[0], []], p, [[0, 1], []])))

    def test_paper_profile_preserves_requested_ablation(self):
        args = SimpleNamespace(paper_config='paper1', variant='no_anchor_loss', alignment_objective='clip', lambda_anchor=.05)
        training.apply_variant(training.apply_paper_profile(args))
        self.assertEqual(args.variant, 'no_anchor_loss')
        self.assertEqual(args.lambda_anchor, 0)

    def test_bootstrap_patient_repetitions_keep_multiplicity(self):
        run = {'protocol_version': 'pasa_patient_metadata_v2',
               'query_vectors': np.asarray([[1., 0.], [1., 0.]]),
               'query_targets': [[0], [1]], 'prototypes': [[1., 0.], [0., 1.]],
               'subject_ids': ['a', 'b'], 'query_valid_ids': [[0, 1], [0, 1]],
               'query_positive_counts': [1, 1], 'query_fields': ['IDH', 'IDH']}
        result = run_metrics_on_indices(run, [0, 1, 1], ['draw_0', 'draw_1', 'draw_2'])
        self.assertAlmostEqual(result['hit@1'], 1 / 3)

    def test_collection_saving_and_gallery_depletion(self):
        class Model:
            def eval(self):
                pass
            def __call__(self, images, **kwargs):
                return {'extras': {'shared': torch.tensor([[[1., 0.]] * 3]),
                                   'adjacency': torch.zeros(1, 3, 3)}}
        bank = training.SemanticPrototypeBank(5, 2)
        with torch.no_grad():
            bank.prototypes.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]]))
        loader = [{'images': torch.zeros(1, 4, 1, 2, 2), 'subject_id': ['p']}]
        lookup = {'p': {'metadata': {'IDH': 'wildtype', 'MGMT': 'unknown'}}}
        records = training.collect_alignment_records(Model(), bank, loader, 'cpu', self.args, lookup, self.keys, self.vocab)
        self.assertEqual(len(records['query_targets']), 9)
        self.assertEqual(training.score_records(records)['skipped_no_positive'], 6)
        self.assertEqual(training.score_records(records, gallery_ids=[4])['recall@1'], 0)
        with tempfile.TemporaryDirectory() as folder:
            training.save_patient_level_records(records, folder)
            with open(Path(folder) / 'patient_level_records.json', encoding='utf-8') as file:
                payload = json.load(file)
            self.assertEqual(payload['query_valid_ids'], records['query_valid_ids'])
            self.assertEqual(payload['query_fields'], records['query_fields'])
        self.args.query_mode = 'global'
        records = training.collect_alignment_records(Model(), bank, loader, 'cpu', self.args, lookup, self.keys, self.vocab)
        self.assertEqual(len(records['query_targets']), 3)
        self.assertEqual(records['query_records'][0]['node_name'], 'PatientGlobal')

    def test_one_epoch_orchestration_with_synthetic_model(self):
        class Model(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.vector = torch.nn.Parameter(torch.tensor([1., .5]))
            def forward(self, images, **kwargs):
                shared = self.vector.expand(len(images), 3, 2)
                zero = shared.sum() * 0
                names = ['cons', 'decouple', 'leak', 'diff', 'diff_norm', 'gate_entropy',
                         'load_balance', 'graph_energy']
                return {'losses': {name: zero for name in names},
                        'extras': {'shared': shared, 'adjacency': torch.zeros(len(images), 3, 3),
                                   'shared_norm': shared.norm(), 'private_norm': zero,
                                   'diffusion_residual_norm': zero}}
        cases = [{'subject_id': str(i), 'label': 0,
                  'metadata': {'IDH': 'wildtype' if i != 1 else 'mutant', 'Tumor Grade': '4'}}
                 for i in range(4)]
        def loader(group, args, name):
            return [{'images': torch.zeros(len(group), 4, 1, 2, 2),
                     'subject_id': [case['subject_id'] for case in group]}]
        with tempfile.TemporaryDirectory() as folder:
            splits = Path(folder) / 'split.json'
            splits.write_text(json.dumps({'train': ['0', '1'], 'val': ['2'], 'test': ['3']}), encoding='utf-8')
            for variant in ('full', 'no_anchor', 'no_anchor_loss', 'global_clip', 'multilabel', 'medclip_style', 'clip', 'no_graph'):
                args = training.build_parser().parse_args(['--data_root', 'synthetic', '--out_dir', str(Path(folder) / variant),
                    '--splits_file', str(splits), '--epochs', '1', '--shared_dim', '2', '--cpu', '--variant', variant])
                with patch.object(training, 'discover_semantic_cases', return_value=cases), \
                     patch.object(training, 'make_loader', side_effect=loader), \
                     patch.object(training, 'GliomaGraphDiffusionNet', Model), \
                     patch.object(training, 'save_alignment_space_plot'), \
                     patch.object(training, 'save_semantic_unit_graph'):
                    training.main(args)
                with open(Path(args.out_dir) / 'test_metrics.json', encoding='utf-8') as file:
                    metrics = json.load(file)
                self.assertEqual(metrics['case_count'], 1)
                self.assertEqual(metrics['protocol_version'], 'pasa_patient_metadata_v2')
                self.assertTrue(np.isfinite(metrics['map']))
                self.assertTrue((Path(args.out_dir) / 'best_semantic_alignment.pt').exists())
                restored = load_run(Path(args.out_dir))
                re_scored = run_metrics_on_indices(restored, list(range(len(restored['query_targets']))))
                self.assertAlmostEqual(re_scored['map'], metrics['map'])


if __name__ == '__main__':
    unittest.main()
