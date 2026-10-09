import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import train_recent_alignment as training
import train_semantic_alignment as base
from recent_alignment import METHODS, RecentAlignmentModel, masked_nce
from semantic_evaluation import score_retrieval_metrics
from third_party.radzero.similarity import SimilarityLogit
from third_party.radzero.reference_losses import multi_positive_nce_loss
from tools import run_pasa_server as server
from tools.summarize_recent_alignment import patient_aps, summarize


class RecentAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def text_fixture(self, anchors=3):
        torch.manual_seed(10)
        tokens = torch.randn(anchors, 5, 12)
        mask = torch.ones(anchors, 5, dtype=torch.bool)
        mask[0, -2:] = False
        return {'global': torch.randn(anchors, 12), 'tokens': tokens, 'mask': mask}

    def test_radzero_singleton_and_official_similarity_formula(self):
        query = torch.tensor([[1., 0.]])
        local = torch.tensor([[[1., 0.], [0., 1.]]])
        scores, attention = SimilarityLogit('cos')(query, local, temperature=.2, need_attn_weights=True)
        weights = F.softmax(torch.tensor([5., 0.]), dim=0)
        pooled = F.normalize(weights.unsqueeze(0), dim=-1)
        self.assertEqual(scores.shape, (1, 1))
        self.assertTrue(torch.allclose(scores[0, 0], pooled[0, 0]))
        self.assertEqual(attention[0].shape, (1, 1, 2))

    def test_radzero_per_positive_does_not_reward_only_one_easy_label(self):
        scores = torch.tensor([[8., -8., 0.]], requires_grad=True)
        positives = torch.tensor([[True, True, False]])
        known = torch.ones_like(positives)
        mass = masked_nce(scores, positives, known, symmetric=False)
        per_positive = masked_nce(scores, positives, known, per_positive=True, symmetric=False)
        self.assertGreater(per_positive.item(), mass.item() + 3)
        per_positive.backward()
        self.assertLess(scores.grad[0, 1].item(), 0)

    def test_masked_mp_nce_matches_official_unique_phrase_groups(self):
        scores = torch.tensor([[.4, -.1, .2], [-.3, .5, .6]])
        positive = torch.tensor([[True, True, False], [False, False, True]])
        ours = masked_nce(scores / .2, positive, torch.ones_like(positive), per_positive=True)
        official = multi_positive_nce_loss(scores.T, torch.tensor([0, 0, 1]), temperature=.2)
        self.assertTrue(torch.allclose(ours, official, atol=2e-6))

    def test_unknown_labels_zero_gradient_and_no_positive_rows(self):
        scores = torch.randn(2, 3, requires_grad=True)
        pos = torch.tensor([[True, False, False], [False, False, False]])
        known = torch.tensor([[True, True, False], [False, False, False]])
        loss = masked_nce(scores, pos, known, per_positive=True)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad[:, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(scores.grad[1], torch.zeros(3)))

    def test_all_six_methods_real_tensor_forward_backward(self):
        for method in METHODS:
            model = RecentAlignmentModel(method, 3, self.text_fixture(), z_slices=3,
                                         feat_dim=16, shared_dim=8, private_dim=8)
            result = model(torch.rand(2, 4, 3, 24, 24))
            self.assertEqual(result['scores'].shape, (2, 3))
            positives = torch.tensor([[True, False, True], [False, True, False]])
            known = torch.tensor([[True, True, True], [True, True, False]])
            loss = model.alignment_loss(result, positives, known)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(grad).all() for grad in gradients))
            model.eval()
            with torch.no_grad():
                singleton = model(torch.rand(1, 4, 3, 24, 24))
            self.assertEqual(singleton['scores'].shape, (1, 3))

    def test_carzero_padding_cannot_change_scores(self):
        model = RecentAlignmentModel('carzero_metadata', 3, self.text_fixture(), z_slices=3,
                                     feat_dim=16, shared_dim=8, private_dim=8).eval()
        images = torch.rand(1, 4, 3, 24, 24)
        with torch.no_grad():
            first = model(images)['scores'].clone()
            model.text_local[0, -2:] = 100000
            second = model(images)['scores']
        self.assertTrue(torch.allclose(first, second))

    def test_precomputed_scores_do_not_get_called_cosine(self):
        result = score_retrieval_metrics([[10., -4.]], [[0]], valid_ids=[[0, 1]], positive_counts=[2])
        self.assertEqual(result['hit@1'], 1)
        self.assertEqual(result['recall@1'], .5)
        self.assertTrue(math.isnan(result['positive_negative_similarity_gap']))

    def test_case_auc_is_distinct_from_candidate_pair_auc(self):
        vocab = [base.make_anchor('IDH', 'wildtype'), base.make_anchor('IDH', 'mutant')]
        keys = {a['key']: i for i, a in enumerate(vocab)}
        lookup = {'a': {'metadata': {'IDH': 'wildtype'}}, 'b': {'metadata': {'IDH': 'mutant'}}}
        result, _ = training.score_patients(np.asarray([[2., 0.], [0., 2.]]), ['a', 'b'], lookup, vocab, keys)
        self.assertEqual(result['biomarker_macro_auc'], 1)
        single, _ = training.score_patients(np.asarray([[2., 0.]]), ['a'], lookup, vocab, keys)
        self.assertTrue(math.isnan(single['biomarker_macro_auc']))

    def test_text_cache_only_uses_train_keys_and_rejects_mutation(self):
        from types import SimpleNamespace
        from tools.build_alignment_text_cache import build
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(_commit_hash='synthetic-revision')
            def forward(self, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.arange(24).float().reshape(2, 3, 4))
        class Tokenizer:
            def __call__(self, prompts, **kwargs):
                self.prompts = prompts
                return {'input_ids': torch.zeros(2, 3, dtype=torch.long),
                        'attention_mask': torch.tensor([[1, 1, 0], [1, 1, 1]])}
        tokenizer = Tokenizer()
        mock_transformers = SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: Encoder()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'transformers': mock_transformers}):
            root = Path(folder)
            keys = ['idh::wildtype', 'idh::mutant']
            server.write_json(root / 'prepared.json', {'identity': {'vocab_keys': keys}})
            build(root, 'synthetic-encoder')
            cached = torch.load(root / 'text_cache.pt', weights_only=True)
            self.assertEqual(cached['keys'], keys)
            self.assertEqual(tokenizer.prompts, ['idh: wildtype.', 'idh: mutant.'])
            self.assertTrue(torch.equal(cached['global'][0], torch.tensor([2., 3., 4., 5.])))
            build(root, 'synthetic-encoder')
            (root / 'text_cache.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                build(root, 'synthetic-encoder')

    def test_one_epoch_synthetic_campaign_and_paired_summary(self):
        cases = [{'subject_id': str(i), 'label': 0,
                  'metadata': {'IDH': 'wildtype' if i != 1 else 'mutant', 'Tumor Grade': '4'}}
                 for i in range(4)]
        def loader(group, args, name):
            return [{'images': torch.rand(len(group), 4, 3, 24, 24),
                     'subject_id': [case['subject_id'] for case in group]}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = root / 'metadata.tsv'
            metadata.write_text('synthetic', encoding='utf-8')
            splits = {'train': ['0', '1'], 'val': ['2'], 'test': ['3']}
            vocab, _ = base.build_anchor_vocab(cases[:2])
            source_names = ['train_recent_alignment.py', 'recent_alignment.py', 'semantic_evaluation.py']
            identity = {'data_root': str(root), 'metadata_tsv': str(metadata), 'metadata_sha256': server.digest(metadata),
                'source_sha256': {name: server.digest(server.ROOT / name) for name in source_names},
                'splits': splits, 'vocab_keys': [anchor['key'] for anchor in vocab],
                'roi_size': 24, 'z_slices': 3, 'batch_size': 2, 'cpu': True, 'sample_seed': 42, 'split_seed': 42,
                'benchmark_settings': {'num_workers': 0}}
            server.write_json(root / 'prepared.json', {'identity': identity})
            server.write_json(root / 'splits.json', splits)
            text = {**self.text_fixture(len(vocab)), 'keys': identity['vocab_keys']}
            torch.save(text, root / 'text_cache.pt')
            server.write_json(root / 'text_cache.json', {'sha256': server.digest(root / 'text_cache.pt')})
            for method in METHODS:
                out = root / 'runs' / 'sota' / f'{method}_s42' / 'attempt_001'
                cli = ['training', '--campaign_root', str(root), '--method', method, '--seed', '42',
                       '--epochs', '1', '--out_dir', str(out)]
                with patch.object(sys, 'argv', cli), patch.object(base, 'discover_semantic_cases', return_value=cases), \
                     patch.object(base, 'make_loader', side_effect=loader):
                    training.main()
                server.write_json(out / 'complete.json', {})
                records = json.loads((out / 'patient_score_records.json').read_text(encoding='utf-8'))
                self.assertEqual(sorted(patient_aps(records)), ['3'])
            summarize(root, list(METHODS), [42], n_bootstrap=10)
            paired = json.loads((root / 'paired_comparisons.json').read_text(encoding='utf-8'))
            self.assertEqual(paired['patients'], 1)
            self.assertTrue((root / 'sota_mean.csv').exists())


if __name__ == '__main__':
    unittest.main()
