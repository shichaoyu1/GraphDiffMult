import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_pasa_server as runner


class ServerRunnerTests(unittest.TestCase):
    def test_core_contract_shared_endpoint_and_single_center_disabled(self):
        args = runner.parser().parse_args(['--stage', 'core', '--output_root', 'synthetic-output'])
        scheduled = list(runner.jobs(args))
        self.assertEqual(len(scheduled), 12)
        self.assertEqual(len({cmd[cmd.index('--splits_file') + 1] for _, _, _, cmd in scheduled}), 1)
        for _, name, _, command in scheduled:
            self.assertEqual(command[command.index('--evaluation_fields') + 1], 'molecular')
            self.assertEqual(command[command.index('--lambda_anchor') + 1], '0')
        single = next(cmd for _, name, _, cmd in scheduled if name == 'unit_single_positive')
        self.assertEqual(single[single.index('--alignment_objective') + 1], 'single_positive')

    def test_completed_job_skipped_and_changed_command_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'runs' / 'core' / 'unit_s42' / 'attempt_001'
            attempt.mkdir(parents=True)
            (attempt / 'complete.json').write_text(json.dumps({'base_command': ['original']}), encoding='utf-8')
            with patch.object(runner.subprocess, 'Popen') as launch:
                runner.run_job(root, 'core', 'unit', 42, ['original'])
                launch.assert_not_called()
            with self.assertRaises(ValueError):
                runner.run_job(root, 'core', 'unit', 42, ['changed'])

    def test_failed_job_retry_preserves_prior_attempt(self):
        class FailedProcess:
            stdout = ['synthetic failure\n']
            def wait(self):
                return 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner.subprocess, 'Popen', return_value=FailedProcess()):
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        runner.run_job(root, 'core', 'unit', 42, ['synthetic'])
            attempts = sorted((root / 'runs' / 'core' / 'unit_s42').glob('attempt_*'))
            self.assertEqual(len(attempts), 2)
            self.assertTrue((attempts[0] / 'train.log').exists())
            self.assertFalse((attempts[0] / 'complete.json').exists())

    def test_preparation_rejects_changed_metadata_and_split(self):
        import train_semantic_alignment as training
        cases = [{'subject_id': str(i), 'label': 0, 'metadata': {'IDH': 'wildtype' if i % 2 else 'mutant'}}
                 for i in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            data.mkdir()
            metadata = root / 'metadata.tsv'
            metadata.write_text('synthetic', encoding='utf-8')
            args = runner.parser().parse_args(['--stage', 'prepare', '--cpu', '--data_root', str(data),
                '--metadata_tsv', str(metadata), '--output_root', str(root / 'output')])
            with patch.object(training, 'discover_semantic_cases', return_value=cases), \
                 patch.object(runner.subprocess, 'run') as freeze:
                freeze.return_value.stdout = 'synthetic environment'
                runner.prepare(args)
                runner.prepare(args)
                split_file = root / 'output' / 'splits.json'
                original = split_file.read_text(encoding='utf-8')
                split_file.write_text('{}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.prepare(args)
                split_file.write_text(original, encoding='utf-8')
                metadata.write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError):
                    runner.prepare(args)

    def test_real_network_tensor_forward_backward_for_server_core(self):
        import math
        import torch
        import train_semantic_alignment as training
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for variant, objective in [('clip', 'clip'), ('global_clip', 'clip'),
                                       ('multilabel', 'multilabel'), ('clip', 'single_positive')]:
                args = training.build_parser().parse_args(['--data_root', 'synthetic', '--variant', variant,
                    '--z_slices', '3', '--feat_dim', '16', '--shared_dim', '16', '--private_dim', '16',
                    '--lambda_anchor', '0'])
                args = training.apply_variant(args)
                args.alignment_objective = objective
                model = training.GliomaGraphDiffusionNet(num_classes=1, z_slices=3, feat_dim=16,
                    shared_dim=16, private_dim=16, graph_type='no_graph', use_anchor=False,
                    use_private=False, use_diffusion=False)
                vocab = [training.make_anchor('IDH', 'wildtype'), training.make_anchor('IDH', 'mutant')]
                keys = {anchor['key']: i for i, anchor in enumerate(vocab)}
                bank = training.SemanticPrototypeBank(2, 16)
                optimizer = torch.optim.SGD(list(model.parameters()) + list(bank.parameters()), lr=.001)
                batch = {'images': torch.rand(2, 4, 3, 24, 24), 'subject_id': ['a', 'b']}
                lookup = {'a': {'metadata': {'IDH': 'wildtype'}}, 'b': {'metadata': {'IDH': 'mutant'}}}
                result = training.run_epoch(model, bank, [batch], optimizer, 'cpu', args, lookup, keys, 1,
                    {'anchor_vocab': vocab, 'medclip_ignore_ids': [[], []]})
                self.assertTrue(math.isfinite(result['total']))
                self.assertTrue(torch.isfinite(bank.prototypes.grad).all())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
