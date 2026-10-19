"""Meaningful invariants for read-only real-data diagnostic helpers."""
import json
import hashlib
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

import train_semantic_alignment as base
from recent_alignment import RecentAlignmentModel
from tools.diagnose_pasa_server import region_counts, gradient_probe, validation_pass
from tools import diagnose_pasa_server as diagnostic


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        cases = [{'subject_id': str(i), 'metadata': {'IDH': label, 'MGMT': '', '1p19Q CODEL': ''}}
                 for i, label in enumerate(['mutated', 'wild type'])]
        vocab, keys = base.build_anchor_vocab(cases)
        lookup = {c['subject_id']: c for c in cases}
        torch.manual_seed(14)
        model = RecentAlignmentModel('pasa_full', len(vocab), z_slices=7)
        batch = {'subject_id': ['0', '1'], 'images': torch.randn(2, 4, 7, 32, 32),
                 'region_masks': torch.ones(2, 3, 7, 32, 32)}
        return model, [batch], lookup, vocab, keys

    def test_empty_region_counts_and_label_identity(self):
        seg = np.zeros((16,16,9), dtype=np.float32)
        seg[3:7,4:8,3:6] = 1
        rows = region_counts(seg, 16, 7)
        self.assertEqual([r['label'] for r in rows], [1,2,4])
        self.assertGreater(rows[0]['sampled_native_voxels'], 0)
        self.assertGreater(rows[0]['network_mask_voxels'], 0)
        self.assertEqual(rows[1]['network_mask_voxels'], 0)
        self.assertEqual(rows[2]['network_mask_voxels'], 0)

    def test_gradient_probe_preserves_checkpoint_parameters_and_buffers(self):
        model, loader, lookup, vocab, keys = self.fixture()
        before = {n: v.clone() for n,v in model.state_dict().items()}
        args = base.build_parser().parse_args(['--data_root', '.'])
        with tempfile.TemporaryDirectory() as directory:
            gradient_probe(model, loader, 'cpu', lookup, vocab, keys, args, 1, 1, Path(directory))
            result = json.loads((Path(directory)/'gradient_probe.json').read_text())
        self.assertEqual(len(result['batches']), 1)
        components = result['batches'][0]['components']
        self.assertGreater(components['alignment']['shared_head_gradient_norm'], 0)
        for n,v in model.state_dict().items():
            self.assertTrue(torch.equal(before[n],v), n)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)

    def test_validation_fixed_seed_replays_without_mutating_state(self):
        model, loader, lookup, vocab, keys = self.fixture()
        before = {n:v.clone() for n,v in model.state_dict().items()}
        base.set_seed(88)
        a = validation_pass(model,loader,'cpu',lookup,vocab,keys)
        b = validation_pass(model,loader,'cpu',lookup,vocab,keys)
        base.set_seed(88)
        replay = validation_pass(model,loader,'cpu',lookup,vocab,keys)
        self.assertEqual(a[3], ['0','1'])
        self.assertEqual(a[0]['scored_patients'],2)
        np.testing.assert_array_equal(a[2],replay[2])
        self.assertFalse(np.array_equal(a[2],b[2]))
        for n,v in model.state_dict().items():
            self.assertTrue(torch.equal(before[n],v),n)

    def test_full_command_uses_only_train_val_and_preserves_checkpoint(self):
        model, loader, lookup, vocab, keys = self.fixture()
        cases = list(lookup.values()) + [
            {'subject_id': '2', 'metadata': lookup['0']['metadata']},
            {'subject_id': '3', 'metadata': lookup['1']['metadata']},
            {'subject_id': '4', 'metadata': lookup['0']['metadata']}]
        frozen = {'train':['0','1'], 'val':['2','3'], 'test':['4']}
        sources = {name:hashlib.sha256((diagnostic.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                   for name in ['recent_alignment.py','experiment_model.py','experiment_dataset.py',
                                'dataset.py','semantic_evaluation.py','train_recent_alignment.py']}
        sources['train_semantic_alignment.py']='496c889709e444b68591129ddcdd8439f27ef862126ecffee226eef1e1e30b0d'
        seen=[]
        def make_loader(group,args,split):
            ids=[c['subject_id'] for c in group]
            self.assertNotIn('4',ids)
            seen.extend(ids)
            return [{**loader[0], 'subject_id':ids}]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            campaign=root/'campaign'
            run=campaign/'runs/sota/pasa_full_s42/attempt_001'
            run.mkdir(parents=True)
            metadata=root/'metadata.tsv'
            metadata.write_text('synthetic fixture')
            identity={'splits':frozen,'metadata_sha256':diagnostic.digest(metadata),'source_sha256':sources,
                'roi_size':32,'z_slices':7,'batch_size':2,'sample_seed':42,'split_seed':42,
                'vocab_keys':[a['key'] for a in vocab]}
            for path,value in [(campaign/'prepared.json',{'identity':identity}), (campaign/'splits.json',frozen),
                (run/'config.json',{'method':'pasa_full','settings':identity}), (run/'anchor_vocab.json',vocab),
                (run/'history.json',[{'epoch':1,'validation':{'map':.8}}])]:
                path.write_text(json.dumps(value))
            torch.save(model.state_dict(),run/'best.pt')
            checkpoint_hash=diagnostic.digest(run/'best.pt')
            argv=['diagnostic','--campaign',str(campaign),'--metadata',str(metadata),'--out',str(root/'out'),
                '--seeds','42','--cpu','--workers','0','--repeats','2','--gradient-batches','1']
            with patch.object(sys,'argv',argv), patch.object(base,'discover_semantic_cases',return_value=cases), \
                    patch.object(base,'make_loader',side_effect=make_loader), patch.object(diagnostic,'scan_regions'):
                diagnostic.main()
            complete=json.loads((root/'out/complete.json').read_text())
            self.assertFalse(complete['test_evaluated'])
            self.assertEqual(complete['optimizer_steps'],0)
            self.assertEqual(checkpoint_hash,diagnostic.digest(run/'best.pt'))
            self.assertEqual(set(seen),{'0','1','2','3'})


if __name__ == '__main__':
    unittest.main()
