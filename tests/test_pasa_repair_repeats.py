"""Scheduler checks use synthetic artifacts, not new clinical training results."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import repeat_pasa_repair_pilot as repeat
from tools import run_pasa_repair_pilot as pilot


class RepeatTests(unittest.TestCase):
    def make_arm(self,folder,protocol,method):
        out=folder/method;out.mkdir()
        history=[]
        records={'subject_ids':protocol['splits']['val'],'scores':[[.1,.2],[.2,.1]]}
        for epoch in range(1,11):
            metrics={'protocol_version':pilot.VERSION,'biomarker_macro_auc':.7 if epoch==2 else .6,'map':.8}
            history.append({'epoch':epoch,'validation':metrics})
            pilot.write_json(out/f'validation_epoch_{epoch:03d}.json',{'metrics':metrics,'records':records})
        pilot.write_json(out/'config.json',{'protocol':protocol})
        pilot.write_json(out/'history.json',history)
        pilot.write_json(out/'validation_best_auc.json',{'selected_epoch':2,'metrics':history[1]['validation'],'records':records})
        result={'method':method,'selected_epoch':2,'validation_auc':.7,'validation_map':.8,
            'training_steps':10,'seconds':.01,'validation_replay_equal':True,'test_evaluated':False}
        pilot.write_json(out/'complete.json',result)
        for n in ('best_auc.pt','best_map.pt','last.pt'):(out/n).write_bytes(b'synthetic-test-artifact')
        return result

    def fixture(self,root):
        reference=root/'reference';reference.mkdir()
        meta=root/'metadata.tsv';meta.write_text('synthetic labels')
        cases=[{'subject_id':sid,'metadata':{'IDH':'mutated' if i%2 else 'wild type'}} for i,sid in enumerate('abcd')]
        vocab,keys=pilot.base.build_anchor_vocab(cases[:2])
        manifest={}
        for sid in 'abcd':
            seg=root/(sid+'.seg');seg.write_bytes(b'synthetic segmentation bytes')
            image=root/(sid+'.image');image.write_bytes(b'synthetic MRI bytes')
            stat=image.stat()
            manifest[sid]={'split':'train' if sid in 'ab' else 'val','region_present':[True,False,False,False,False],
                'segmentation_file':str(seg),'segmentation_sha256':pilot.digest(seg),
                'image_files':{'t1':{'path':str(image),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}}
        protocol={'version':pilot.VERSION,'stage':'pilot','seed':42,'epochs':10,'methods':list(pilot.METHODS),
            'source_sha256':{n:pilot.digest(pilot.ROOT/n) for n in pilot.SOURCES},
            'metadata':str(meta),'metadata_sha256':pilot.digest(meta),'data_root':str(root),
            'splits':{'train':list('ab'),'val':list('cd')},'batch_size':2,'cpu':True,'workers':0,'roi_size':16,'z_slices':1,
            'excluded':[{'subject_id':'excluded','reason':'synthetic zero segmentation'}]}
        pilot.write_json(reference/'protocol.json',protocol)
        pilot.write_json(reference/'inputs.json',manifest)
        pilot.write_json(reference/'anchor_vocab.json',vocab)
        summaries=[self.make_arm(reference,protocol,m) for m in pilot.METHODS]
        pilot.write_json(reference/'summary.json',summaries)
        pilot.write_json(reference/'complete.json',{'version':pilot.VERSION,'stage':'pilot','runs':4,'test_evaluated':False})
        return reference,cases

    def invoke(self,reference,out,cases,trainer,resume=False):
        argv=['repeat','--reference',str(reference),'--out',str(out)]+(['--resume'] if resume else [])
        with patch.object(sys,'argv',argv),patch.object(pilot.base,'discover_semantic_cases',return_value=cases),patch.object(pilot,'train_arm',side_effect=trainer):
            repeat.main()

    def test_eight_runs_preserve_frozen_inputs_and_skip_completed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference,cases=self.fixture(root);out=root/'repeat'
            calls=[]
            def train(args,folder,protocol,manifest,lookup,vocab,keys,method):
                calls.append((protocol['seed'],method))
                return self.make_arm(folder,protocol,method)
            self.invoke(reference,out,cases,train)
            self.assertEqual(len(calls),8)
            for seed in (43,44):
                child=out/f'seed_{seed}/attempt_001'
                p=pilot.read(child/'protocol.json')
                original=pilot.read(reference/'protocol.json')
                self.assertEqual(p['seed'],seed)
                for key in original:
                    if key!='seed':self.assertEqual(p[key],original[key])
                for name in ('inputs.json','anchor_vocab.json'):
                    self.assertEqual(pilot.digest(child/name),pilot.digest(reference/name))
            calls.clear();self.invoke(reference,out,cases,train,resume=True)
            self.assertEqual(calls,[])
            self.assertEqual(len(pilot.read(out/'summary.json')['runs']),12)
            self.assertFalse((out/'.run.lock').exists())
            # Completed data corruption must stop a resume before training.
            (out/'seed_43/attempt_001/inputs.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                self.invoke(reference,out,cases,train,resume=True)
            self.assertEqual(calls,[])

    def test_failed_seed_restarts_new_attempt_and_preserves_completed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference,cases=self.fixture(root);out=root/'repeat'
            def fail(args,folder,protocol,manifest,lookup,vocab,keys,method):
                if protocol['seed']==44:raise RuntimeError('synthetic interruption')
                return self.make_arm(folder,protocol,method)
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                self.invoke(reference,out,cases,fail)
            self.assertTrue((out/'seed_43/attempt_001/complete.json').exists())
            self.assertFalse((out/'complete.json').exists())
            self.assertTrue((out/'seed_44/attempt_001/failed.json').exists())
            # Gaps in retained attempt numbering must not cause overwrite.
            (out/'seed_44/attempt_003').mkdir()
            calls=[]
            def train(args,folder,protocol,manifest,lookup,vocab,keys,method):
                calls.append(protocol['seed']);return self.make_arm(folder,protocol,method)
            self.invoke(reference,out,cases,train,resume=True)
            self.assertEqual(calls,[44]*4)
            self.assertTrue((out/'seed_44/attempt_004/complete.json').exists())

    def test_reference_metadata_and_lock_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference,cases=self.fixture(root)
            protocol=pilot.read(reference/'protocol.json')
            Path(protocol['metadata']).write_text('changed')
            with patch.object(pilot.base,'discover_semantic_cases',return_value=cases):
                with self.assertRaisesRegex(ValueError,'Metadata changed'):repeat.load_reference(reference)
            lock_root=root/'locked';lock_root.mkdir()
            with repeat.campaign_lock(lock_root):
                with self.assertRaisesRegex(ValueError,'locked'):
                    with repeat.campaign_lock(lock_root):self.fail('Acquired a duplicate lock')
            self.assertFalse((lock_root/'.run.lock').exists())
