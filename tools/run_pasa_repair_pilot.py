"""Frozen train/validation-only pilot for repaired raw-region MRI inputs."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_semantic_alignment as base
from train_recent_alignment import label_masks, score_patients as original_score_patients
from pasa_region_inputs import select_region_inputs, EmptyRegionInputError
from pasa_repair_pilot import VERSION, REGION_LABELS, METHODS, RepairDataset, RepairAlignmentModel, validation_rng
from json_results import write_json
from semantic_evaluation import validate_patient_splits
from dataset import find_modality_file

SOURCES=['tools/run_pasa_repair_pilot.py','pasa_region_inputs.py','pasa_repair_pilot.py',
    'train_semantic_alignment.py','train_recent_alignment.py','semantic_evaluation.py',
    'experiment_dataset.py','experiment_model.py','dataset.py','recent_alignment.py','json_results.py']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed);base.set_seed(seed)


def score_patients(scores,subjects,lookup,vocab,keys):
    metrics,records=original_score_patients(scores,subjects,lookup,vocab,keys)
    metrics.update(protocol_version=VERSION,score_type='patient_cosine_logits',
        aggregation='patient_score_then_field_then_patient')
    metrics['biomarker_estimable_fields']=sum(np.isfinite(v['macro_one_vs_rest_auc']) for v in metrics['biomarker_auc_by_field'].values())
    records['protocol_version']=VERSION
    return metrics,records


def prepare(args,root):
    import nibabel as nib
    if root.exists():raise ValueError('Output already exists; preserve it and choose a new path')
    reference=read(Path(args.reference_campaign)/'prepared.json')['identity']
    original=read(Path(args.reference_campaign)/'splits.json')
    if original!=reference['splits'] or digest(args.metadata)!=reference['metadata_sha256']:
        raise ValueError('Frozen split or metadata changed')
    audited=read(Path(args.audit)/'records.json')
    records={r['subject_id']:r for r in audited}
    if set(records)!=(set(original['train'])|set(original['val'])):
        raise ValueError('Audit must cover exactly the original train and validation patients')
    lookup={str(c['subject_id']):c for c in base.discover_semantic_cases(args.data_root,metadata_tsv=args.metadata,seed=reference['sample_seed'])}
    manifest,excluded={},[]
    for split in ('train','val'):
        for i,sid in enumerate(original[split],1):
            row=records[sid]
            if row['split']!=split or digest(row['segmentation_file'])!=row['segmentation_sha256']:
                raise ValueError(f'Segmentation/audit changed: {sid}')
            seg_image=nib.load(row['segmentation_file'])
            seg=seg_image.get_fdata(dtype=np.float32)
            try:
                selected=select_region_inputs(seg,args.roi_size,args.z_slices,region_labels=REGION_LABELS)
            except EmptyRegionInputError as error:
                # Common target-independent exclusion, frozen before training all arms.
                excluded.append({'subject_id':sid,'split':split,'reason':str(error),
                    'segmentation_file':row['segmentation_file'],'segmentation_sha256':row['segmentation_sha256']})
                continue
            image_files={}
            for modality in ('t1','t1ce','t2','flair'):
                path=find_modality_file(lookup[sid]['patient_dir'],modality,prefer_registered=False)
                image=nib.load(path)
                if image.shape!=seg.shape or not np.allclose(image.affine,seg_image.affine,atol=1e-4,rtol=0):
                    raise ValueError(f'Image geometry mismatch during prepare: {sid} {modality}')
                stat=Path(path).stat()
                image_files[modality]={'path':str(Path(path).resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            manifest[sid]={'split':split,'segmentation_file':row['segmentation_file'],'image_files':image_files,
                'segmentation_sha256':row['segmentation_sha256'],'z_indices':selected['z_indices'],
                'bbox':list(selected['bbox']),'region_present':selected['region_present'].tolist(),
                'audit':selected['audit']}
            if i%50==0:print(f'prepare {split}: {i}/{len(original[split])}',flush=True)
    splits={name:[s for s in original[name] if s in manifest] for name in ('train','val')}
    # Validate all original partitions for overlaps; never create a test dataset.
    validate_patient_splits({name:[lookup[s] for s in original[name]] for name in ('train','val','test')})
    if not all(splits.values()):raise ValueError('Empty retained train/validation partition')
    if args.stage=='smoke':
        splits={name:ids[:args.smoke_patients] for name,ids in splits.items()}
    vocab,keys=base.build_anchor_vocab([lookup[s] for s in splits['train']])
    if not any(a['field']=='IDH' for a in vocab):raise ValueError('Training vocabulary has no molecular candidates')
    protocol={'version':VERSION,'stage':args.stage,'methods':args.methods,'seed':args.seed,
        'epochs':1 if args.stage=='smoke' else args.epochs,'roi_size':args.roi_size,'z_slices':args.z_slices,
        'batch_size':args.batch_size,'workers':args.workers,'cpu':args.cpu,'learning_rate':1e-4,'weight_decay':1e-4,
        'metadata':str(Path(args.metadata).resolve()),'metadata_sha256':digest(args.metadata),'data_root':args.data_root,
        'raw_region_labels':REGION_LABELS,'region_semantics':'unverified numeric bins; not histology',
        'sampling':'max_area_slice_center_lowest_z_tie','exclusion':'common no-accepted-region before/after resize',
        'primary_selection':'validation_biomarker_macro_auc_strict_improvement_first_tie',
        'validation_seed':20261020,'objective_direction':'patient_image_to_anchor_only',
        'supervision':'all available patient anchors; missing fields masked',
        'graph':False,'diffusion':False,'center_loss':False,'independent_regional_pathology':False,
        'original_split_sha256':digest(Path(args.reference_campaign)/'splits.json'),
        'test_policy':'no test loader, no test scoring; existing test already used in development',
        'data_hash_scope':'metadata and selected segmentations hashed; MRI geometry checked at load, raw MRI voxels not hashed',
        'source_sha256':{name:digest(ROOT/name) for name in SOURCES},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'splits':splits,'excluded':excluded,'retained_before_smoke_subsample':{name:sum(r['split']==name for r in manifest.values()) for name in ('train','val')}}
    protocol['input_coverage']={name:{
        'retained':sum(r['split']==name for r in manifest.values()),
        'present_by_raw_label':{str(label):sum(r['split']==name and r['region_present'][j] for r in manifest.values()) for j,label in enumerate(REGION_LABELS)},
        'resize_lost_by_raw_label':{str(label):sum(r['split']==name and label in r['audit']['labels_lost_at_resize'] for r in manifest.values()) for label in REGION_LABELS}}
        for name in ('train','val')}
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',protocol)
    write_json(root/'inputs.json',manifest)
    write_json(root/'anchor_vocab.json',vocab)
    return protocol,manifest,lookup,vocab,keys


def evaluate(model,loader,device,lookup,vocab,keys,seed):
    model.eval()
    scores,subjects=[],[]
    generator=loader.generator
    state=generator.get_state() if generator is not None else None
    try:
        with validation_rng(seed,device),torch.no_grad():
            for b in loader:
                result=model(b['images'].to(device),b['region_masks'].to(device),b['region_present'].to(device))
                scores.extend(result['scores'].cpu().tolist());subjects.extend(b['subject_id'])
    finally:
        if state is not None:generator.set_state(state)
    return score_patients(np.asarray(scores),subjects,lookup,vocab,keys)


def train_arm(args,root,protocol,manifest,lookup,vocab,keys,method):
    for name,h in protocol['source_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Source changed after protocol freeze: '+name)
    if digest(protocol['metadata'])!=protocol['metadata_sha256']:raise ValueError('Metadata changed after freeze')
    out=root/method;out.mkdir()
    seed_all(protocol['seed'])
    device='cpu' if args.cpu else 'cuda'
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    groups={name:[lookup[s] for s in ids] for name,ids in protocol['splits'].items()}
    datasets={name:RepairDataset(cases,manifest,args.roi_size,args.z_slices,augment=name=='train') for name,cases in groups.items()}
    loaders={name:DataLoader(ds,batch_size=args.batch_size,shuffle=name=='train',num_workers=args.workers,
        generator=torch.Generator().manual_seed(protocol['seed']+int(name=='train'))) for name,ds in datasets.items()}
    model=RepairAlignmentModel(method,len(vocab),z_slices=args.z_slices).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'])
    write_json(out/'config.json',{'method':method,'protocol':protocol,'parameter_count':sum(p.numel() for p in model.parameters())})
    history=[];best_auc=best_map=-float('inf');steps=0;started=time.time()
    for epoch in range(1,protocol['epochs']+1):
        model.train();loss_sum=0.;n=0;views=0
        for batch in loaders['train']:
            result=model(batch['images'].to(device),batch['region_masks'].to(device),batch['region_present'].to(device))
            pos,known,_=label_masks(batch['subject_id'],lookup,vocab,keys,device)
            loss=model.loss(result,pos,known,vocab)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_sum+=float(loss.detach())*len(batch['subject_id']);n+=len(batch['subject_id']);steps+=1
            views+=len(batch['subject_id']) if method.startswith('whole_roi') else int(batch['region_present'].sum())
        scheduler.step()
        metrics,records=evaluate(model,loaders['val'],device,lookup,vocab,keys,protocol['validation_seed'])
        auc=metrics['biomarker_macro_auc']
        if not np.isfinite(auc):raise ValueError('Validation molecular AUROC not estimable; no endpoint fallback')
        payload={'model':model.state_dict(),'epoch':epoch,'selection_metric':auc,
            'selection_metric_name':'biomarker_macro_auc','protocol_version':VERSION}
        torch.save(payload,out/'last.pt')
        if auc>best_auc:
            best_auc=auc;torch.save(payload,out/'best_auc.pt')
        if metrics['map']>best_map:
            best_map=metrics['map'];torch.save({**payload,'selection_metric':metrics['map'],'selection_metric_name':'map'},out/'best_map.pt')
        write_json(out/f'validation_epoch_{epoch:03d}.json',{'protocol_version':VERSION,'metrics':metrics,'records':records})
        history.append({'epoch':epoch,'loss':loss_sum/n,'encoded_training_views':views,'validation':metrics})
        write_json(out/'history.json',history)
        print(f'{method} epoch={epoch} loss={loss_sum/n:.5f} val_AUROC={auc:.6f} val_mAP={metrics["map"]:.6f}',flush=True)
    checkpoint=torch.load(out/'best_auc.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    first,records=evaluate(model,loaders['val'],device,lookup,vocab,keys,protocol['validation_seed'])
    second,replay=evaluate(model,loaders['val'],device,lookup,vocab,keys,protocol['validation_seed'])
    if not np.array_equal(records['scores'],replay['scores']):raise ValueError('Fixed validation replay differed')
    prior_counts=np.ones(len(vocab))
    for c in groups['train']:
        for a in base.semantic_anchors(c['metadata']):
            if a['key'] in keys:prior_counts[keys[a['key']]]+=1
    prior,_=score_patients(np.tile(np.log(prior_counts),(len(groups['val']),1)),protocol['splits']['val'],lookup,vocab,keys)
    prior['score_type']='train_log_frequency'
    write_json(out/'validation_best_auc.json',{'metrics':first,'records':records,'selected_epoch':checkpoint['epoch']})
    write_json(out/'frequency_baseline_validation.json',prior)
    summary={'method':method,'selected_epoch':checkpoint['epoch'],'validation_auc':first['biomarker_macro_auc'],
        'validation_map':first['map'],'training_steps':steps,'seconds':time.time()-started,
        'validation_replay_equal':True,'test_evaluated':False}
    write_json(out/'complete.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','pilot'],default='smoke')
    p.add_argument('--reference-campaign',default='output/sota_alignmentv2_20261009')
    p.add_argument('--audit',default='output/pasa_region_audit_20261019_173123')
    p.add_argument('--data-root',default='/root/autodl-tmp/dataset/UTSW-Glioma')
    p.add_argument('--metadata',default='/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv')
    p.add_argument('--out')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--roi-size',type=int,default=96)
    p.add_argument('--z-slices',type=int,default=7);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke-patients',type=int,default=32);p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke-reference',help='Completed smoke directory required before pilot training')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.roi_size,args.z_slices,args.smoke_patients)<1 or args.workers<0:raise ValueError('Invalid sizes')
    if len(set(args.methods))!=len(args.methods):raise ValueError('Duplicate methods')
    if args.dry_run:
        print(json.dumps({'version':VERSION,'methods':args.methods,'epochs':1 if args.stage=='smoke' else args.epochs,
            'test_evaluated':False,'raw_labels':REGION_LABELS,'stage':args.stage},indent=2));return
    if args.stage=='pilot':
        if not args.smoke_reference:raise ValueError('Run smoke first, then pass --smoke-reference <completed directory>')
        smoke_root=Path(args.smoke_reference)
        smoke_complete=read(smoke_root/'complete.json');smoke_protocol=read(smoke_root/'protocol.json')
        if smoke_complete['version']!=VERSION or smoke_complete['stage']!='smoke':raise ValueError('Not a completed smoke of this version')
        if smoke_complete['runs']!=len(args.methods):raise ValueError('Smoke did not complete all requested methods')
        for key,value in [('roi_size',args.roi_size),('z_slices',args.z_slices),('seed',args.seed),('methods',args.methods),
            ('batch_size',args.batch_size),('metadata_sha256',digest(args.metadata)),
            ('original_split_sha256',digest(Path(args.reference_campaign)/'splits.json'))]:
            if smoke_protocol[key]!=value:raise ValueError('Smoke/pilot setting differs: '+key)
        for name,h in smoke_protocol['source_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Source changed since smoke: '+name)
    root=Path(args.out or 'output/pasa_repair_'+args.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    for old in (Path(args.reference_campaign).resolve(),Path(args.audit).resolve()):
        if root==old or old in root.parents or root in old.parents:raise ValueError('Output overlaps prior evidence')
    protocol,manifest,lookup,vocab,keys=prepare(args,root)
    summaries=[]
    for method in args.methods:
        summaries.append(train_arm(args,root,protocol,manifest,lookup,vocab,keys,method))
        write_json(root/'summary.json',summaries)
    write_json(root/'complete.json',{'version':VERSION,'stage':args.stage,'runs':len(summaries),'test_evaluated':False})
    print('COMPLETE: '+str(root),flush=True)


if __name__=='__main__':main()
