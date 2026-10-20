"""Repeat a completed frozen pilot with new training seeds, without redefining inputs."""
import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from tools import run_pasa_repair_pilot as pilot
from json_results import write_json

REFERENCE_FILES=('protocol.json','inputs.json','anchor_vocab.json','complete.json','summary.json')


def verify_completed(folder,protocol,require_root_complete=True):
    """A directory alone is not evidence of a completed training repeat."""
    if require_root_complete:
        complete=pilot.read(folder/'complete.json')
        if complete!={'version':pilot.VERSION,'stage':'pilot','runs':len(pilot.METHODS),'test_evaluated':False}:
            raise ValueError('Incomplete or incompatible pilot: '+str(folder))
    summaries=pilot.read(folder/'summary.json')
    if {r['method'] for r in summaries}!=set(pilot.METHODS) or len(summaries)!=len(pilot.METHODS):
        raise ValueError('Missing or duplicated methods')
    for method in pilot.METHODS:
        run=folder/method
        if pilot.read(run/'config.json')['protocol']!=protocol:
            raise ValueError('Method protocol differs: '+method)
        h=pilot.read(run/'history.json')
        if [r['epoch'] for r in h]!=list(range(1,protocol['epochs']+1)):
            raise ValueError('Incomplete epoch history: '+method)
        if any(r['validation']['protocol_version']!=pilot.VERSION or not np.isfinite(r['validation']['biomarker_macro_auc']) for r in h):
            raise ValueError('Invalid validation endpoint: '+method)
        selected=h[int(np.argmax([r['validation']['biomarker_macro_auc'] for r in h]))]['epoch']
        result=pilot.read(run/'complete.json')
        saved=pilot.read(run/'validation_best_auc.json')
        if result!=next(r for r in summaries if r['method']==method):raise ValueError('Summary differs')
        if (result['test_evaluated'] or not result['validation_replay_equal'] or result['selected_epoch']!=selected
                or saved['selected_epoch']!=selected or not np.isfinite(result['validation_auc'])
                or result['validation_auc']!=h[selected-1]['validation']['biomarker_macro_auc']
                or result['validation_map']!=h[selected-1]['validation']['map']):
            raise ValueError('Completion/selection mismatch: '+method)
        expected_steps=((len(protocol['splits']['train'])+protocol['batch_size']-1)//protocol['batch_size'])*protocol['epochs']
        if result['training_steps']!=expected_steps:raise ValueError('Training budget differs')
        if set(saved['records']['subject_ids'])!=set(protocol['splits']['val']):raise ValueError('Validation membership differs')
        if saved['metrics']!=h[selected-1]['validation']:raise ValueError('Best metrics differ from history')
        if saved['records']!=pilot.read(run/f'validation_epoch_{selected:03d}.json')['records']:
            raise ValueError('Selected patient scores differ')
        for name in ('best_auc.pt','best_map.pt','last.pt'):
            if (run/name).stat().st_size==0:raise ValueError('Empty checkpoint: '+name)
        for epoch in range(1,protocol['epochs']+1):
            if not (run/f'validation_epoch_{epoch:03d}.json').is_file():raise ValueError('Missing epoch predictions')
    return summaries


def load_reference(reference,check_data=True):
    protocol=pilot.read(reference/'protocol.json')
    if (protocol['version']!=pilot.VERSION or protocol['stage']!='pilot' or protocol['seed']!=42
            or protocol['epochs']!=10 or protocol['methods']!=list(pilot.METHODS)):
        raise ValueError('Expected completed seed-42 four-arm 10-epoch pilot')
    verify_completed(reference,protocol)
    if set(protocol['source_sha256'])!=set(pilot.SOURCES):raise ValueError('Scientific source manifest incomplete')
    for name,expected in protocol['source_sha256'].items():
        if pilot.digest(ROOT/name)!=expected:raise ValueError('Scientific source changed: '+name)
    manifest=pilot.read(reference/'inputs.json')
    vocab=pilot.read(reference/'anchor_vocab.json')
    splits=protocol['splits']
    if set(splits)!={'train','val'} or any(len(ids)!=len(set(ids)) or not ids for ids in splits.values()):
        raise ValueError('Invalid frozen train/val splits')
    if set(splits['train'])&set(splits['val']):raise ValueError('Train/validation overlap')
    if set(manifest)!=set(splits['train'])|set(splits['val']):raise ValueError('Input membership differs')
    for split,ids in splits.items():
        for sid in ids:
            if manifest[sid]['split']!=split or not any(manifest[sid]['region_present']):
                raise ValueError('Invalid frozen region input: '+sid)
    lookup=keys=None
    if check_data:
        if pilot.digest(protocol['metadata'])!=protocol['metadata_sha256']:raise ValueError('Metadata changed')
        cases=pilot.base.discover_semantic_cases(protocol['data_root'],metadata_tsv=protocol['metadata'],seed=42)
        lookup={str(c['subject_id']):c for c in cases}
        built,keys=pilot.base.build_anchor_vocab([lookup[s] for s in splits['train']])
        if built!=vocab:raise ValueError('Frozen train vocabulary changed')
        for sid,row in manifest.items():
            if pilot.digest(row['segmentation_file'])!=row['segmentation_sha256']:raise ValueError('Segmentation changed: '+sid)
            for image in row['image_files'].values():
                stat=Path(image['path']).stat()
                if stat.st_size!=image['bytes'] or stat.st_mtime_ns!=image['mtime_ns']:
                    raise ValueError('MRI identity changed: '+sid)
    return protocol,manifest,vocab,lookup,keys


def aggregate(reference,completed):
    rows=[{'seed':42,**r} for r in pilot.read(reference/'summary.json')]
    for seed,folder in completed.items():
        rows.extend({'seed':seed,**r} for r in pilot.read(folder/'summary.json'))
    stats={}
    for method in pilot.METHODS:
        aucs=[r['validation_auc'] for r in rows if r['method']==method]
        stats[method]={'seeds':len(aucs),'validation_auc_mean':float(np.mean(aucs)),
            'training_seed_sd':float(np.std(aucs,ddof=1)) if len(aucs)>1 else None}
    return {'scope':'training randomness on one reused development validation split; not independent generalization',
        'runs':rows,'method_summary':stats}


@contextmanager
def campaign_lock(out):
    path=out/'.run.lock'
    try:
        file=path.open('x',encoding='utf-8')
    except FileExistsError:
        raise ValueError('Campaign is locked. Check the recorded PID before removing a stale .run.lock')
    with file:file.write(str(os.getpid()))
    try:yield
    finally:path.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',default='output/pasa_repair_pilot_20261020_095259_432520')
    p.add_argument('--seeds',nargs='+',type=int,default=[43,44])
    p.add_argument('--out')
    p.add_argument('--resume',action='store_true',help='Skip verified completed seeds; retry incomplete seeds in new attempts')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 or s==42 for s in args.seeds):raise ValueError('Unique new nonnegative seeds required')
    reference=Path(args.reference).resolve()
    protocol,manifest,vocab,lookup,keys=load_reference(reference,check_data=not args.dry_run)
    if args.dry_run:
        print({'seeds':args.seeds,'methods':protocol['methods'],'epochs':protocol['epochs'],
            'new_runs':len(args.seeds)*len(pilot.METHODS),'splits':{k:len(v) for k,v in protocol['splits'].items()},'test_evaluated':False});return
    if args.resume and not args.out:raise ValueError('--resume needs an explicit --out')
    out=Path(args.out or 'output/pasa_repair_repeats_'+datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if out==reference or reference in out.parents or out in reference.parents:raise ValueError('Output overlaps reference evidence')
    identity={'version':'pasa_repair_repeats_v1','seeds':args.seeds,'reference':str(reference),
        'reference_sha256':{n:pilot.digest(reference/n) for n in REFERENCE_FILES},
        'coordinator_sha256':pilot.digest(Path(__file__))}
    if out.exists():
        if not args.resume or pilot.read(out/'repeat_config.json')['identity']!=identity:raise ValueError('Output exists or resume identity differs')
    else:
        if args.resume:raise ValueError('Resume output does not exist')
        out.mkdir(parents=True)
        write_json(out/'repeat_config.json',{'identity':identity,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    with campaign_lock(out):
        run_seeds(args.seeds,out,reference,identity,protocol,manifest,lookup,vocab,keys)


def run_seeds(seeds,out,reference,identity,protocol,manifest,lookup,vocab,keys):
    completed={}
    for seed in seeds:
        child=deepcopy(protocol)
        child['seed']=seed
        child['repeat_reference']={'path':str(reference),'protocol_sha256':identity['reference_sha256']['protocol.json']}
        seed_root=out/f'seed_{seed}';seed_root.mkdir(exist_ok=True)
        previous=sorted(seed_root.glob('attempt_[0-9][0-9][0-9]'))
        done=None
        for attempt in previous:
            if (attempt/'complete.json').exists():
                if pilot.read(attempt/'protocol.json')!=child:raise ValueError('Completed repeat protocol changed')
                for name in ('inputs.json','anchor_vocab.json'):
                    if pilot.digest(attempt/name)!=identity['reference_sha256'][name]:
                        raise ValueError('Completed repeat frozen artifact changed: '+name)
                verify_completed(attempt,child);done=attempt;break
        if done is None:
            attempt=seed_root/f'attempt_{max([int(d.name[-3:]) for d in previous],default=0)+1:03d}'
            attempt.mkdir()
            write_json(attempt/'protocol.json',child)
            shutil.copyfile(reference/'inputs.json',attempt/'inputs.json')
            shutil.copyfile(reference/'anchor_vocab.json',attempt/'anchor_vocab.json')
            settings=argparse.Namespace(**{k:child[k] for k in ('cpu','roi_size','z_slices','batch_size','workers')})
            summaries=[]
            try:
                for method in pilot.METHODS:
                    print(f'RUN seed={seed} method={method} -> {attempt/method}',flush=True)
                    summaries.append(pilot.train_arm(settings,attempt,child,manifest,lookup,vocab,keys,method))
                    write_json(attempt/'summary.json',summaries)
                verify_completed(attempt,child,require_root_complete=False)
                write_json(attempt/'complete.json',{'version':pilot.VERSION,'stage':'pilot','runs':4,'test_evaluated':False})
            except Exception as error:
                write_json(attempt/'failed.json',{'seed':seed,'error':str(error),'retry_policy':'new attempt; no optimizer-state continuation'})
                raise
            done=attempt
        completed[seed]=done
        write_json(out/'summary.json',aggregate(reference,completed))
    write_json(out/'complete.json',{'version':identity['version'],'new_seeds':seeds,
        'new_runs':4*len(seeds),'test_evaluated':False})
    print('COMPLETE: '+str(out),flush=True)


if __name__=='__main__':main()
