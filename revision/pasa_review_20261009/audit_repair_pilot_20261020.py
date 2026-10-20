"""Read-only audit of returned pilot scores, selection and coverage."""
import json
from pathlib import Path
import sys
import collections
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from semantic_evaluation import score_retrieval_metrics,binary_auc

DATA=ROOT/'output/20261019_AAAI_review/pasa_repair_pilot_20261020_095259_432520'
OUT=Path(__file__).parent/'repair_pilot_audit_20261020'
OUT.mkdir(exist_ok=True)


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def rescore(payload,expected_patients):
    r,m=payload['records'],payload['metrics']
    assert set(r['subject_ids'])==set(expected_patients)
    recomputed=score_retrieval_metrics(r['scores'],r['target_ids'],subject_ids=r['subject_ids'],
        valid_ids=r['valid_ids'],positive_counts=r['positive_counts'],field_names=r['field_names'])
    for key in ('hit@1','map','mrr','pair_auc'):
        assert np.isclose(recomputed[key],m[key]),key
    fields={}
    for field in ('IDH','MGMT','1p19Q CODEL'):
        rows=[]
        for scores,targets,valid,f in zip(r['scores'],r['target_ids'],r['valid_ids'],r['field_names']):
            if f==field and targets:
                assert len(targets)==1 and len(valid)==2
                margin=scores[valid[0]]-scores[valid[1]]
                rows.append((int(targets[0]==valid[0]),int(margin>=0),margin))
        y,p,s=np.array(rows).T
        auc=binary_auc(y,s)
        assert np.isclose(auc,m['biomarker_auc_by_field'][field]['macro_one_vs_rest_auc'])
        recalls=[float((p[y==label]==label).mean()) for label in (1,0)]
        fields[field]={'patients':len(rows),'class_a_count':int(y.sum()),'class_b_count':int((1-y).sum()),
            'predicted_a':int(p.sum()),'recall_a':recalls[0],'recall_b':recalls[1],
            'balanced_accuracy':float(np.mean(recalls)),'auc':auc}
    assert np.isclose(np.mean([v['auc'] for v in fields.values()]),m['biomarker_macro_auc'])
    return fields


protocol=read(DATA/'protocol.json')
inputs=read(DATA/'inputs.json')
result={'version':protocol['version'],'stage':protocol['stage'],'seed':protocol['seed'],
    'split_sizes':{k:len(v) for k,v in protocol['splits'].items()},'input_coverage':protocol['input_coverage'],
    'empty_retained_inputs':sum(not any(r['region_present']) for r in inputs.values()),'arms':{}}
assert result['empty_retained_inputs']==0
assert set(inputs)==set(protocol['splits']['train'])|set(protocol['splits']['val'])
assert not set(protocol['splits']['train'])&set(protocol['splits']['val'])
for method in protocol['methods']:
    folder=DATA/method
    config=read(folder/'config.json')
    assert config['protocol']==protocol
    history=read(folder/'history.json')
    assert len(history)==10
    for epoch in history:
        payload=read(folder/f'validation_epoch_{epoch["epoch"]:03d}.json')
        rescore(payload,protocol['splits']['val'])
        assert payload['metrics']==epoch['validation']
    selected=int(np.argmax([r['validation']['biomarker_macro_auc'] for r in history]))+1
    best=read(folder/'validation_best_auc.json')
    fields=rescore(best,protocol['splits']['val'])
    assert best['selected_epoch']==selected
    saved=read(folder/f'validation_epoch_{selected:03d}.json')
    assert saved['records']['scores']==best['records']['scores']
    complete=read(folder/'complete.json')
    assert complete['selected_epoch']==selected and complete['validation_replay_equal'] and not complete['test_evaluated']
    result['arms'][method]={'selected_epoch':selected,'parameters':config['parameter_count'],
        'best_auc':best['metrics']['biomarker_macro_auc'],'best_map':best['metrics']['map'],
        'last_auc':history[-1]['validation']['biomarker_macro_auc'],'fields':fields,
        'auc_trajectory':[r['validation']['biomarker_macro_auc'] for r in history],
        'map_trajectory':[r['validation']['map'] for r in history],
        'loss_trajectory':[r['loss'] for r in history], 'seconds':complete['seconds']}
assert len({r['parameters'] for r in result['arms'].values()})==1
result['comparisons']={
    'field_minus_mass_whole_roi':result['arms']['whole_roi_field']['best_auc']-result['arms']['whole_roi_mass']['best_auc'],
    'field_minus_mass_region_pool':result['arms']['region_pool_field']['best_auc']-result['arms']['region_pool_mass']['best_auc'],
    'region_minus_whole_roi_field':result['arms']['region_pool_field']['best_auc']-result['arms']['whole_roi_field']['best_auc'],
    'region_minus_whole_roi_mass':result['arms']['region_pool_mass']['best_auc']-result['arms']['whole_roi_mass']['best_auc']}
result['rescored_epoch_files']=40
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for method,arm in result['arms'].items():
    line=axes[0].plot(range(1,11),arm['auc_trajectory'],marker='.',label=method)[0]
    axes[0].scatter(arm['selected_epoch'],arm['best_auc'],color=line.get_color(),s=55)
    axes[1].plot(range(1,11),arm['loss_trajectory'],marker='.',label=method)
axes[0].set(xlabel='Epoch',ylabel='Validation molecular macro AUROC',ylim=(.5,.85),title='Seed 42: dots mark selected checkpoints')
axes[1].set(xlabel='Epoch',ylabel='Training loss',title='Different objectives: loss scales not comparable')
axes[0].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'pilot_trajectories.png',dpi=160);plt.close(fig)
print(json.dumps({'rescored':40,'comparisons':result['comparisons'],'coverage':result['split_sizes']},indent=2))
