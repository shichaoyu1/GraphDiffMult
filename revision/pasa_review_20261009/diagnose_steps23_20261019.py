"""Post-hoc diagnostics of frozen records; never selects a new test checkpoint."""
from pathlib import Path
import collections
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semantic_evaluation import binary_auc

DATA = ROOT / 'output/20261019_AAAI_review/sota_baline_output'
OUT = Path(__file__).parent / 'steps23_20261019'
OUT.mkdir(exist_ok=True)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


fields, histories, patients, loss_rows, regions = [], [], {}, [], []
campaign = DATA / 'sota_alignmentv2_20261009'
for p in sorted(campaign.glob('runs/sota/*/attempt_001/patient_score_records.json')):
    config, records = read(p.parent / 'config.json'), read(p)
    method, seed = config['method'], config['seed']
    vocab = read(p.parent / 'anchor_vocab.json')
    grouped = collections.defaultdict(list)
    for sid, field, scores, targets, valid in zip(records['subject_ids'], records['field_names'], records['scores'], records['target_ids'], records['valid_ids']):
        if not targets:
            continue
        assert len(targets) == 1 and len(valid) == 2
        # Fixed canonical first candidate; never choose the positive class from predictions.
        a, b = valid
        margin = scores[a] - scores[b]
        truth, pred = int(targets[0] == a), int(margin >= 0)
        grouped[field].append((sid, truth, pred, margin, a, b))
    for field, rows in grouped.items():
        y, pred, margin = [np.array([row[i] for row in rows]) for i in (1, 2, 3)]
        recall_a = float(np.mean(pred[y == 1] == 1))
        recall_b = float(np.mean(pred[y == 0] == 0))
        fields.append({'method': method, 'seed': seed, 'field': field, 'n': len(rows),
            'class_a': vocab[rows[0][4]]['key'], 'class_b': vocab[rows[0][5]]['key'],
            'n_a': int(y.sum()), 'n_b': int((1-y).sum()), 'predicted_a': int(pred.sum()),
            'recall_a': recall_a, 'recall_b': recall_b, 'balanced_accuracy': (recall_a+recall_b)/2,
            'accuracy': float(np.mean(y == pred)), 'auc': binary_auc(y, margin),
            'margin_min': float(margin.min()), 'margin_q25': float(np.quantile(margin,.25)),
            'margin_median': float(np.median(margin)), 'margin_q75': float(np.quantile(margin,.75)),
            'margin_max': float(margin.max()), 'margin_sd': float(margin.std())})
        patients[(method, seed, field)] = {sid: int(y == pred) for sid, y, pred, *_ in rows}
    h = read(p.parent / 'history.json')
    maps = [row['validation']['map'] for row in h]
    aucs = [row['validation']['biomarker_macro_auc'] for row in h]
    chosen, auc_peak = int(np.argmax(maps)), int(np.argmax(aucs))
    histories.append({'method': method, 'seed': seed, 'selected_epoch': chosen+1,
        'map_distinct': len(set(maps)), 'selected_val_auc': aucs[chosen],
        'last_val_auc': aucs[-1], 'max_val_auc': aucs[auc_peak], 'max_val_auc_epoch': auc_peak+1,
        'first_loss': h[0]['loss'], 'last_loss': h[-1]['loss']})

gains=[]
for (method, seed, field), values in patients.items():
    if method == 'pasa_full':
        continue
    base = patients[('pasa_full', seed, field)]
    assert values.keys() == base.keys()
    gains.append({'method': method, 'seed': seed, 'field': field,
        'corrected_vs_full': sum(values[k] > base[k] for k in values),
        'new_errors_vs_full': sum(values[k] < base[k] for k in values),
        'unchanged': sum(values[k] == base[k] for k in values)})

# First campaign contains component magnitudes, but no parameter gradient logs.
for p in sorted((DATA/'pasa_v2_20261009_1707').glob('runs/*/*/attempt_001/history.json')):
    config = read(p.parent/'config.json')
    record = read(p.parent/'patient_level_records.json')
    grouped = collections.defaultdict(dict)
    targets = collections.defaultdict(set)
    for sid, node, vec, field, ids in zip(record['subject_ids'],record['node_names'],record['query_vectors'],record['query_fields'],record['query_targets']):
        grouped[sid][node] = vec
        targets[(sid,field)].add(tuple(ids))
    cosines=[]
    for nodes in grouped.values():
        q=np.array(list(nodes.values()))
        q=q/np.linalg.norm(q,axis=1,keepdims=True).clip(1e-12)
        cosines.extend((q@q.T)[np.triu_indices(len(q),1)])
    if cosines:
        regions.append({'run':p.parts[-3], 'mean_within_patient_region_cosine':float(np.mean(cosines)),
            'region_pairs':len(cosines),'patient_fields_with_different_region_targets':sum(len(v)>1 for v in targets.values())})
    for row in read(p):
        epoch = row['epoch']
        warmup = config['graph_warmup_epochs']
        # Match graph_cons_scale in frozen source (checked separately).
        scale = min(1., epoch/max(1,warmup))
        t = row['train']
        weighted = {key: config['lambda_'+key] * t[key] * (scale if key=='cons' else 1.)
                    for key in ['anchor','cons','decouple','leak','diff','diff_norm','gate_entropy','load_balance']}
        reconstructed = t['alignment'] + sum(weighted.values())
        assert np.isclose(reconstructed,t['total'],rtol=1e-5,atol=1e-6), (p,epoch,reconstructed,t['total'])
        loss_rows.append({'run': p.parts[-3], 'epoch': epoch, 'alignment': t['alignment'],
            **weighted, 'total': t['total'], 'nonfinite_batches': t.get('nonfinite_batches',None)})

write_csv('field_diagnostics.csv',fields)
write_csv('validation_history_diagnostics.csv',histories)
write_csv('paired_field_gains.csv',gains)
write_csv('weighted_loss_history.csv',loss_rows)
write_csv('region_similarity.csv',regions)
prior_dir=campaign/'runs/sota/pasa_full_s42/attempt_001'
prior=read(prior_dir/'train_prior_score_records.json')
vocab=read(prior_dir/'anchor_vocab.json')
train_counts={a['key']:int(round(np.exp(prior['scores'][0][i])-1)) for i,a in enumerate(vocab)}
(OUT/'train_label_counts_from_prior.json').write_text(json.dumps(train_counts,indent=2),encoding='utf-8')
(OUT/'diagnostics.json').write_text(json.dumps({'fields':fields,'validation':histories,'paired_gains':gains},indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,2,figsize=(12,8))
for method, ax in zip(['pasa_full','pasa_minimal','pasa_text_control'],axes.flat[:3]):
    for seed in [42,43,44]:
        h=read(campaign/f'runs/sota/{method}_s{seed}/attempt_001/history.json')
        ax.plot([r['epoch'] for r in h],[r['validation']['biomarker_macro_auc'] for r in h],label=f'seed {seed}')
        selected=next(r['selected_epoch'] for r in histories if r['method']==method and r['seed']==seed)
        ax.scatter(selected,h[selected-1]['validation']['biomarker_macro_auc'],s=45)
    ax.set(title=method+' validation AUROC',xlabel='Epoch',ylabel='Macro AUROC',ylim=(.3,.9))
    ax.legend()
methods=['pasa_full','pasa_minimal','text_cosine','carzero_metadata','radzero_metadata']
for field in ['IDH','MGMT','1p19Q CODEL']:
    axes[1,1].plot(methods,[np.mean([r['auc'] for r in fields if r['method']==m and r['field']==field]) for m in methods],marker='o',label=field)
axes[1,1].set(title='Test AUROC by field (descriptive)',ylabel='AUROC',ylim=(.2,1))
axes[1,1].tick_params(axis='x',rotation=20)
axes[1,1].legend()
fig.tight_layout()
fig.savefig(OUT/'validation_and_field_auc.png',dpi=160)
plt.close(fig)
print(json.dumps(histories,indent=2))
