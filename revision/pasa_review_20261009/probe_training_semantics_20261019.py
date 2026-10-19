"""Synthetic probes of unchanged training code; not patient experiments."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from recent_alignment import masked_nce, RecentAlignmentModel

torch.set_num_threads(2)
out={}
# Two positives from different fields; one easy positive can satisfy the set loss.
positive=torch.tensor([[True,True,False,False]])
known=torch.ones_like(positive)
probes=[]
for per_positive in [False,True]:
    scores=torch.tensor([[10.,0.,2.,2.]],requires_grad=True)
    loss=masked_nce(scores,positive,known,per_positive=per_positive,symmetric=False)
    loss.backward()
    probes.append({'per_positive':per_positive,'loss':loss.item(),'gradient':scores.grad.tolist(),
        'easy_to_hard_positive_gradient_ratio':abs(scores.grad[0,0].item()/scores.grad[0,1].item())})
assert probes[0]['easy_to_hard_positive_gradient_ratio']>20000
assert probes[1]['easy_to_hard_positive_gradient_ratio']<.01
out['positive_competition_synthetic']=probes
# Unknown labels must have no effect on loss or gradient.
s=torch.tensor([[1.,-1.,100.]],requires_grad=True)
p=torch.tensor([[True,False,False]]);k=torch.tensor([[True,True,False]])
loss=masked_nce(s,p,k,symmetric=False);loss.backward()
assert s.grad[0,2].item()==0
out['unknown_mask_gradient']=s.grad.tolist()

# Use real selected weights but synthetic images, never infer clinical performance.
campaign=ROOT/'output/20261019_AAAI_review/sota_baline_output/sota_alignmentv2_20261009'
graph_checks=[]
for seed in [42,43,44]:
    folder=campaign/f'runs/sota/pasa_full_s{seed}/attempt_001'
    torch.manual_seed(seed)
    model=RecentAlignmentModel('pasa_full',26,z_slices=7)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('encoder.graph_builder.')}
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state)
    differences={n:float((state[n]-v).abs().max()) for n,v in initial.items()}
    graph_checks.append({'seed':seed,'graph_parameter_max_abs_change':max(differences.values()),'parameters':differences})
out['selected_graph_weights_vs_initialization']=graph_checks
torch.manual_seed(20261019)
images=torch.randn(2,4,7,32,32)
masks=torch.ones(2,3,7,32,32)
model.eval()
with torch.no_grad():
    torch.manual_seed(100)
    first=model(images,region_masks=masks)['scores']
    second=model(images,region_masks=masks)['scores']
    torch.manual_seed(100)
    replay=model(images,region_masks=masks)['scores']
out['synthetic_eval_repetition']={'different_rng_max_abs_difference':float((first-second).abs().max()),
    'reset_rng_max_abs_difference':float((first-replay).abs().max())}
assert not torch.equal(first,second)
assert torch.equal(first,replay)

# Graph warmup blocks graph-builder gradients, but does not remove message passing.
model.train()
model.zero_grad(set_to_none=True)
result=model(images,region_masks=masks,freeze_graph=True)
result['scores'].square().sum().backward()
out['graph_warmup_synthetic']={'builder_parameters_with_gradient':sum(p.grad is not None for p in model.encoder.graph_builder.parameters()),
    'shared_heads_with_gradient':sum(p.grad is not None for p in model.encoder.shared_heads.parameters())}
assert out['graph_warmup_synthetic']['builder_parameters_with_gradient']==0

path=Path(__file__).parent/'steps23_20261019/training_probes.json'
path.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(out,indent=2))
