"""Exploratory, versioned inputs/models; raw region IDs have no inferred histology."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dataset import percentile_norm
from experiment_dataset import _resize_stack
from experiment_model import ROI2DEncoder, SemanticGraphBuilder, graph_laplacian_consistency
from pasa_region_inputs import select_region_inputs
from recent_alignment import masked_nce

VERSION = 'pasa_repair_pilot_v1'
REGION_LABELS = (1,2,3,4,5)
METHODS = ('whole_roi_mass','whole_roi_field','region_pool_mass','region_pool_field')


@contextmanager
def validation_rng(seed, device='cpu'):
    """Fixed validation noise/loader state without changing training RNG streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # torch.manual_seed also seeds CUDA; preserve every initialized device stream.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.backends.cudnn.flags(deterministic=True,benchmark=False):
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class RepairDataset(Dataset):
    def __init__(self, cases, manifest, roi_size=96, z_slices=7, augment=False):
        self.cases, self.manifest = cases, manifest
        self.roi_size, self.z_slices, self.augment = roi_size,z_slices,augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self,index):
        import nibabel as nib
        case=self.cases[index]
        sid=str(case['subject_id'])
        frozen=self.manifest[sid]
        if hashlib.sha256(Path(frozen['segmentation_file']).read_bytes()).hexdigest()!=frozen['segmentation_sha256']:
            raise ValueError(f'Segmentation bytes changed after freeze: {sid}')
        seg_image=nib.load(frozen['segmentation_file'])
        seg=seg_image.get_fdata(dtype=np.float32)
        selected=select_region_inputs(seg,self.roi_size,self.z_slices,region_labels=REGION_LABELS)
        if (selected['z_indices'] != frozen['z_indices'] or list(selected['region_present'].tolist()) != frozen['region_present']
                or list(selected['bbox']) != frozen['bbox']):
            raise ValueError(f'Input changed after freeze: {sid}')
        y0,y1,x0,x1=selected['bbox']
        images=[]
        for modality in ('t1','t1ce','t2','flair'):
            image_record=frozen['image_files'][modality]
            path=image_record['path']
            stat=Path(path).stat()
            if stat.st_size!=image_record['bytes'] or stat.st_mtime_ns!=image_record['mtime_ns']:
                raise ValueError(f'MRI file changed after freeze: {sid} {modality}')
            image=nib.load(path)
            if image.shape!=seg.shape or not np.allclose(image.affine,seg_image.affine,atol=1e-4,rtol=0):
                raise ValueError(f'Image/segmentation geometry differs: {sid} {modality}')
            volume=percentile_norm(image.get_fdata(dtype=np.float32))
            stack=np.stack([volume[y0:y1+1,x0:x1+1,z] for z in selected['z_indices']])
            images.append(_resize_stack(stack,self.roi_size))
        images=torch.stack(images)
        masks=selected['region_masks']
        if self.augment:
            for axis in (-1,-2):
                if torch.rand(())>.5:
                    images,masks=torch.flip(images,(axis,)),torch.flip(masks,(axis,))
        return {'subject_id':sid,'images':images,'region_masks':masks,'region_present':selected['region_present']}


def masked_mean(values,present):
    if present.dtype!=torch.bool or present.shape!=values.shape[:2] or not present.any(dim=1).all():
        raise ValueError('Each patient needs at least one valid region')
    weights=present.to(values.dtype).unsqueeze(-1)
    return (values*weights).sum(dim=1)/weights.sum(dim=1)


def fieldwise_loss(scores,positives,known,vocab):
    """Equal mean over each patient's observed fields, then over patients."""
    if scores.shape!=positives.shape or scores.shape!=known.shape or torch.any(positives&~known):
        raise ValueError('Positive/known masks must match scores and all positives must be known')
    field_losses,field_valid=[],[]
    for field in dict.fromkeys(a['field'] for a in vocab):
        ids=[i for i,a in enumerate(vocab) if a['field']==field]
        p,k=positives[:,ids],known[:,ids]
        active=p.any(dim=1)
        # Avoid all-infinite logsumexp for missing rows; these rows never contribute.
        s=scores[:,ids]
        safe_known=k.clone(); safe_positive=p.clone()
        safe_known[~active]=True; safe_positive[~active]=True
        values=s.masked_fill(~safe_known,float('-inf')).logsumexp(-1)-s.masked_fill(~safe_positive,float('-inf')).logsumexp(-1)
        field_losses.append(values*active)
        field_valid.append(active)
    active=torch.stack(field_valid,1)
    counts=active.sum(1)
    if not (counts>0).any():
        return scores.sum()*0
    losses=torch.stack(field_losses,1).sum(1)/counts.clamp_min(1)
    return losses[counts>0].mean()


class RepairAlignmentModel(nn.Module):
    """Same shared CNN, head, prototypes and parameter budget in all four arms.

    Whole-ROI sees the union mask; region-pool encodes only nonempty region views.
    Both produce patient scores before the same one-direction supervision.
    Graph is optional for mask verification; pilot arms all disable it.
    """
    def __init__(self,method,num_anchors,z_slices=7,feat_dim=256,shared_dim=128,graph=False):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method,self.use_graph=method,graph
        self.encoder=ROI2DEncoder(4*z_slices,feat_dim)
        self.head=nn.Sequential(nn.Linear(feat_dim,shared_dim),nn.LayerNorm(shared_dim),nn.SiLU())
        self.prototypes=nn.Parameter(torch.randn(num_anchors,shared_dim)*.02)
        self.graph=SemanticGraphBuilder(shared_dim,'learnable') if graph else None

    def forward(self,images,region_masks,region_present):
        actual=region_masks.flatten(2).sum(-1)>0
        if not torch.equal(region_present,actual) or not actual.any(1).all():
            raise ValueError('Presence must match nonempty input masks')
        if self.method.startswith('whole_roi'):
            masks=region_masks.amax(1,keepdim=True)
            present=torch.ones((len(images),1),dtype=torch.bool,device=images.device)
        else:
            masks,present=region_masks,region_present
        # Only nonempty views reach BatchNorm. Missing views have zero gradient.
        patient,node=torch.where(present)
        views=images[patient]*masks[patient,node].unsqueeze(1)
        encoded=self.head(self.encoder(views.flatten(1,2)))
        local=encoded.new_zeros((len(images),masks.shape[1],encoded.shape[-1]))
        local[patient,node]=encoded
        adjacency=local.new_zeros(local.shape[0],local.shape[1],local.shape[1])
        if self.use_graph:
            raw=self.graph(local)
            edge=present[:,:,None]&present[:,None,:]
            edge &= ~torch.eye(local.shape[1],dtype=torch.bool,device=local.device)[None]
            adjacency=raw*edge
            adjacency=adjacency/adjacency.sum(-1,keepdim=True).clamp_min(1e-12)
            local=(local+adjacency@local)*present.unsqueeze(-1)
        pooled=masked_mean(local,present)
        scores=F.normalize(pooled,dim=-1)@F.normalize(self.prototypes,dim=-1).T
        return {'scores':scores,'local':local,'present':present,'adjacency':adjacency,
            'graph_loss':graph_laplacian_consistency(local,adjacency)}

    def loss(self,result,positives,known,vocab):
        scores=result['scores']/.07
        if self.method.endswith('_field'):
            return fieldwise_loss(scores,positives,known,vocab)
        return masked_nce(scores,positives,known,symmetric=False)
