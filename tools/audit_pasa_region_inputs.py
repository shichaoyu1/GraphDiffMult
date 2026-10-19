"""CPU-only investigation of empty PASA region inputs; no label remapping."""
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from tools.diagnose_pasa_server import read, digest, region_counts
from experiment_dataset import _bbox_from_mask, _z_indices
from dataset import find_modality_file, find_segmentation_file
import train_semantic_alignment as base
from json_results import write_json


def counts(array):
    values, amounts = np.unique(array, return_counts=True)
    return {str(float(v)): int(n) for v, n in zip(values, amounts)}


def inspect_segmentation(seg, roi_size, z_slices):
    if seg.ndim != 3 or not np.isfinite(seg).all():
        raise ValueError('Segmentation must be a finite 3D volume')
    y0,y1,x0,x1,z = _bbox_from_mask(seg)
    indices = _z_indices(z, seg.shape[2], z_slices)
    slab = seg[y0:y1+1, x0:x1+1, indices]
    regions = region_counts(seg, roi_size, z_slices)
    return {'whole_volume_labels': counts(seg), 'sampled_slab_labels': counts(slab),
        'sampled_z_indices': [int(i) for i in indices],
        'whole_positive_voxels': int((seg>0).sum()), 'sampled_positive_voxels': int((slab>0).sum()),
        'sampled_positive_outside_1_2_4': int(((slab>0)&~np.isin(slab,[1,2,4])).sum()),
        'noninteger_voxels': int((seg != np.rint(seg)).sum()),
        'regions': regions, 'all_network_regions_empty': all(r['network_mask_voxels']==0 for r in regions)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics', required=True)
    p.add_argument('--data-root')
    p.add_argument('--metadata')
    p.add_argument('--out')
    cli=p.parse_args()
    root=Path(cli.diagnostics).resolve()
    cfg=read(root/'diagnostic_config.json')
    frozen=cfg['frozen_identity']
    data_root=cli.data_root or cfg['arguments']['data_root']
    metadata=cli.metadata or cfg['arguments']['metadata']
    if digest(metadata)!=frozen['metadata_sha256']:
        raise ValueError('Metadata differs from frozen diagnostic')
    out=Path(cli.out or 'output/pasa_region_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if out==root or root in out.parents or out in root.parents:
        raise ValueError('Output must be outside existing diagnostics')
    out.mkdir(parents=True,exist_ok=False)
    lookup={str(c['subject_id']):c for c in base.discover_semantic_cases(data_root,metadata_tsv=metadata,seed=frozen['sample_seed'])}
    previous={(r['split'],r['subject_id']):r for r in read(root/'region_records.json')}
    write_json(out/'config.json',{'arguments':vars(cli),'source_diagnostics':str(root),
        'metadata_sha256':digest(metadata),'scope':'train/validation segmentation inventory; no remapping or training',
        'source_sha256':{name:digest(ROOT/name) for name in ['tools/audit_pasa_region_inputs.py','tools/diagnose_pasa_server.py','experiment_dataset.py','dataset.py']}})
    import nibabel as nib
    records=[]
    for split in ('train','val'):
        for i,sid in enumerate(frozen['splits'][split],1):
            sid=str(sid)
            case=lookup[sid]
            image_path=find_modality_file(case['patient_dir'],'t1',prefer_registered=False)
            reference=nib.load(image_path)
            seg_path=find_segmentation_file(case['patient_dir'],image_shape=tuple(reference.shape))
            seg_image=nib.load(seg_path)
            seg=seg_image.get_fdata(dtype=np.float32)
            details=inspect_segmentation(seg,frozen['roi_size'],frozen['z_slices'])
            if details['regions']!=previous[(split,sid)]['regions']:
                raise ValueError(f'Region counts changed since previous diagnostic: {sid}')
            records.append({'split':split,'subject_id':sid,'segmentation_file':str(seg_path),
                'image_file':str(image_path),'segmentation_sha256':digest(seg_path),
                'shape_matches_reference':tuple(seg.shape)==tuple(reference.shape),
                'affine_matches_reference':bool(np.allclose(seg_image.affine,reference.affine,atol=1e-4,rtol=0)),**details})
            if i%50==0:
                print(f'{split}: {i}/{len(frozen["splits"][split])}',flush=True)
                write_json(out/'records.json',records)
    summary={}
    for split in ('train','val'):
        rows=[r for r in records if r['split']==split]
        empty=[r for r in rows if r['all_network_regions_empty']]
        summary[split]={'patients':len(rows),'all_region_empty':len(empty),
            'all_empty_with_empty_full_segmentation':sum(r['whole_positive_voxels']==0 for r in empty),
            'all_empty_with_empty_sampled_slab':sum(r['whole_positive_voxels']>0 and r['sampled_positive_voxels']==0 for r in empty),
            'all_empty_with_unhandled_sampled_labels':sum(r['sampled_positive_outside_1_2_4']>0 for r in empty),
            'all_empty_despite_sampled_known_label':sum(any(a['sampled_native_voxels']>0 for a in r['regions']) for r in empty),
            'noninteger_segmentations':sum(r['noninteger_voxels']>0 for r in rows),
            'affine_mismatches':sum(not r['affine_matches_reference'] for r in rows),
            'whole_label_case_counts':dict(Counter(label for r in rows for label in r['whole_volume_labels']))}
    write_json(out/'records.json',records)
    write_json(out/'summary.json',summary)
    write_json(out/'complete.json',{'status':'complete','patients':len(records),'test_evaluated':False})
    print(summary,flush=True)
    print('COMPLETE: '+str(out),flush=True)


if __name__=='__main__':
    main()
