"""Deterministic inputs from explicitly accepted raw segmentation labels.

Labels remain separate numerical bins; this module assigns no tissue meanings.
For a nonempty [Y, X, Z] segmentation, select the lowest-index slice among
those with the largest accepted-label area. The resulting slab always contains
that slice; boundary indices repeat within the same volume. The in-plane crop
uses the accepted-label union across the volume, plus a fixed margin.

Nearest-neighbor resizing may remove small bins. Report presence only after
resizing, expose the lost bins in the audit, and reject a completely empty input.
Unknown labels, non-integral values, and empty accepted unions raise errors;
there is no patient substitution or empty-volume fallback. Callers must freeze
their accepted labels and exclusion rules in the experiment manifest.
"""

import numbers

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_REGION_LABELS = (1, 2, 3, 4)


class EmptyRegionInputError(ValueError):
    """The accepted-label input is empty before or after resizing."""


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def validate_segmentation_labels(seg, region_labels=DEFAULT_REGION_LABELS):
    """Validate a finite 3D integer-valued array and return raw label counts.

    Background is 0. Every nonzero value must occur in ``region_labels``.
    Explicitly including 5 preserves a fifth raw bin; it does not map it to
    another label. The order of ``region_labels`` defines mask channel order.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3 or any(size == 0 for size in seg.shape):
        raise ValueError('seg must be a nonempty [Y, X, Z] array')
    if not np.issubdtype(seg.dtype, np.number) or np.iscomplexobj(seg):
        raise ValueError('seg must contain numeric integer-valued labels')
    if not np.isfinite(seg).all() or not np.equal(seg, np.floor(seg)).all():
        raise ValueError('seg contains nonfinite or non-integral labels')
    labels = tuple(_integer(label, 'region label', 1) for label in region_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('region_labels must be nonempty and unique')
    values, counts = np.unique(seg, return_counts=True)
    label_counts = {str(int(value)): int(count) for value, count in zip(values, counts)}
    unknown = sorted(int(value) for value in values if value != 0 and value not in labels)
    if unknown:
        raise ValueError(f'Unaccepted raw segmentation labels: {unknown}; accepted={labels}')
    return {'region_labels': labels, 'raw_label_counts': label_counts}


def lesion_slab_plan(seg, z_slices=7, margin=8, region_labels=DEFAULT_REGION_LABELS):
    """Return inclusive crop bounds and deterministic slab indices.

    The center maximizes accepted-label area, breaking ties toward the smaller
    z index. Empty volumes raise ``EmptyRegionInputError``. No labels or data
    are changed. ``bbox`` is (y_min, y_max, x_min, x_max), all inclusive.
    """
    z_slices = _integer(z_slices, 'z_slices', 1)
    margin = _integer(margin, 'margin', 0)
    audit = validate_segmentation_labels(seg, region_labels)
    seg = np.asarray(seg)
    union = np.isin(seg, audit['region_labels'])
    area_by_z = union.sum(axis=(0, 1))
    if not area_by_z.any():
        raise EmptyRegionInputError('No accepted-label voxels in the segmentation volume')
    center = int(np.argmax(area_by_z))
    ys = np.flatnonzero(union.any(axis=(1, 2)))
    xs = np.flatnonzero(union.any(axis=(0, 2)))
    bbox = (max(0, int(ys[0]) - margin), min(seg.shape[0] - 1, int(ys[-1]) + margin),
            max(0, int(xs[0]) - margin), min(seg.shape[1] - 1, int(xs[-1]) + margin))
    offsets = np.arange(z_slices) - z_slices // 2
    indices = np.clip(center + offsets, 0, seg.shape[2] - 1).astype(int).tolist()
    return {'bbox': bbox, 'z_center': center, 'z_indices': indices,
            'region_labels': audit['region_labels'],
            'audit': {**audit, 'accepted_area_by_z': area_by_z.astype(int).tolist(),
                      'z_center_tie_rule': 'lowest_z_index', 'empty_policy': 'error'}}


def select_region_inputs(seg, roi_size=96, z_slices=7, margin=8,
                         region_labels=DEFAULT_REGION_LABELS):
    """Build separate raw-label masks and post-resize presence flags.

    Returns the slab plan plus ``region_masks`` float32 [R,Z,H,W],
    ``region_present`` bool [R], and ``whole_lesion_mask`` float32 [Z,H,W].
    All masks use the same crop and nearest-neighbor resize. ``audit`` contains
    per-channel voxel counts before/after resizing and labels lost at resize.
    A nonempty selected slab that becomes entirely empty at resize is rejected.
    The helper performs no image IO or intensity processing.
    """
    roi_size = _integer(roi_size, 'roi_size', 1)
    plan = lesion_slab_plan(seg, z_slices=z_slices, margin=margin, region_labels=region_labels)
    y_min, y_max, x_min, x_max = plan['bbox']
    slab = np.asarray(seg)[y_min:y_max + 1, x_min:x_max + 1, :][:, :, plan['z_indices']]
    masks = np.stack([(slab == label).transpose(2, 0, 1)
                      for label in plan['region_labels']], axis=0)
    before = masks.sum(axis=(1, 2, 3)).astype(int)
    resized = F.interpolate(torch.from_numpy(masks.astype(np.float32)),
                            size=(roi_size, roi_size), mode='nearest')
    after = resized.sum(dim=(1, 2, 3)).to(torch.long)
    present = after > 0
    if not bool(present.any()):
        raise EmptyRegionInputError('All accepted-label voxels disappeared during nearest resize')
    labels = plan['region_labels']
    plan['audit'].update({
        'slab_voxel_counts_before_resize': {str(label): int(count) for label, count in zip(labels, before)},
        'slab_voxel_counts_after_resize': {str(label): int(count) for label, count in zip(labels, after)},
        'labels_lost_at_resize': [label for label, count_before, count_after in zip(labels, before, after)
                                  if count_before > 0 and int(count_after) == 0],
    })
    return {**plan, 'region_masks': resized, 'region_present': present,
            'whole_lesion_mask': resized.amax(dim=0)}
