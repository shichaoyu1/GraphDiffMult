"""
Patient-level datasets for the UTSW-Glioma molecular/grading experiments.

This module implements the first practical version of the experiment plan:
2.5D tumor ROI crops from T1, T1ce, T2, and FLAIR, with labels derived from
UTSW-Glioma metadata.
"""

import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset import (
    find_modality_file,
    find_segmentation_file,
    get_utsw_cases,
    load_utsw_metadata,
    percentile_norm,
)


MODALITIES = ('t1', 't1ce', 't2', 'flair')


def _clean_text(value):
    if value is None:
        return ''
    return str(value).strip().lower()


def parse_utsw_label(metadata: dict, task: str):
    task = task.lower()

    if task == 'idh':
        value = _clean_text(metadata.get('IDH'))
        if value in {'mutated', 'mutant'}:
            return 1
        if value in {'wild type', 'wildtype', 'wt'}:
            return 0
        return None

    if task == 'mgmt':
        value = _clean_text(metadata.get('MGMT'))
        if value == 'methylated':
            return 1
        if value == 'unmethylated':
            return 0
        return None

    if task == '1p19q':
        value = _clean_text(metadata.get('1p19Q CODEL'))
        if value in {'co-deleted', 'codeleted', 'co deleted'}:
            return 1
        if value in {'non co-deleted', 'non-codeleted', 'non co deleted'}:
            return 0
        return None

    if task == 'grade':
        value = _clean_text(metadata.get('Tumor Grade'))
        if value in {'2', 'ii'}:
            return 0
        if value in {'3', 'iii'}:
            return 1
        if value in {'4', 'iv'}:
            return 2
        return None

    raise ValueError(f'Unsupported task: {task}')


def label_names_for_task(task: str):
    names = {
        'idh': ['IDH-wildtype', 'IDH-mutant'],
        'mgmt': ['MGMT-unmethylated', 'MGMT-methylated'],
        '1p19q': ['1p19q-non-codeleted', '1p19q-codeleted'],
        'grade': ['WHO-grade-2', 'WHO-grade-3', 'WHO-grade-4'],
    }
    return names[task.lower()]


def discover_utsw_labeled_cases(
    root_dir: str,
    task: str = 'idh',
    metadata_tsv: str = None,
    max_cases: int = None,
    seed: int = 42,
):
    metadata_path = metadata_tsv
    if metadata_path is None:
        sibling = os.path.join(os.path.dirname(root_dir), 'UTSW_Glioma_Metadata-2-1.tsv')
        metadata_path = sibling if os.path.exists(sibling) else None

    metadata = load_utsw_metadata(metadata_path) if metadata_path else {}
    cases = []
    for case in get_utsw_cases(root_dir, metadata_tsv=metadata_path):
        info = metadata.get(case['subject_id'], case.get('metadata', {}))
        label = parse_utsw_label(info, task)
        if label is None:
            continue
        case = dict(case)
        case['label'] = int(label)
        case['metadata'] = info
        cases.append(case)

    if max_cases and len(cases) > max_cases:
        rng = np.random.default_rng(seed)
        grouped = defaultdict(list)
        for case in cases:
            grouped[case['label']].append(case)
        sampled = []
        for group in grouped.values():
            rng.shuffle(group)
        while len(sampled) < max_cases and any(grouped.values()):
            for label in sorted(grouped):
                if grouped[label] and len(sampled) < max_cases:
                    sampled.append(grouped[label].pop())
        cases = sorted(sampled, key=lambda item: item['subject_id'])

    return cases


def stratified_split(cases, train_ratio=0.7, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case['label']].append(case)

    splits = {'train': [], 'val': [], 'test': []}
    for label_cases in grouped.values():
        label_cases = list(label_cases)
        rng.shuffle(label_cases)
        n_cases = len(label_cases)
        n_train = max(1, int(round(n_cases * train_ratio))) if n_cases else 0
        n_val = int(round(n_cases * val_ratio)) if n_cases >= 5 else 0
        if n_train + n_val >= n_cases and n_cases > 1:
            n_train = n_cases - 1
            n_val = 0
        splits['train'].extend(label_cases[:n_train])
        splits['val'].extend(label_cases[n_train:n_train + n_val])
        splits['test'].extend(label_cases[n_train + n_val:])

    for split_cases in splits.values():
        split_cases.sort(key=lambda item: item['subject_id'])
    return splits


def describe_cases(cases):
    return dict(sorted(Counter(case['label'] for case in cases).items()))


def _bbox_from_mask(seg, margin=8):
    coords = np.argwhere(seg > 0)
    if len(coords) == 0:
        center = np.array(seg.shape) // 2
        return (0, seg.shape[0] - 1, 0, seg.shape[1] - 1, int(center[2]))

    y_min, x_min, z_min = coords.min(axis=0)
    y_max, x_max, z_max = coords.max(axis=0)
    y_min = max(0, int(y_min) - margin)
    x_min = max(0, int(x_min) - margin)
    y_max = min(seg.shape[0] - 1, int(y_max) + margin)
    x_max = min(seg.shape[1] - 1, int(x_max) + margin)
    z_center = int(round((int(z_min) + int(z_max)) / 2))
    return y_min, y_max, x_min, x_max, z_center


def _z_indices(z_center, depth, z_slices):
    half = z_slices // 2
    indices = [z_center + offset for offset in range(-half, half + 1)]
    if len(indices) > z_slices:
        indices = indices[:z_slices]
    while len(indices) < z_slices:
        indices.append(indices[-1] if indices else z_center)
    return [int(np.clip(index, 0, depth - 1)) for index in indices]


def _resize_stack(stack, roi_size):
    tensor = torch.from_numpy(stack.astype(np.float32)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(roi_size, roi_size), mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)


def _resize_mask_stack(stack, roi_size):
    tensor = torch.from_numpy(stack.astype(np.float32)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(roi_size, roi_size), mode='nearest')
    return tensor.squeeze(0)


class UTSWROIPatientDataset(Dataset):
    """
    Return one patient-level sample:
        images: [M, z_slices, roi_size, roi_size]
        label: scalar class index
    """

    def __init__(
        self,
        cases,
        roi_size: int = 96,
        z_slices: int = 7,
        modalities=MODALITIES,
        prefer_registered: bool = False,
        augment: bool = False,
        cache: bool = False,
    ):
        self.cases = list(cases)
        self.roi_size = roi_size
        self.z_slices = z_slices
        self.modalities = tuple(modalities)
        self.prefer_registered = prefer_registered
        self.augment = augment
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.cases)

    def _load_case(self, case):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError('Please install nibabel: pip install nibabel')

        patient_dir = case['patient_dir']
        first_modality = find_modality_file(
            patient_dir,
            self.modalities[0],
            prefer_registered=self.prefer_registered,
        )
        image_shape = tuple(nib.load(first_modality).shape)
        seg_path = find_segmentation_file(patient_dir, image_shape=image_shape)
        seg = nib.load(seg_path).get_fdata(dtype=np.float32)

        y_min, y_max, x_min, x_max, z_center = _bbox_from_mask(seg)
        z_ids = _z_indices(z_center, seg.shape[2], self.z_slices)

        stacks = []
        for modality in self.modalities:
            path = find_modality_file(
                patient_dir,
                modality,
                prefer_registered=self.prefer_registered,
            )
            volume = nib.load(path).get_fdata(dtype=np.float32)
            volume = percentile_norm(volume)
            slices = []
            for z_idx in z_ids:
                crop = volume[y_min:y_max + 1, x_min:x_max + 1, z_idx]
                slices.append(crop)
            stacks.append(_resize_stack(np.stack(slices, axis=0), self.roi_size))

        region_masks = []
        for label_value in [1, 2, 4]:
            mask_slices = []
            for z_idx in z_ids:
                crop = (seg[y_min:y_max + 1, x_min:x_max + 1, z_idx] == label_value).astype(np.float32)
                mask_slices.append(crop)
            region_masks.append(_resize_mask_stack(np.stack(mask_slices, axis=0), self.roi_size))

        return torch.stack(stacks, dim=0), torch.stack(region_masks, dim=0)

    def __getitem__(self, index):
        case = self.cases[index]
        subject_id = case['subject_id']
        if self.cache and subject_id in self._cache:
            images, region_masks = self._cache[subject_id]
            images = images.clone()
            region_masks = region_masks.clone()
        else:
            images, region_masks = self._load_case(case)
            if self.cache:
                self._cache[subject_id] = (images.clone(), region_masks.clone())

        if self.augment:
            if torch.rand(()) > 0.5:
                images = torch.flip(images, dims=(-1,))
                region_masks = torch.flip(region_masks, dims=(-1,))
            if torch.rand(()) > 0.5:
                images = torch.flip(images, dims=(-2,))
                region_masks = torch.flip(region_masks, dims=(-2,))

        return {
            'images': images,
            'region_masks': region_masks,
            'label': torch.tensor(case['label'], dtype=torch.long),
            'subject_id': subject_id,
        }
