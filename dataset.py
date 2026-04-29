"""
dataset.py - BraTS / UTSW-Glioma patient loading and patch sampling.

Usage:
    ds = BraTSPatchDataset('/path/to/BraTS2021_00060', patch_size=32, n_patches=200)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    imgs, label = next(iter(loader))

    # imgs: list of 3 tensors [B, 1, 32, 32] (T1, FLAIR, T1ce)
    # label: [B] long tensor (0=background, 1=edema/necrotic, 2=enhancing tumor)
"""

import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MODALITY_CANDIDATES = {
    't1': [
        'brain_t1.nii.gz',
        'brain_t1_ants.nii.gz',
        '*_t1.nii.gz',
        '*-t1.nii.gz',
    ],
    'flair': [
        'brain_flair.nii.gz',
        'brain_fl_ants.nii.gz',
        '*_flair.nii.gz',
        '*-flair.nii.gz',
        '*_fl_*.nii.gz',
    ],
    't1ce': [
        'brain_t1ce.nii.gz',
        'brain_t1ce_ants.nii.gz',
        '*_t1ce.nii.gz',
        '*-t1ce.nii.gz',
        '*_t1gd.nii.gz',
        '*-t1gd.nii.gz',
    ],
    't2': [
        'brain_t2.nii.gz',
        'brain_t2_ants.nii.gz',
        '*_t2.nii.gz',
        '*-t2.nii.gz',
    ],
}

SEGMENTATION_CANDIDATES = [
    'rtumorseg_manual_correction.nii.gz',
    'tumorseg_manual_correction.nii.gz',
    'tumorseg_FeTS.nii.gz',
    '*_seg.nii.gz',
    '*-seg.nii.gz',
    '*seg*.nii.gz',
]

UTSW_METADATA_FILENAME = 'UTSW_Glioma_Metadata-2-1.tsv'


def percentile_norm(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Normalize non-zero foreground intensities to [0, 1]."""
    fg = vol[vol > 0]
    if len(fg) == 0:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, [p_lo, p_hi])
    out = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
    return out.astype(np.float32)


def _glob_candidates(folder: str, patterns: list) -> list:
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(hits))


def _prefer_shape_match(paths: list, shape: tuple = None) -> str:
    if not paths:
        return None
    if shape is None:
        return paths[0]

    try:
        import nibabel as nib
    except ImportError:
        return paths[0]

    for path in paths:
        if tuple(nib.load(path).shape) == tuple(shape):
            return path
    return paths[0]


def find_modality_file(folder: str, modality: str, prefer_registered: bool = False) -> str:
    """Find one MRI modality file in BraTS or UTSW-Glioma naming layouts."""
    key = modality.lower()
    patterns = MODALITY_CANDIDATES.get(key, [f'*{modality}*.nii.gz'])
    hits = _glob_candidates(folder, patterns)

    if prefer_registered:
        registered = [path for path in hits if '_ants' in os.path.basename(path).lower()]
        if registered:
            hits = registered
    else:
        native = [path for path in hits if '_ants' not in os.path.basename(path).lower()]
        if native:
            hits = native

    if not hits:
        raise FileNotFoundError(f"Cannot find modality '{modality}' under folder={folder}")
    return hits[0]


def find_segmentation_file(folder: str, image_shape: tuple = None) -> str:
    """Find a segmentation file, preferring masks aligned to image_shape."""
    hits = _glob_candidates(folder, SEGMENTATION_CANDIDATES)
    if not hits:
        raise FileNotFoundError(f"Cannot find a segmentation file under folder={folder}")
    return _prefer_shape_match(hits, image_shape)


def resolve_patient_dir(path: str, patient_id: str = None) -> str:
    """Accept either a patient folder or a dataset root plus optional patient_id."""
    if patient_id:
        candidate = os.path.join(path, patient_id)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Cannot find patient_id={patient_id} under {path}")
        return candidate

    if glob.glob(os.path.join(path, '*.nii.gz')):
        return path

    case_dirs = [
        entry.path for entry in os.scandir(path)
        if entry.is_dir() and glob.glob(os.path.join(entry.path, '*.nii.gz'))
    ]
    if not case_dirs:
        raise FileNotFoundError(f"Cannot find NIfTI patient folders under {path}")

    case_dirs.sort()
    print(f"  No patient_id provided; using first patient folder: {os.path.basename(case_dirs[0])}")
    return case_dirs[0]


def find_utsw_metadata(patient_dir: str, metadata_tsv: str = None) -> str:
    if metadata_tsv and os.path.exists(metadata_tsv):
        return metadata_tsv

    current = os.path.abspath(patient_dir)
    for _ in range(4):
        parent = os.path.dirname(current)
        for folder in (current, parent):
            candidate = os.path.join(folder, UTSW_METADATA_FILENAME)
            if os.path.exists(candidate):
                return candidate
        if parent == current:
            break
        current = parent
    return None


def load_utsw_metadata(metadata_tsv: str) -> dict:
    records = {}
    with open(metadata_tsv, 'r', encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file, delimiter='\t'):
            subject_id = row.get('Subject ID')
            if subject_id:
                records[subject_id] = row
    return records


def get_utsw_patient_info(patient_dir: str, metadata_tsv: str = None) -> dict:
    metadata_path = find_utsw_metadata(patient_dir, metadata_tsv)
    if not metadata_path:
        return {}

    subject_id = os.path.basename(os.path.normpath(patient_dir))
    return load_utsw_metadata(metadata_path).get(subject_id, {})


def get_utsw_cases(root_dir: str, metadata_tsv: str = None, require_seg: bool = True) -> list:
    metadata_path = find_utsw_metadata(root_dir, metadata_tsv)
    metadata = load_utsw_metadata(metadata_path) if metadata_path else {}
    cases = []

    for entry in sorted(os.scandir(root_dir), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        if not glob.glob(os.path.join(entry.path, '*.nii.gz')):
            continue
        if require_seg and not _glob_candidates(entry.path, SEGMENTATION_CANDIDATES):
            continue

        info = metadata.get(entry.name, {})
        cases.append({
            'subject_id': entry.name,
            'patient_dir': entry.path,
            'tumor_grade': info.get('Tumor Grade'),
            'tumor_type': info.get('Tumor Type'),
            'metadata': info,
        })
    return cases


def load_brats_patient(folder: str, prefer_registered: bool = False, metadata_tsv: str = None):
    """
    Load one BraTS or UTSW-Glioma patient folder.

    Returns:
        vols: {'t1': ndarray, 'flair': ndarray, 't1ce': ndarray, 't2': ndarray}
        seg: ndarray, expected labels 0/1/2/4
        shape: tuple (H, W, D)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError('Please install nibabel: pip install nibabel')

    vols = {}
    for key in ['t1', 'flair', 't1ce', 't2']:
        path = find_modality_file(folder, key, prefer_registered=prefer_registered)
        data = nib.load(path).get_fdata().astype(np.float32)
        vols[key] = percentile_norm(data)

    image_shape = next(iter(vols.values())).shape
    seg_path = find_segmentation_file(folder, image_shape=image_shape)
    seg = nib.load(seg_path).get_fdata().astype(np.float32)

    if seg.shape != image_shape:
        raise ValueError(f'Segmentation shape {seg.shape} does not match image shape {image_shape}')

    patient_info = get_utsw_patient_info(folder, metadata_tsv)
    if patient_info:
        print(
            '  UTSW metadata: '
            f"type={patient_info.get('Tumor Type', 'NA')}, "
            f"WHO grade={patient_info.get('Tumor Grade', 'NA')}"
        )

    print(f"  Volume shape: {seg.shape}")
    print(f"  Non-zero tumor voxels: {int((seg > 0).sum())}")
    return vols, seg, seg.shape


def seg_to_label(seg_patch: np.ndarray) -> int:
    """
    Map a segmentation patch to a 3-class patch label:
      0 = background-dominant
      1 = edema / necrotic core (labels 1 or 2)
      2 = enhancing tumor (label 4)
    """
    counts = {
        0: float((seg_patch == 0).sum()),
        1: float((seg_patch == 1).sum()),
        2: float((seg_patch == 2).sum()),
        4: float((seg_patch == 4).sum()),
    }
    dominant = max(counts, key=counts.get)
    mapping = {0: 0, 1: 1, 2: 1, 4: 2}
    return mapping[dominant]


class BraTSPatchDataset(Dataset):
    """
    Sample tumor-region axial patches from one BraTS or UTSW-Glioma patient.
    """

    def __init__(
        self,
        patient_dir: str,
        patch_size: int = 32,
        n_patches: int = 200,
        min_tumor: int = 100,
        modalities: list = None,
        prefer_registered: bool = False,
        metadata_tsv: str = None,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.modalities = modalities or ['t1', 'flair', 't1ce']

        print(f"Loading patient data: {patient_dir}")
        vols, seg, (height, width, depth) = load_brats_patient(
            patient_dir,
            prefer_registered=prefer_registered,
            metadata_tsv=metadata_tsv,
        )
        self.vols = vols
        self.seg = seg
        self.patient_info = get_utsw_patient_info(patient_dir, metadata_tsv)

        tumor_per_z = (seg > 0).sum(axis=(0, 1))
        valid_zs = np.where(tumor_per_z >= min_tumor)[0]
        if len(valid_zs) == 0:
            valid_zs = np.where(tumor_per_z > 0)[0]
        if len(valid_zs) == 0:
            raise ValueError(f'No tumor voxels found in segmentation for {patient_dir}')
        print(f"  Valid slices: {len(valid_zs)}")

        rng = np.random.default_rng(seed)
        half = patch_size // 2
        self.samples = []

        attempt = 0
        while len(self.samples) < n_patches and attempt < n_patches * 10:
            attempt += 1
            z = int(rng.choice(valid_zs))
            ys, xs = np.where((seg[:, :, z] > 0))
            if len(ys) < 10:
                continue
            idx = rng.integers(0, len(ys))
            cy = int(np.clip(ys[idx], half, height - half))
            cx = int(np.clip(xs[idx], half, width - half))
            self.samples.append((cy, cx, z))

        print(f"  Sampled patches: {len(self.samples)}")
        label_counts = {}
        for cy, cx, z in self.samples:
            seg_p = self.seg[cy-half:cy+half, cx-half:cx+half, z]
            label = seg_to_label(seg_p)
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Label distribution: {label_counts} (0=BG, 1=edema/necrotic, 2=enhancing)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cy, cx, z = self.samples[index]
        half = self.patch_size // 2

        patches = []
        for mod in self.modalities:
            sl = self.vols[mod][cy-half:cy+half, cx-half:cx+half, z]
            if np.random.rand() > 0.5:
                sl = sl[::-1, :].copy()
            if np.random.rand() > 0.5:
                sl = sl[:, ::-1].copy()
            patches.append(torch.tensor(sl, dtype=torch.float32).unsqueeze(0))

        seg_p = self.seg[cy-half:cy+half, cx-half:cx+half, z]
        label = torch.tensor(seg_to_label(seg_p), dtype=torch.long)
        return patches, label


def collate_fn(batch):
    """Collate list[(patches, label)] into (list[tensor], labels)."""
    patches_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    n_mod = len(patches_list[0])
    imgs = [torch.stack([p[m] for p in patches_list]) for m in range(n_mod)]
    return imgs, labels


def get_dataloader(patient_dir: str, batch_size: int = 16, **kwargs) -> DataLoader:
    """Build a DataLoader for one patient folder."""
    ds = BraTSPatchDataset(patient_dir, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    ), ds
