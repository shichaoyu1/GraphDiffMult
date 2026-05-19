import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


GRADE_COLORS = {2: "#4AA66A", 3: "#E59F2F", 4: "#BF2F35"}
SEG_COLORS = {"ED": "#30B7C7", "ET": "#F2A900", "NCR": "#8A5BD1"}


@dataclass
class DatasetSpec:
    dataset_type: str
    dataset_label: str
    grade_col: str
    meta_id_col: str
    modality_patterns: Dict[str, List[str]]
    seg_patterns: List[str]
    tumor_type_cols: List[str]
    idh_cols: List[str]
    codeletion_cols: List[str]
    mgmt_cols: List[str]
    age_cols: List[str]
    sex_cols: List[str]


def percentile_norm(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(foreground, [1, 99])
    return np.clip((volume - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def infer_grade(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def parse_numeric_id(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(\d+)", str(text))
    if not match:
        return None
    return int(match.group(1))


def detect_dataset_type(dataset_root: Path) -> str:
    name = dataset_root.name.lower()
    if "ucsf-pdgm" in name:
        return "ucsf"
    if "utsw-glioma" in name:
        return "utsw"
    sample_dirs = [p.name.lower() for p in dataset_root.iterdir() if p.is_dir()]
    if any("ucsf-pdgm" in d for d in sample_dirs):
        return "ucsf"
    return "utsw"


def build_spec(dataset_type: str) -> DatasetSpec:
    if dataset_type == "ucsf":
        return DatasetSpec(
            dataset_type="ucsf",
            dataset_label="UCSF-PDGM",
            grade_col="WHO CNS Grade",
            meta_id_col="ID",
            modality_patterns={
                "T1": ["*_T1.nii.gz", "*_T1_bias.nii.gz"],
                "T1ce": ["*_T1c.nii.gz", "*_T1c_bias.nii.gz", "*_T1gd.nii.gz"],
                "T2": ["*_T2.nii.gz", "*_T2_bias.nii.gz"],
                "FLAIR": ["*_FLAIR.nii.gz", "*_FLAIR_bias.nii.gz"],
            },
            seg_patterns=["*_tumor_segmentation.nii.gz", "*seg*.nii.gz"],
            tumor_type_cols=["Final pathologic diagnosis (WHO 2021)"],
            idh_cols=["IDH"],
            codeletion_cols=["1p/19q"],
            mgmt_cols=["MGMT status", "MGMT"],
            age_cols=["Age at MRI", "Age"],
            sex_cols=["Sex", "Sex at birth"],
        )

    return DatasetSpec(
        dataset_type="utsw",
        dataset_label="UTSW-Glioma",
        grade_col="Tumor Grade",
        meta_id_col="Subject ID",
        modality_patterns={
            "T1": ["brain_t1.nii.gz", "brain_t1_ants.nii.gz", "*_t1.nii.gz"],
            "T1ce": ["brain_t1ce.nii.gz", "brain_t1ce_ants.nii.gz", "*_t1ce.nii.gz", "*_t1gd.nii.gz"],
            "T2": ["brain_t2.nii.gz", "brain_t2_ants.nii.gz", "*_t2.nii.gz"],
            "FLAIR": ["brain_flair.nii.gz", "brain_fl_ants.nii.gz", "*_flair.nii.gz", "*_fl_*.nii.gz"],
        },
        seg_patterns=[
            "rtumorseg_manual_correction.nii.gz",
            "tumorseg_manual_correction.nii.gz",
            "tumorseg_FeTS.nii.gz",
            "*_seg.nii.gz",
            "*seg*.nii.gz",
        ],
        tumor_type_cols=["Tumor Type"],
        idh_cols=["IDH"],
        codeletion_cols=["1p19Q CODEL", "1p/19q"],
        mgmt_cols=["MGMT"],
        age_cols=["Age at Imaging"],
        sex_cols=["Sex at birth"],
    )


def resolve_dataset_root(dataset_root_arg: Optional[str], search_root: Path, dataset_type_arg: str) -> Path:
    if dataset_root_arg:
        path = Path(dataset_root_arg)
        if not path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {path}")
        return path

    if dataset_type_arg in ("auto", "utsw"):
        for path in search_root.rglob("UTSW-Glioma"):
            if path.is_dir():
                return path
    if dataset_type_arg in ("auto", "ucsf"):
        for path in search_root.rglob("UCSF-PDGM-v5"):
            if path.is_dir():
                return path
    raise FileNotFoundError("Cannot auto-resolve dataset root. Please pass --dataset-root explicitly.")


def find_metadata_file(dataset_root: Path, search_root: Path, spec: DatasetSpec) -> Path:
    if spec.dataset_type == "ucsf":
        candidates = [
            dataset_root.parent / "UCSF-PDGM-metadata_v5.csv",
            dataset_root / "UCSF-PDGM-metadata_v5.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        for path in search_root.rglob("UCSF-PDGM-metadata_v5.csv"):
            if path.is_file():
                return path
        raise FileNotFoundError("Cannot find UCSF-PDGM-metadata_v5.csv")

    candidates = [
        dataset_root / "UTSW_Glioma_Metadata-2-1.tsv",
        dataset_root.parent / "UTSW_Glioma_Metadata-2-1.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    for path in search_root.rglob("UTSW_Glioma_Metadata-2-1.tsv"):
        if path.is_file():
            return path
    raise FileNotFoundError("Cannot find UTSW_Glioma_Metadata-2-1.tsv")


def load_metadata(metadata_file: Path, spec: DatasetSpec) -> pd.DataFrame:
    if metadata_file.suffix.lower() == ".tsv":
        return pd.read_csv(metadata_file, sep="\t")
    return pd.read_csv(metadata_file)


def choose_first_file(patient_dir: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(patient_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def map_segmentation_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    values = set(np.unique(seg.astype(np.int32)).tolist())
    if 4 in values or 3 in values:
        return {"ED": seg == 2, "ET": np.isin(seg, [4, 3]), "NCR": seg == 1}
    if 300 in values or 200 in values or 100 in values:
        return {"ED": seg == 200, "ET": seg == 300, "NCR": seg == 100}
    tumor = seg > 0
    return {"ED": tumor, "ET": np.zeros_like(tumor), "NCR": np.zeros_like(tumor)}


def compute_case_metrics(seg_regions: Dict[str, np.ndarray], slice_idx: int) -> Dict[str, float]:
    ed_count = int(seg_regions["ED"].sum())
    et_count = int(seg_regions["ET"].sum())
    ncr_count = int(seg_regions["NCR"].sum())
    total = max(ed_count + et_count + ncr_count, 1)
    return {
        "ed_voxels": ed_count,
        "et_voxels": et_count,
        "ncr_voxels": ncr_count,
        "total_voxels": total,
        "ed_ratio": ed_count / total,
        "et_ratio": et_count / total,
        "ncr_ratio": ncr_count / total,
    }


def get_meta_value(meta: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in meta and not pd.isna(meta[key]):
            text = str(meta[key]).strip()
            if text:
                return text
    return "NA"


def canonical_case_id(dir_name: str, spec: DatasetSpec) -> str:
    if spec.dataset_type == "ucsf":
        match = re.search(r"(UCSF-PDGM-\d+)", dir_name)
        if match:
            return match.group(1)
        return dir_name.replace("_nifti", "")
    return dir_name


def build_metadata_lookup(metadata: pd.DataFrame, spec: DatasetSpec):
    if spec.dataset_type == "ucsf":
        lookup = {}
        for _, row in metadata.iterrows():
            key = parse_numeric_id(row.get(spec.meta_id_col, ""))
            if key is not None:
                lookup[key] = row
        return lookup
    return {str(row.get(spec.meta_id_col, "")).strip(): row for _, row in metadata.iterrows()}


def enumerate_cases(dataset_root: Path, metadata: pd.DataFrame, spec: DatasetSpec) -> List[Dict]:
    lookup = build_metadata_lookup(metadata, spec)
    cases = []
    for d in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        seg = choose_first_file(d, spec.seg_patterns)
        if seg is None:
            continue
        modality_paths = {k: choose_first_file(d, patterns) for k, patterns in spec.modality_patterns.items()}
        if any(v is None for v in modality_paths.values()):
            continue

        case_id = canonical_case_id(d.name, spec)
        if spec.dataset_type == "ucsf":
            meta_row = lookup.get(parse_numeric_id(case_id))
        else:
            meta_row = lookup.get(case_id)
        if meta_row is None:
            meta_row = pd.Series(dtype=object)

        grade = infer_grade(meta_row.get(spec.grade_col))
        cases.append(
            {
                "dir_name": d.name,
                "case_id": case_id,
                "patient_dir": d,
                "meta": meta_row,
                "grade": grade,
                "seg_path": seg,
                "modality_paths": modality_paths,
            }
        )
    return cases


def pick_demo_cases(cases: List[Dict], n_cases: int) -> List[Dict]:
    selected: List[Dict] = []
    for grade in (2, 3, 4):
        for case in cases:
            if case["grade"] == grade and case not in selected:
                selected.append(case)
                break
    for case in cases:
        if case in selected:
            continue
        selected.append(case)
        if len(selected) >= n_cases:
            break
    return selected[:n_cases]


def draw_patient_card(ax, case: Dict, spec: DatasetSpec, metrics: Dict[str, float], slice_idx: int):
    meta = case["meta"]
    grade = case["grade"]
    grade_label = f"Grade {grade}" if grade else "Grade NA"
    grade_color = GRADE_COLORS.get(grade, "#5D6778")

    ax.set_facecolor("#FFFFFF")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#D8DDE6")

    ax.text(0.05, 0.95, "Patient Card", fontsize=14, weight="bold", color="#1C2431", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.89, case["case_id"], fontsize=12, color="#5D6778", transform=ax.transAxes)

    ax.add_patch(plt.Rectangle((0.05, 0.73), 0.9, 0.13, color=grade_color, transform=ax.transAxes))
    ax.text(0.08, 0.81, "Tumor Grade", color="white", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.75, grade_label, color="white", fontsize=18, weight="bold", transform=ax.transAxes)

    fields = [
        ("Tumor Type", get_meta_value(meta, spec.tumor_type_cols)),
        ("IDH", get_meta_value(meta, spec.idh_cols)),
        ("1p/19q", get_meta_value(meta, spec.codeletion_cols)),
        ("MGMT", get_meta_value(meta, spec.mgmt_cols)),
        ("Age / Sex", f"{get_meta_value(meta, spec.age_cols)} / {get_meta_value(meta, spec.sex_cols)}"),
        ("Slice selected", f"max tumor area (z={slice_idx})"),
        ("Tumor burden", f"{metrics['total_voxels']:,} voxels"),
    ]
    y = 0.66
    for key, val in fields:
        ax.text(0.05, y, key, fontsize=10, color="#5D6778", transform=ax.transAxes)
        ax.text(0.5, y, val, fontsize=10.5, color="#1C2431", transform=ax.transAxes, ha="left")
        y -= 0.075

    ax.text(
        0.05,
        0.03,
        "WHO-like grade is for stratified display only, not final WHO diagnosis.",
        fontsize=8.5,
        color="#7B8494",
        transform=ax.transAxes,
    )


def draw_modalities_panel(fig, spec_slot, modalities: Dict[str, np.ndarray], slice_idx: int):
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec_slot, wspace=0.03, hspace=0.06)
    for i, title in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        ax.imshow(modalities[title][:, :, slice_idx].T, cmap="gray", origin="lower")
        ax.set_title(title, fontsize=11, color="#1C2431")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#D8DDE6")


def draw_overlay_panel(ax, flair: np.ndarray, seg_regions: Dict[str, np.ndarray], slice_idx: int, metrics: Dict[str, float]):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#D8DDE6")
    ax.set_title("Tumor Overlay (FLAIR + Segmentation)", fontsize=12, color="#1C2431", pad=8)
    ax.imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")

    for key, alpha in [("ED", 0.45), ("ET", 0.75), ("NCR", 0.75)]:
        mask = seg_regions[key][:, :, slice_idx].T
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        color = SEG_COLORS[key]
        rgb = tuple(int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        rgba[mask, 0], rgba[mask, 1], rgba[mask, 2], rgba[mask, 3] = rgb[0], rgb[1], rgb[2], alpha
        ax.imshow(rgba, origin="lower")

    lines = [
        f"ED  {metrics['ed_ratio'] * 100:5.1f}%  ({metrics['ed_voxels']:,})",
        f"ET  {metrics['et_ratio'] * 100:5.1f}%  ({metrics['et_voxels']:,})",
        f"NCR {metrics['ncr_ratio'] * 100:5.1f}%  ({metrics['ncr_voxels']:,})",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        fontsize=9.5,
        color="white",
        transform=ax.transAxes,
        bbox={"facecolor": "#1C2431", "alpha": 0.65, "pad": 6, "edgecolor": "none"},
    )


def draw_cohort_grade(ax, metadata: pd.DataFrame, grade_col: str, current_grade: Optional[int]):
    grades = metadata[grade_col].apply(infer_grade).dropna().astype(int)
    counts = grades.value_counts().reindex([2, 3, 4], fill_value=0)
    bars = ax.bar(["Grade 2", "Grade 3", "Grade 4"], counts.values, color=[GRADE_COLORS[2], GRADE_COLORS[3], GRADE_COLORS[4]], width=0.58)
    ax.set_title("Cohort Grade Distribution", fontsize=12, color="#1C2431")
    ax.set_ylabel("Cases")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}", ha="center", va="bottom", fontsize=10)
    if current_grade in (2, 3, 4):
        idx = [2, 3, 4].index(current_grade)
        bars[idx].set_edgecolor("#1C2431")
        bars[idx].set_linewidth(2.5)


def draw_scatter(ax, context_rows: List[Dict], current_id: str):
    ax.set_title("Tumor Burden vs Enhancement Ratio", fontsize=12, color="#1C2431")
    ax.set_xlabel("Total Tumor Burden (voxel count)")
    ax.set_ylabel("ET ratio")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    for row in context_rows:
        color = GRADE_COLORS.get(row["grade"], "#7B8494")
        if row["case_id"] == current_id:
            ax.scatter(row["total_voxels"], row["et_ratio"], s=180, facecolors="white", edgecolors=color, linewidths=2.8, zorder=4)
        else:
            ax.scatter(row["total_voxels"], row["et_ratio"], s=75, color=color, alpha=0.75, zorder=3)
    ax.set_ylim(-0.02, 1.02)


def build_single_dashboard(case: Dict, spec: DatasetSpec, metadata: pd.DataFrame, output_dir: Path, context_rows: List[Dict]) -> Path:
    modalities = {}
    for key, path in case["modality_paths"].items():
        modalities[key] = percentile_norm(nib.load(str(path)).get_fdata().astype(np.float32))

    seg = nib.load(str(case["seg_path"])).get_fdata().astype(np.float32)
    seg_regions = map_segmentation_regions(seg)
    slice_idx = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    metrics = compute_case_metrics(seg_regions, slice_idx)

    fig = plt.figure(figsize=(18, 10), facecolor="#F6F7F9")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 2.35, 1.4], height_ratios=[2.3, 1.0], wspace=0.12, hspace=0.18)
    draw_patient_card(fig.add_subplot(gs[0, 0]), case, spec, metrics, slice_idx)
    draw_modalities_panel(fig, gs[0, 1], modalities, slice_idx)
    draw_overlay_panel(fig.add_subplot(gs[0, 2]), modalities["FLAIR"], seg_regions, slice_idx, metrics)
    draw_cohort_grade(fig.add_subplot(gs[1, 0:2]), metadata, spec.grade_col, case["grade"])
    draw_scatter(fig.add_subplot(gs[1, 2]), context_rows, case["case_id"])

    fig.suptitle(f"{spec.dataset_label} WHO-like Atlas Baseline | {case['case_id']}", fontsize=18, color="#172033", weight="bold", y=0.98)
    fig.text(0.01, 0.01, "Note: WHO-like/tumor-grade is used as stratification context and not equivalent to final integrated WHO diagnosis.", fontsize=9, color="#5D6778")

    out_name = re.sub(r"[^A-Za-z0-9._-]", "_", case["case_id"]) + "_dashboard.png"
    output_path = output_dir / out_name
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def collect_context_metrics(cases: List[Dict]) -> List[Dict]:
    rows = []
    for case in cases:
        seg = nib.load(str(case["seg_path"])).get_fdata().astype(np.float32)
        seg_regions = map_segmentation_regions(seg)
        slice_idx = int(np.argmax((seg > 0).sum(axis=(0, 1))))
        metrics = compute_case_metrics(seg_regions, slice_idx)
        rows.append({"case_id": case["case_id"], "grade": case["grade"], "total_voxels": metrics["total_voxels"], "et_ratio": metrics["et_ratio"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build static WHO-like dashboard PNG examples for UTSW/UCSF datasets.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root folder. Example: .../UTSW-Glioma or .../UCSF-PDGM-v5")
    parser.add_argument("--dataset-search-root", type=str, default="D:/dataset")
    parser.add_argument("--dataset-type", type=str, choices=["auto", "utsw", "ucsf"], default="auto")
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="output/dashboard_examples")
    args = parser.parse_args()

    search_root = Path(args.dataset_search_root)
    dataset_root = resolve_dataset_root(args.dataset_root, search_root, args.dataset_type)
    detected_type = detect_dataset_type(dataset_root) if args.dataset_type == "auto" else args.dataset_type
    spec = build_spec(detected_type)
    metadata_file = find_metadata_file(dataset_root, search_root, spec)
    metadata = load_metadata(metadata_file, spec)
    if spec.grade_col not in metadata.columns:
        raise KeyError(f"Metadata missing grade column: {spec.grade_col}")

    cases = enumerate_cases(dataset_root, metadata, spec)
    if not cases:
        raise RuntimeError(f"No valid cases found under {dataset_root}")

    selected = pick_demo_cases(cases, max(1, int(args.n_cases)))
    context_cases = selected[:]
    if len(context_cases) < 12:
        for case in cases:
            if case in context_cases:
                continue
            context_cases.append(case)
            if len(context_cases) >= 12:
                break

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_rows = collect_context_metrics(context_cases)
    generated = [build_single_dashboard(case, spec, metadata, output_dir, context_rows) for case in selected]

    print(f"Dataset root : {dataset_root}")
    print(f"Dataset type : {spec.dataset_type}")
    print(f"Metadata file: {metadata_file}")
    print("Selected IDs : ", [c["case_id"] for c in selected])
    print("Generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

