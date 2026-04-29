import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


GRADE_COLORS = {
    2: "#4AA66A",
    3: "#E59F2F",
    4: "#BF2F35",
}
SEG_COLORS = {
    "ED": "#30B7C7",
    "ET": "#F2A900",
    "NCR": "#8A5BD1",
}
SEG_CANDIDATES = [
    "rtumorseg_manual_correction.nii.gz",
    "tumorseg_manual_correction.nii.gz",
    "tumorseg_FeTS.nii.gz",
]
MODALITIES = [
    ("T1", "brain_t1.nii.gz"),
    ("T1ce", "brain_t1ce.nii.gz"),
    ("T2", "brain_t2.nii.gz"),
    ("FLAIR", "brain_flair.nii.gz"),
]


def percentile_norm(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(foreground, [1, 99])
    return np.clip((volume - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def find_dataset_root(search_root: Path) -> Path:
    for path in search_root.rglob("UTSW-Glioma"):
        if path.is_dir():
            return path
    raise FileNotFoundError("Cannot find dataset folder named UTSW-Glioma under D:/dataset")


def find_metadata_file(search_root: Path) -> Path:
    for path in search_root.rglob("UTSW_Glioma_Metadata-2-1.tsv"):
        if path.is_file():
            return path
    raise FileNotFoundError("Cannot find UTSW_Glioma_Metadata-2-1.tsv under D:/dataset")


def choose_seg_path(patient_dir: Path) -> Optional[Path]:
    for name in SEG_CANDIDATES:
        candidate = patient_dir / name
        if candidate.exists():
            return candidate
    hits = sorted(patient_dir.glob("*seg*.nii.gz"))
    return hits[0] if hits else None


def choose_modality_path(patient_dir: Path, base_name: str) -> Optional[Path]:
    primary = patient_dir / base_name
    if primary.exists():
        return primary
    ants_name = base_name.replace(".nii.gz", "_ants.nii.gz")
    secondary = patient_dir / ants_name
    if secondary.exists():
        return secondary
    prefix = base_name.replace(".nii.gz", "")
    hits = sorted(patient_dir.glob(f"{prefix}*.nii.gz"))
    return hits[0] if hits else None


def map_segmentation_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    values = set(np.unique(seg.astype(np.int32)).tolist())

    # UTSW mostly follows BraTS labels: 1=NCR/NET, 2=ED, 4=ET.
    # Some cases include value 3; we treat it as ET-like foreground.
    if 4 in values or 3 in values:
        et = np.isin(seg, [4, 3])
        ncr = seg == 1
        ed = seg == 2
        return {"ED": ed, "ET": et, "NCR": ncr}

    # Fallback for remapped labels sometimes seen in exported masks.
    if 300 in values or 200 in values or 100 in values:
        return {"ED": seg == 200, "ET": seg == 300, "NCR": seg == 100}

    tumor = seg > 0
    return {"ED": tumor, "ET": np.zeros_like(tumor), "NCR": np.zeros_like(tumor)}


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


def pick_demo_cases(metadata: pd.DataFrame, dataset_root: Path, n_cases: int) -> List[str]:
    available = {p.name for p in dataset_root.iterdir() if p.is_dir()}
    rows = metadata[metadata["Subject ID"].isin(available)].copy()
    rows["grade_int"] = rows["Tumor Grade"].apply(infer_grade)
    rows = rows.dropna(subset=["grade_int"])

    selected: List[str] = []
    for grade in (2, 3, 4):
        bucket = rows[rows["grade_int"] == grade]
        for _, row in bucket.iterrows():
            patient_dir = dataset_root / row["Subject ID"]
            if choose_seg_path(patient_dir) is not None:
                selected.append(row["Subject ID"])
                break

    for _, row in rows.iterrows():
        pid = row["Subject ID"]
        if pid in selected:
            continue
        if choose_seg_path(dataset_root / pid) is None:
            continue
        selected.append(pid)
        if len(selected) >= n_cases:
            break

    return selected[:n_cases]


def compute_case_metrics(seg_regions: Dict[str, np.ndarray], slice_idx: int) -> Dict[str, float]:
    ed_count = int(seg_regions["ED"].sum())
    et_count = int(seg_regions["ET"].sum())
    ncr_count = int(seg_regions["NCR"].sum())
    total = max(ed_count + et_count + ncr_count, 1)

    ed_ratio = ed_count / total
    et_ratio = et_count / total
    ncr_ratio = ncr_count / total
    slice_area = int((seg_regions["ED"][:, :, slice_idx] | seg_regions["ET"][:, :, slice_idx] | seg_regions["NCR"][:, :, slice_idx]).sum())
    return {
        "ed_voxels": ed_count,
        "et_voxels": et_count,
        "ncr_voxels": ncr_count,
        "total_voxels": total,
        "ed_ratio": ed_ratio,
        "et_ratio": et_ratio,
        "ncr_ratio": ncr_ratio,
        "slice_area": slice_area,
    }


def draw_patient_card(ax, patient_id: str, meta: pd.Series, grade: Optional[int], metrics: Dict[str, float], slice_idx: int):
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#D8DDE6")

    grade_label = f"Grade {grade}" if grade else "Grade NA"
    grade_color = GRADE_COLORS.get(grade, "#5D6778")

    ax.text(0.05, 0.95, "Patient Card", fontsize=14, weight="bold", color="#1C2431", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.89, patient_id, fontsize=12, color="#5D6778", transform=ax.transAxes)

    ax.add_patch(plt.Rectangle((0.05, 0.73), 0.9, 0.13, color=grade_color, transform=ax.transAxes))
    ax.text(0.08, 0.81, "Tumor Grade", color="white", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.75, grade_label, color="white", fontsize=18, weight="bold", transform=ax.transAxes)

    fields = [
        ("Tumor Type", meta.get("Tumor Type", "NA")),
        ("IDH", meta.get("IDH", "NA")),
        ("1p/19q", meta.get("1p19Q CODEL", "NA")),
        ("MGMT", meta.get("MGMT", "NA")),
        ("Age / Sex", f"{meta.get('Age at Imaging', 'NA')} / {meta.get('Sex at birth', 'NA')}"),
        ("Slice selected", f"max tumor area (z={slice_idx})"),
        ("Tumor burden", f"{metrics['total_voxels']:,} voxels"),
    ]

    y = 0.66
    for key, value in fields:
        val = "NA" if pd.isna(value) else str(value)
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


def draw_modalities_panel(fig, spec, modalities: Dict[str, np.ndarray], slice_idx: int):
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.03, hspace=0.06)
    titles = ["T1", "T1ce", "T2", "FLAIR"]
    for i, title in enumerate(titles):
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

    base = flair[:, :, slice_idx].T
    ax.imshow(base, cmap="gray", origin="lower")

    for key, alpha in [("ED", 0.45), ("ET", 0.75), ("NCR", 0.75)]:
        mask = seg_regions[key][:, :, slice_idx].T
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        color = SEG_COLORS[key]
        rgb = tuple(int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha
        ax.imshow(rgba, origin="lower")

    legend_lines = [
        f"ED  {metrics['ed_ratio'] * 100:5.1f}%  ({metrics['ed_voxels']:,})",
        f"ET  {metrics['et_ratio'] * 100:5.1f}%  ({metrics['et_voxels']:,})",
        f"NCR {metrics['ncr_ratio'] * 100:5.1f}%  ({metrics['ncr_voxels']:,})",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(legend_lines),
        fontsize=9.5,
        color="white",
        transform=ax.transAxes,
        bbox={"facecolor": "#1C2431", "alpha": 0.65, "pad": 6, "edgecolor": "none"},
    )


def draw_cohort_grade(ax, metadata: pd.DataFrame, current_grade: Optional[int]):
    grade_values = metadata["Tumor Grade"].apply(infer_grade).dropna().astype(int)
    counts = grade_values.value_counts().reindex([2, 3, 4], fill_value=0)
    colors = [GRADE_COLORS[g] for g in [2, 3, 4]]
    bars = ax.bar(["Grade 2", "Grade 3", "Grade 4"], counts.values, color=colors, width=0.58)
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


def draw_scatter(ax, context_rows: List[Dict[str, float]], current_id: str):
    ax.set_title("Tumor Burden vs Enhancement Ratio", fontsize=12, color="#1C2431")
    ax.set_xlabel("Total Tumor Burden (voxel count)")
    ax.set_ylabel("ET ratio")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    for row in context_rows:
        grade = row["grade"]
        color = GRADE_COLORS.get(grade, "#7B8494")
        if row["subject_id"] == current_id:
            ax.scatter(
                row["total_voxels"],
                row["et_ratio"],
                s=180,
                facecolors="white",
                edgecolors=color,
                linewidths=2.8,
                zorder=4,
                label="Current case",
            )
        else:
            ax.scatter(row["total_voxels"], row["et_ratio"], s=75, color=color, alpha=0.75, zorder=3)

    ax.set_ylim(-0.02, 1.02)


def build_single_dashboard(
    patient_id: str,
    dataset_root: Path,
    metadata: pd.DataFrame,
    output_dir: Path,
    context_rows: List[Dict[str, float]],
):
    patient_dir = dataset_root / patient_id
    meta_row = metadata[metadata["Subject ID"] == patient_id].iloc[0]
    grade = infer_grade(meta_row.get("Tumor Grade"))

    modalities: Dict[str, np.ndarray] = {}
    for key, file_name in MODALITIES:
        path = choose_modality_path(patient_dir, file_name)
        if path is None:
            raise FileNotFoundError(f"Missing modality {file_name} for patient {patient_id}")
        modalities[key] = percentile_norm(nib.load(str(path)).get_fdata().astype(np.float32))

    seg_path = choose_seg_path(patient_dir)
    if seg_path is None:
        raise FileNotFoundError(f"Missing segmentation for patient {patient_id}")
    seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)
    seg_regions = map_segmentation_regions(seg)
    tumor_mask = seg > 0
    tumor_per_z = tumor_mask.sum(axis=(0, 1))
    slice_idx = int(np.argmax(tumor_per_z))
    metrics = compute_case_metrics(seg_regions, slice_idx)

    fig = plt.figure(figsize=(18, 10), facecolor="#F6F7F9")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 2.35, 1.4], height_ratios=[2.3, 1.0], wspace=0.12, hspace=0.18)

    ax_card = fig.add_subplot(gs[0, 0])
    draw_patient_card(ax_card, patient_id, meta_row, grade, metrics, slice_idx)

    draw_modalities_panel(fig, gs[0, 1], modalities, slice_idx)

    ax_overlay = fig.add_subplot(gs[0, 2])
    draw_overlay_panel(ax_overlay, modalities["FLAIR"], seg_regions, slice_idx, metrics)

    ax_grade = fig.add_subplot(gs[1, 0:2])
    draw_cohort_grade(ax_grade, metadata, grade)

    ax_scatter = fig.add_subplot(gs[1, 2])
    draw_scatter(ax_scatter, context_rows, patient_id)

    fig.suptitle(
        f"UTSW-Glioma WHO-like Atlas Baseline | {patient_id}",
        fontsize=18,
        color="#172033",
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        "Note: WHO-like/tumor-grade is used as stratification context and not equivalent to final integrated WHO diagnosis.",
        fontsize=9,
        color="#5D6778",
    )

    output_path = output_dir / f"{patient_id}_dashboard.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def collect_context_metrics(patient_ids: List[str], dataset_root: Path, metadata: pd.DataFrame) -> List[Dict[str, float]]:
    context_rows: List[Dict[str, float]] = []
    for pid in patient_ids:
        patient_dir = dataset_root / pid
        seg_path = choose_seg_path(patient_dir)
        if seg_path is None:
            continue
        seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_regions = map_segmentation_regions(seg)
        tumor_per_z = (seg > 0).sum(axis=(0, 1))
        slice_idx = int(np.argmax(tumor_per_z))
        metrics = compute_case_metrics(seg_regions, slice_idx)
        row = metadata[metadata["Subject ID"] == pid]
        grade = infer_grade(row.iloc[0]["Tumor Grade"]) if not row.empty else None
        context_rows.append(
            {
                "subject_id": pid,
                "grade": grade,
                "total_voxels": metrics["total_voxels"],
                "et_ratio": metrics["et_ratio"],
            }
        )
    return context_rows


def main():
    parser = argparse.ArgumentParser(description="Build static UTSW-Glioma WHO-like dashboard PNG examples.")
    parser.add_argument("--dataset-search-root", type=str, default="D:/dataset")
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="output/utsw_dashboard_examples")
    args = parser.parse_args()

    search_root = Path(args.dataset_search_root)
    dataset_root = find_dataset_root(search_root)
    metadata_file = find_metadata_file(search_root)
    metadata = pd.read_csv(metadata_file, sep="\t")

    n_cases = max(1, int(args.n_cases))
    selected_ids = pick_demo_cases(metadata, dataset_root, n_cases)
    if not selected_ids:
        raise RuntimeError("No valid demo cases found.")

    context_ids = selected_ids[:]
    if len(context_ids) < 9:
        available = [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()]
        for pid in available:
            if pid in context_ids:
                continue
            if choose_seg_path(dataset_root / pid) is None:
                continue
            context_ids.append(pid)
            if len(context_ids) >= 12:
                break

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_rows = collect_context_metrics(context_ids, dataset_root, metadata)
    generated = []
    for pid in selected_ids:
        out_path = build_single_dashboard(pid, dataset_root, metadata, output_dir, context_rows)
        generated.append(out_path)

    print(f"Dataset root : {dataset_root}")
    print(f"Metadata file: {metadata_file}")
    print(f"Selected IDs : {selected_ids}")
    print("Generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
