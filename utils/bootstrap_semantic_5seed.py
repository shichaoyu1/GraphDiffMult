import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC_KEYS = [
    "map",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "pair_auc",
    "edge_precision@10",
    "edge_precision@25",
    "edge_precision@50",
    "anchor_consistency",
    "positive_negative_distance_gap",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[pos].sum()
    return float((pos_ranks - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def retrieval_metrics(query_vectors, target_ids, prototypes, gallery_ids=None, ks=(1, 5, 10), subject_ids=None):
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if gallery_ids is None:
        gallery_ids = list(range(len(prototypes)))
    gallery_ids = list(gallery_ids)
    if len(query_vectors) == 0 or len(gallery_ids) == 0:
        return {}

    query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ proto_norm[np.asarray(gallery_ids)].T

    recalls = {k: [] for k in ks}
    reciprocal_ranks = []
    average_precisions = []
    pos_scores = []
    neg_scores = []
    pos_dists = []
    neg_dists = []
    edge_labels = []
    edge_scores = []

    for row, positives in enumerate(target_ids):
        positives = set(positives).intersection(gallery_ids)
        if not positives:
            continue
        ranking = np.argsort(-scores[row])
        ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
        for k in ks:
            top_k = ranked_anchor_ids[: min(k, len(ranked_anchor_ids))]
            recalls[k].append(float(any(anchor_id in positives for anchor_id in top_k)))
        first_rank = next(
            (rank + 1 for rank, anchor_id in enumerate(ranked_anchor_ids) if anchor_id in positives),
            None,
        )
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        precision_hits = []
        hit_count = 0
        for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
            if anchor_id in positives:
                hit_count += 1
                precision_hits.append(hit_count / rank_idx)
        if precision_hits:
            average_precisions.append(float(np.mean(precision_hits)))

        for col, anchor_id in enumerate(gallery_ids):
            score = float(scores[row, col])
            distance = float(np.linalg.norm(query_norm[row] - proto_norm[anchor_id]))
            is_positive = anchor_id in positives
            edge_scores.append(score)
            edge_labels.append(1 if is_positive else 0)
            if is_positive:
                pos_scores.append(score)
                pos_dists.append(distance)
            else:
                neg_scores.append(score)
                neg_dists.append(distance)

    metrics = {f"recall@{k}": float(np.mean(values)) if values else float("nan") for k, values in recalls.items()}
    map_query = float(np.mean(average_precisions)) if average_precisions else float("nan")
    metrics["map_query"] = map_query
    if subject_ids is not None and len(subject_ids) == len(target_ids):
        patient_ap = defaultdict(list)
        for row, positives in enumerate(target_ids):
            positives = set(positives).intersection(gallery_ids)
            if not positives:
                continue
            ranking = np.argsort(-scores[row])
            ranked_anchor_ids = [gallery_ids[idx] for idx in ranking]
            precision_hits = []
            hit_count = 0
            for rank_idx, anchor_id in enumerate(ranked_anchor_ids, start=1):
                if anchor_id in positives:
                    hit_count += 1
                    precision_hits.append(hit_count / rank_idx)
            if precision_hits:
                patient_ap[str(subject_ids[row])].append(float(np.mean(precision_hits)))
        patient_means = [float(np.mean(values)) for values in patient_ap.values() if values]
        metrics["map"] = float(np.mean(patient_means)) if patient_means else map_query
    else:
        metrics["map"] = map_query
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    metrics["pair_auc"] = binary_auc(edge_labels, edge_scores)
    metrics["anchor_consistency"] = (
        float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else float("nan")
    )
    metrics["positive_negative_distance_gap"] = (
        float(np.mean(neg_dists) - np.mean(pos_dists)) if pos_dists and neg_dists else float("nan")
    )
    if edge_scores:
        order = np.argsort(-np.asarray(edge_scores))
        for k in (10, 25, 50):
            top = order[: min(k, len(order))]
            metrics[f"edge_precision@{k}"] = (
                float(np.mean(np.asarray(edge_labels)[top])) if len(top) else float("nan")
            )
    return metrics


def sample_patient_indices(patient_to_indices, rng):
    patients = list(patient_to_indices.keys())
    sampled = rng.choice(patients, size=len(patients), replace=True)
    subset = []
    for patient in sampled:
        subset.extend(patient_to_indices[patient])
    return subset


def load_run(run_dir):
    config = load_json(run_dir / "config.json")
    records = load_json(run_dir / "patient_level_records.json")
    query_vectors = np.asarray(records["query_vectors"], dtype=np.float32)
    prototypes = np.asarray(records["prototypes"], dtype=np.float32)
    query_targets = [list(map(int, ids)) for ids in records["query_targets"]]
    subject_ids = [str(x) for x in records["subject_ids"]]
    patient_to_indices = defaultdict(list)
    for idx, sid in enumerate(subject_ids):
        patient_to_indices[sid].append(idx)
    return {
        "run_dir": str(run_dir),
        "variant": str(config.get("variant", "")),
        "seed": int(config.get("seed", -1)),
        "query_vectors": query_vectors,
        "prototypes": prototypes,
        "query_targets": query_targets,
        "subject_ids": subject_ids,
        "patient_to_indices": dict(patient_to_indices),
    }


def run_metrics_on_indices(run_data, indices):
    if not indices:
        return {key: float("nan") for key in METRIC_KEYS}
    q = run_data["query_vectors"][indices]
    t = [run_data["query_targets"][idx] for idx in indices]
    subject_ids = [run_data["subject_ids"][idx] for idx in indices]
    metrics = retrieval_metrics(q, t, run_data["prototypes"], subject_ids=subject_ids)
    return {key: float(metrics.get(key, float("nan"))) for key in METRIC_KEYS}


def bootstrap_variant(runs, n_bootstrap=2000, seed=2026):
    rng = np.random.default_rng(seed)
    point_rows = []
    for run_data in runs:
        all_indices = []
        for idxs in run_data["patient_to_indices"].values():
            all_indices.extend(idxs)
        point_rows.append(run_metrics_on_indices(run_data, all_indices))

    point = {}
    for key in METRIC_KEYS:
        values = [row.get(key, float("nan")) for row in point_rows]
        values = [float(v) for v in values if not math.isnan(float(v))]
        point[key] = float(np.mean(values)) if values else float("nan")

    samples = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        iter_values = defaultdict(list)
        for run_data in sampled_runs:
            indices = sample_patient_indices(run_data["patient_to_indices"], rng)
            row = run_metrics_on_indices(run_data, indices)
            for key in METRIC_KEYS:
                value = row.get(key, float("nan"))
                if not math.isnan(float(value)):
                    iter_values[key].append(float(value))
        for key in METRIC_KEYS:
            if iter_values[key]:
                samples[key].append(float(np.mean(iter_values[key])))

    ci = {}
    for key in METRIC_KEYS:
        values = np.asarray(samples[key], dtype=np.float64)
        if values.size == 0:
            ci[key] = [float("nan"), float("nan")]
        else:
            ci[key] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return point, ci


def discover_runs(output_root, run_id=None):
    runs = []
    for record_file in output_root.rglob("patient_level_records.json"):
        run_dir = record_file.parent
        if run_id and not run_dir.name.endswith(run_id):
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        runs.append(load_run(run_dir))
    return runs


def main():
    parser = argparse.ArgumentParser(description="5-seed patient-level bootstrap for semantic alignment.")
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_csv", type=str, default="results/semantic_bootstrap_5seed.csv")
    parser.add_argument("--out_json", type=str, default="results/semantic_bootstrap_5seed.json")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    runs = discover_runs(output_root, run_id=args.run_id)
    if not runs:
        raise SystemExit("No runs with patient_level_records.json found.")

    by_variant = defaultdict(list)
    for run_data in runs:
        by_variant[run_data["variant"]].append(run_data)

    summary_json = {}
    rows = []
    for variant, variant_runs in sorted(by_variant.items()):
        variant_runs = sorted(variant_runs, key=lambda item: item["seed"])
        point, ci = bootstrap_variant(variant_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)
        seeds = sorted({run_data["seed"] for run_data in variant_runs})
        summary_json[variant] = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "point": point,
            "ci95": ci,
        }
        for metric in METRIC_KEYS:
            rows.append(
                {
                    "variant": variant,
                    "n_seeds": len(seeds),
                    "seeds": " ".join(str(x) for x in seeds),
                    "metric": metric,
                    "point": point.get(metric, float("nan")),
                    "ci95_low": ci.get(metric, [float("nan"), float("nan")])[0],
                    "ci95_high": ci.get(metric, [float("nan"), float("nan")])[1],
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["variant", "n_seeds", "seeds", "metric", "point", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2, ensure_ascii=False)

    print(f"Saved bootstrap CSV: {out_csv}")
    print(f"Saved bootstrap JSON: {out_json}")
    for variant, payload in sorted(summary_json.items()):
        print(f"{variant}: seeds={payload['seeds']} n={payload['n_seeds']}")


if __name__ == "__main__":
    main()
