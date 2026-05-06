import argparse
import csv
import json
from pathlib import Path


SEMANTIC_KEYS = [
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

CLASSIFICATION_KEYS = [
    "auc",
    "accuracy",
    "f1_macro",
    "balanced_accuracy",
    "sensitivity_macro",
    "specificity_macro",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def flatten_nested(prefix, payload, keys):
    row = {}
    nested = payload.get(prefix)
    if isinstance(nested, dict):
        for key in keys:
            row[f"{prefix}_{key}"] = nested.get(key)
    return row


def best_from_history(history_path):
    if not history_path.exists():
        return {}
    history = load_json(history_path)
    best = {}
    for record in history:
        val = record.get("val", {})
        for key, value in val.items():
            if not isinstance(value, (int, float)):
                continue
            out_key = f"best_val_{key}"
            if out_key not in best or value > best[out_key]:
                best[out_key] = value
    return best


def run_fields(run_dir):
    name = run_dir.name
    fields = {
        "run": name,
        "output_dir": str(run_dir),
        "experiment_type": "unknown",
        "variant": "",
        "seed": "",
    }

    config_path = run_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
        fields["variant"] = config.get("variant", "")
        fields["seed"] = config.get("seed", "")
        fields["task"] = config.get("task", "")
        fields["graph_type"] = config.get("graph_type", "")
        fields["node_mode"] = config.get("node_mode", "")
        fields["lambda_cons"] = config.get("lambda_cons", "")
        fields["lambda_diff"] = config.get("lambda_diff", "")
        fields["lambda_decouple"] = config.get("lambda_decouple", "")

    semantic_path = run_dir / "semantic_alignment_metrics.json"
    test_path = run_dir / "test_metrics.json"

    if semantic_path.exists():
        metrics = load_json(semantic_path)
        fields["experiment_type"] = "semantic_alignment"
        for key in SEMANTIC_KEYS:
            fields[key] = metrics.get(key)
        fields.update(flatten_nested("pathology_unavailable", metrics, SEMANTIC_KEYS))
        fields.update(flatten_nested("molecular_unavailable", metrics, SEMANTIC_KEYS))
        fields["case_count"] = metrics.get("case_count")
        fields["query_count"] = metrics.get("query_count")
        fields["anchor_count"] = metrics.get("anchor_count")
    elif test_path.exists():
        metrics = load_json(test_path)
        fields["experiment_type"] = "classification"
        for key in CLASSIFICATION_KEYS:
            fields[key] = metrics.get(key)

    fields.update(best_from_history(run_dir / "history.json"))
    if "best_val_accuracy" in fields:
        fields["best_acc"] = fields["best_val_accuracy"]
    elif "accuracy" in fields:
        fields["best_acc"] = fields["accuracy"]

    return fields


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics from output directories.")
    parser.add_argument("output_root", help="Directory containing per-run output folders")
    parser.add_argument("out_csv", help="Destination CSV path")
    parser.add_argument("--run_id", default=None, help="Only include output directories ending with this run id")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = []
    for run_dir in sorted(output_root.iterdir() if output_root.exists() else []):
        if not run_dir.is_dir():
            continue
        if args.run_id and not run_dir.name.endswith(args.run_id):
            continue
        if not ((run_dir / "semantic_alignment_metrics.json").exists() or (run_dir / "test_metrics.json").exists()):
            continue
        rows.append(run_fields(run_dir))

    preferred = [
        "run",
        "experiment_type",
        "variant",
        "seed",
        "task",
        "graph_type",
        "node_mode",
        "lambda_cons",
        "lambda_diff",
        "lambda_decouple",
        "best_acc",
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr",
        "pair_auc",
        "edge_precision@25",
        "anchor_consistency",
        "positive_negative_distance_gap",
        "pathology_unavailable_mrr",
        "molecular_unavailable_mrr",
        "auc",
        "accuracy",
        "f1_macro",
        "balanced_accuracy",
        "best_val_mrr",
        "best_val_recall@1",
        "best_val_auc",
        "case_count",
        "query_count",
        "anchor_count",
        "output_dir",
    ]
    all_keys = sorted({key for row in rows for key in row})
    fieldnames = preferred + [key for key in all_keys if key not in preferred]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
