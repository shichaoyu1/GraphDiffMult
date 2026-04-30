import argparse
import csv
import math
from pathlib import Path


TABLE_COLUMNS = [
    ("variant", "Method"),
    ("seed", "Seed"),
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("mrr", "MRR"),
    ("pair_auc", "Pair AUC"),
    ("edge_precision@25", "Edge P@25"),
    ("anchor_consistency", "Anchor"),
]


def latex_escape(value):
    return str(value).replace("_", "\\_")


def as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def format_value(key, value):
    number = as_float(value)
    if number is None:
        return latex_escape(value) if value not in (None, "") else "--"
    if key == "seed":
        return str(int(number))
    return f"{number:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Create a LaTeX table from summary CSV.")
    parser.add_argument("csv_file")
    parser.add_argument("tex_file")
    parser.add_argument("--type", default="semantic_alignment")
    parser.add_argument("--sort_metric", default="mrr")
    args = parser.parse_args()

    with open(args.csv_file, "r", encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file)]

    rows = [row for row in rows if row.get("experiment_type") == args.type]
    rows.sort(key=lambda row: as_float(row.get(args.sort_metric)) or -1e9, reverse=True)

    tex_path = Path(args.tex_file)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" * 2 + "c" * (len(TABLE_COLUMNS) - 2)
    with open(tex_path, "w", encoding="utf-8") as file:
        file.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        file.write("\\toprule\n")
        file.write(" & ".join(header for _, header in TABLE_COLUMNS) + " \\\\\n")
        file.write("\\midrule\n")
        for row in rows:
            values = [format_value(key, row.get(key)) for key, _ in TABLE_COLUMNS]
            file.write(" & ".join(values) + " \\\\\n")
        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")

    print(f"Saved: {tex_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
