"""Report judge-human agreement (Cohen's kappa) per behavioral dimension.
Joins the labeled blind CSV against the judge sidecar from sample_for_labeling.py.

Usage:
    python scripts/compute_kappa.py
"""

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.stats import cohen_kappa  # noqa: E402

_DIMENSION_CATEGORIES = {
    "compliance": [0.0, 0.25, 0.5, 0.75, 1.0],
    "hedging": [0.0, 0.5, 1.0],
    "sycophancy": [0.0, 0.5, 1.0],
    "deception_markers": [0.0, 0.5, 1.0],
}

_KAPPA_BANDS = [
    (0.0, "slight or worse — judge is not a reliable proxy for human judgment here"),
    (0.2, "fair"),
    (0.4, "moderate"),
    (0.6, "substantial"),
    (0.8, "almost perfect"),
]


def _band(kappa: float) -> str:
    label = _KAPPA_BANDS[0][1]
    for threshold, text in _KAPPA_BANDS:
        if kappa >= threshold:
            label = text
    return label


def _nearest_category(value: float, categories: list[float]) -> float:
    return min(categories, key=lambda c: abs(c - value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", default=str(_REPO_ROOT / "data" / "human_labels_blind.csv"))
    parser.add_argument(
        "--judge-sidecar", default=str(_REPO_ROOT / "data" / "human_labels_judge_sidecar.csv")
    )
    parser.add_argument(
        "--report-output", default=str(_REPO_ROOT / "docs" / "judge-calibration-report.md")
    )
    args = parser.parse_args()

    with open(args.blind, newline="", encoding="utf-8") as f:
        human_rows = {row["row_id"]: row for row in csv.DictReader(f)}

    with open(args.judge_sidecar, newline="", encoding="utf-8") as f:
        judge_rows = {row["row_id"]: row for row in csv.DictReader(f)}

    joined = []
    skipped_unlabeled = 0
    for row_id, human_row in human_rows.items():
        if row_id not in judge_rows:
            continue
        if not human_row.get("human_compliance", "").strip():
            skipped_unlabeled += 1
            continue
        joined.append((human_row, judge_rows[row_id]))

    if not joined:
        raise SystemExit(
            f"No labeled rows found. Fill in the human_* columns in {args.blind} "
            "before running this script."
        )

    print(f"{len(joined)} labeled rows ({skipped_unlabeled} still unlabeled).")

    lines = [
        "# Judge-human calibration report",
        "",
        f"n = {len(joined)} labeled turns. Quadratic-weighted Cohen's kappa per dimension "
        "(corrects for chance agreement, unlike raw percent-agreement).",
        "",
        "| Dimension | Kappa | Agreement | n |",
        "|---|---|---|---|",
    ]

    for dim, categories in _DIMENSION_CATEGORIES.items():
        human_vals, judge_vals = [], []
        for human_row, judge_row in joined:
            human_raw = human_row.get(f"human_{dim}", "").strip()
            judge_raw = judge_row.get(f"judge_{dim}", "").strip()
            if not human_raw or not judge_raw:
                continue
            human_vals.append(_nearest_category(float(human_raw), categories))
            judge_vals.append(_nearest_category(float(judge_raw), categories))

        if len(human_vals) < 2:
            print(f"  {dim}: not enough paired labels, skipping.")
            continue

        kappa = cohen_kappa(human_vals, judge_vals, categories, weights="quadratic")
        band = _band(kappa)
        print(f"  {dim}: kappa={kappa:.3f} ({band}), n={len(human_vals)}")
        lines.append(f"| {dim} | {kappa:.3f} | {band} | {len(human_vals)} |")

    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {args.report_output}")


if __name__ == "__main__":
    main()
