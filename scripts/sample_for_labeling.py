"""Sample judge-scored turns into a blind CSV for human calibration labeling
(judge scores go to a separate sidecar so labeling stays blind).

Usage:
    python scripts/sample_for_labeling.py --n 40
"""

import argparse
import csv
import glob
import hashlib
import json
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUTS_GLOB = str(_REPO_ROOT / "outputs" / "*.json")

_BLIND_FIELDS = [
    "row_id", "category", "adversarial_prompt", "target_response",
    "human_compliance", "human_hedging", "human_sycophancy",
    "human_deception_markers", "labeler_notes",
]
_SIDECAR_FIELDS = [
    "row_id", "objective_id", "target_model", "turn",
    "judge_compliance", "judge_hedging", "judge_sycophancy", "judge_deception_markers",
]


def _row_id(objective_id: str, target_model: str, turn: int) -> str:
    raw = f"{objective_id}|{target_model}|{turn}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_all_turns(outputs_glob: str) -> list[dict]:
    """Flatten scoreable turns across output files, skipping judge_failed/blocked_by_filter."""
    records = []
    for path in glob.glob(outputs_glob):
        try:
            with open(path, "r", encoding="utf-8") as f:
                trajectories = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for traj in trajectories:
            for turn_data in traj.get("turns", []):
                metrics = turn_data.get("metrics", {})
                if metrics.get("judge_failed") or turn_data.get("blocked_by_filter"):
                    continue
                records.append({
                    "row_id": _row_id(traj["objective_id"], traj["target_model"], turn_data["turn"]),
                    "objective_id": traj["objective_id"],
                    "category": traj.get("category", "Unknown"),
                    "target_model": traj["target_model"],
                    "turn": turn_data["turn"],
                    "adversarial_prompt": turn_data["adversarial_prompt"],
                    "target_response": turn_data["target_response"],
                    "judge_compliance": metrics.get("compliance"),
                    "judge_hedging": metrics.get("hedging"),
                    "judge_sycophancy": metrics.get("sycophancy"),
                    "judge_deception_markers": metrics.get("deception_markers"),
                })
    return records


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Sample roughly evenly across categories."""
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)

    categories = list(by_category.keys())
    if not categories:
        return []

    per_category = max(1, n // len(categories))
    sample_ids = set()
    sample = []
    for cat in categories:
        pool = by_category[cat]
        rng.shuffle(pool)
        for r in pool[:per_category]:
            sample_ids.add(r["row_id"])
            sample.append(r)

    if len(sample) < n:
        remaining = [r for r in records if r["row_id"] not in sample_ids]
        rng.shuffle(remaining)
        sample.extend(remaining[: n - len(sample)])

    rng.shuffle(sample)
    return sample[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-glob", default=_DEFAULT_OUTPUTS_GLOB)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--blind-output", default=str(_REPO_ROOT / "data" / "human_labels_blind.csv")
    )
    parser.add_argument(
        "--judge-sidecar", default=str(_REPO_ROOT / "data" / "human_labels_judge_sidecar.csv")
    )
    args = parser.parse_args()

    records = load_all_turns(args.outputs_glob)
    if not records:
        raise SystemExit(
            f"No scoreable turns found under {args.outputs_glob}. Run run_evals.py first."
        )

    sample = stratified_sample(records, args.n, args.seed)
    if len(sample) < args.n:
        print(f"Warning: only {len(sample)} scoreable turns available (requested {args.n}).")

    with open(args.blind_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BLIND_FIELDS)
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "row_id": r["row_id"],
                "category": r["category"],
                "adversarial_prompt": r["adversarial_prompt"],
                "target_response": r["target_response"],
                "human_compliance": "",
                "human_hedging": "",
                "human_sycophancy": "",
                "human_deception_markers": "",
                "labeler_notes": "",
            })

    with open(args.judge_sidecar, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SIDECAR_FIELDS)
        writer.writeheader()
        for r in sample:
            writer.writerow({k: r[k] for k in _SIDECAR_FIELDS})

    print(f"Wrote {len(sample)} rows to {args.blind_output}")
    print(
        "Label these using the exact rubric in src/judge.py's _JUDGE_SYSTEM_PROMPT — "
        "same scale, same dimensions, so the comparison is apples-to-apples."
    )
    print(f"Judge scores withheld in {args.judge_sidecar} — do not open this while labeling.")
    print("Run scripts/compute_kappa.py once human_* columns are filled in.")


if __name__ == "__main__":
    main()
