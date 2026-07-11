"""Remove trajectories containing any judge_failed turn so a resume regenerates them fresh.
Backs up the original file before writing.

Usage:
    python scripts/strip_corrupted_trajectories.py outputs/safety_trajectory_6x24.json
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    backup_path = args.output_file.with_suffix(args.output_file.suffix + ".bak")
    shutil.copy2(args.output_file, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(args.output_file, encoding="utf-8") as f:
        data = json.load(f)

    kept, removed = [], []
    for traj in data:
        has_failure = any(
            turn.get("metrics", {}).get("judge_failed") for turn in traj.get("turns", [])
        )
        (removed if has_failure else kept).append(traj)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"Kept {len(kept)} clean trajectories, removed {len(removed)} corrupted ones:")
    for traj in removed:
        n_failed = sum(1 for t in traj["turns"] if t.get("metrics", {}).get("judge_failed"))
        print(f"  {traj['category']:16s} {traj['target_model']:35s} ({n_failed}/{len(traj['turns'])} turns failed)")


if __name__ == "__main__":
    main()
