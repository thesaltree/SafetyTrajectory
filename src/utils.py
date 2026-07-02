import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SEED_CSV = _REPO_ROOT / "data" / "seed_prompts.csv"
_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)

_REQUIRED_COLUMNS = {"objective_id", "category", "source", "goal"}


def load_seed_prompts(csv_path: Optional[Path] = None) -> list[dict]:
    """Parse local seed CSV into a list of objective dictionaries."""
    path = Path(csv_path) if csv_path else _DEFAULT_SEED_CSV

    if not path.exists():
        raise FileNotFoundError(f"Seed prompt CSV not found at: {path}")

    df = pd.read_csv(path)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Seed CSV is missing required column(s): {missing}")

    records = df[list(_REQUIRED_COLUMNS)].dropna(subset=["goal"]).to_dict("records")
    logger.info("Loaded %d seed prompts from %s", len(records), path)
    return records


def fetch_advbench_dataset(num_prompts: int = 10) -> list[dict]:
    """Download official AdvBench harmful-behaviors CSV and return top-N rows."""
    try:
        logger.info("Fetching AdvBench dataset from %s", _ADVBENCH_URL)
        df = pd.read_csv(_ADVBENCH_URL)

        if "goal" not in df.columns:
            if "behavior" in df.columns:
                df = df.rename(columns={"behavior": "goal"})
            else:
                raise ValueError("AdvBench CSV missing behavior/goal column")

        sliced = df.head(num_prompts).copy()
        records = []
        for idx, row in sliced.iterrows():
            records.append(
                {
                    "objective_id": f"ADVBENCH_{idx:04d}",
                    "category": "AdvBench",
                    "source": "AdvBench",
                    "goal": str(row["goal"]),
                }
            )

        logger.info("Retrieved %d AdvBench prompts.", len(records))
        return records

    except Exception as exc:
        logger.error("Failed to fetch AdvBench dataset: %s. Continuing offline.", exc)
        return []


def merge_objectives(
    seed_prompts: list[dict],
    advbench_prompts: list[dict],
) -> list[dict]:
    """Merge seed and AdvBench objective lists, deduplicating by goal text."""
    seen_goals = set()
    merged = []

    for obj in seed_prompts + advbench_prompts:
        goal_key = obj["goal"].strip().lower()
        if goal_key not in seen_goals:
            seen_goals.add(goal_key)
            merged.append(obj)

    logger.info("Merged objective pool: %d total", len(merged))
    return merged
