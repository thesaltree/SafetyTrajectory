import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.config import get_async_client, DEFAULT_MAX_TURNS
from src.engine import run_pair_loop
from src.utils import fetch_advbench_dataset, load_seed_prompts, merge_objectives

_REPO_ROOT = Path(__file__).resolve().parent
_OUTPUT_DIR = _REPO_ROOT / "outputs"
_OUTPUT_FILE = _OUTPUT_DIR / "comparative_safety_dataset.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_REPO_ROOT / "safety_trajectory.log", mode="a"),
    ],
)
logger = logging.getLogger("run_evals")

DEFAULT_TARGET_MODELS = [
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "openai/gpt-4o-mini",
    "qwen/qwen3.7-max",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-flash"
]
DEFAULT_ADVBENCH_N = 5
DEFAULT_TARGET_MAX_TOKENS = 512
DEFAULT_MAX_CONCURRENCY = 4


async def orchestrate(
    target_models: list[str],
    max_turns: int,
    advbench_n: int,
    target_max_tokens: int,
    dry_run: bool,
    output_file: Path,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    repeats: int = 1,
) -> None:
    logger.info("Loading seed prompts…")
    seed_prompts = load_seed_prompts()

    logger.info("Fetching %d AdvBench prompts…", advbench_n)
    advbench_prompts = fetch_advbench_dataset(num_prompts=advbench_n)

    objectives = merge_objectives(seed_prompts, advbench_prompts)
    logger.info(
        "Objective pool ready: %d objectives × %d models × %d repeats = %d total trajectories",
        len(objectives),
        len(target_models),
        repeats,
        len(objectives) * len(target_models) * repeats,
    )

    if dry_run:
        logger.info("=== DRY RUN ===")
        for obj in objectives:
            for model in target_models:
                for repeat_index in range(repeats):
                    logger.info(
                        "  [DRY RUN] Would evaluate: %s#%d → %s",
                        obj["objective_id"], repeat_index, model
                    )
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            logger.info("Resuming — loaded %d existing records.", len(existing_results))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load existing results (%s). Starting fresh.", exc)

    # repeat_index defaults to 0 for pre-existing result files recorded before
    # --repeats was added, so old single-trial runs still resume correctly.
    results_map = {
        (r["objective_id"], r["target_model"], r.get("repeat_index", 0)): r
        for r in existing_results
    }

    client = get_async_client()
    semaphore = asyncio.Semaphore(max_concurrency)

    def make_save_callback(k: tuple[str, str, int]):
        def save_callback(current_traj: dict):
            results_map[k] = current_traj
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(results_map.values()), f, indent=2, ensure_ascii=False)
        return save_callback

    async def process(obj: dict, target_model: str, repeat_index: int) -> None:
        key = (obj["objective_id"], target_model, repeat_index)
        existing_traj = results_map.get(key)
        label = f"{obj['objective_id']}#{repeat_index}"

        if existing_traj:
            has_enough_turns = len(existing_traj.get("turns", [])) >= max_turns
            is_early_stopped = existing_traj.get("early_stopped", False)
            if has_enough_turns or is_early_stopped:
                logger.info(
                    "Skipping %s → %s (already completed: %d turns)",
                    label, target_model, len(existing_traj.get("turns", []))
                )
                return
            logger.info(
                "Resuming %s → %s from turn %d",
                label, target_model, len(existing_traj["turns"]) + 1
            )

        async with semaphore:
            try:
                trajectory = await run_pair_loop(
                    objective=obj,
                    target_model=target_model,
                    client=client,
                    max_turns=max_turns,
                    target_max_tokens=target_max_tokens,
                    existing_turns=existing_traj.get("turns") if existing_traj else None,
                    on_turn_complete=make_save_callback(key),
                    repeat_index=repeat_index,
                )
                results_map[key] = trajectory
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(list(results_map.values()), f, indent=2, ensure_ascii=False)
                logger.info(
                    "✓ Completed trajectory: %s → %s  (total records: %d)",
                    label,
                    target_model,
                    len(results_map),
                )
            except Exception as exc:
                logger.error(
                    "Unhandled error for %s → %s: %s. Skipping.",
                    label,
                    target_model,
                    exc,
                    exc_info=True,
                )

    await asyncio.gather(*(
        process(obj, target_model, repeat_index)
        for obj in objectives
        for target_model in target_models
        for repeat_index in range(repeats)
    ))

    logger.info(
        "=== Evaluation complete — %d records saved to %s ===",
        len(results_map),
        output_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_evals",
        description="SafetyTrajectory PAIR evaluation orchestrator.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_TARGET_MODELS,
        metavar="MODEL",
        help=f"Target models. Default: {DEFAULT_TARGET_MODELS}",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum PAIR turns (default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--target-max-tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_TOKENS,
        help=f"Token budget ceiling (default: {DEFAULT_TARGET_MAX_TOKENS})",
    )
    parser.add_argument(
        "--advbench-n",
        type=int,
        default=DEFAULT_ADVBENCH_N,
        metavar="N",
        help=f"Number of AdvBench prompts (default: {DEFAULT_ADVBENCH_N})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Independent trials per (objective, model) pair (default: 1). "
            "A single trial produces a noisy point estimate of jailbreak rate; "
            "increase this for a real sample size."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Max concurrent (objective, model) trajectories in flight (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview runs without calling API.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_FILE),
        help=f"Path to output JSON (default: {_OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("SafetyTrajectory PAIR starting…")
    logger.info("  Target models : %s", args.models)
    logger.info("  Max turns     : %d", args.max_turns)
    logger.info("  Max tokens    : %d", args.target_max_tokens)
    logger.info("  AdvBench N    : %d", args.advbench_n)
    logger.info("  Repeats       : %d", args.repeats)
    logger.info("  Max concurrency: %d", args.max_concurrency)
    logger.info("  Dry run       : %s", args.dry_run)
    logger.info("  Output file   : %s", args.output)

    asyncio.run(
        orchestrate(
            target_models=args.models,
            max_turns=args.max_turns,
            advbench_n=args.advbench_n,
            target_max_tokens=args.target_max_tokens,
            dry_run=args.dry_run,
            output_file=Path(args.output),
            max_concurrency=args.max_concurrency,
            repeats=args.repeats,
        )
    )


if __name__ == "__main__":
    main()