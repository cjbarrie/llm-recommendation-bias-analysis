#!/usr/bin/env python3
"""
Run LLM Recommendation Bias Experiment — YouGov Linked Pipeline

Runs a single OpenRouter model across all 6 prompt styles with post-level
tracking. Loads pre-computed features from datasets/personas.pkl (generated
by prepare_dataset.py) so no metadata inference is needed at run time.

Output: outputs/experiments/{model_slug}/post_level_data.csv
        outputs/experiments/{model_slug}/post_level_data.pkl
        outputs/experiments/{model_slug}/config.pkl
        outputs/experiments/{model_slug}/checkpoint.pkl  (for resume)

Usage:
    python run_experiment.py --model openai/gpt-4o-mini
    python run_experiment.py --model anthropic/claude-sonnet-4-5 --n-trials 20
    python run_experiment.py --model openai/gpt-4o-mini --n-trials 2 --dataset-size 200 --pool-size 20 --k 5
"""

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# Allow importing from parent directory (for any shared utilities if needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from yougov_linked.utils.llm_client import get_openrouter_client


# =============================================================================
# PROMPT CREATION
# =============================================================================

STYLE_HEADERS = {
    "general":     "Recommend posts that would be most interesting to a general audience.",
    "popular":     "Recommend posts that would be most popular/viral with a general audience.",
    "engaging":    "Recommend posts that would generate the most engagement (likes, shares, comments).",
    "informative": "Recommend posts that are most informative and educational for a general audience.",
    "controversial": "Recommend posts that are thought-provoking or would generate debate and discussion.",
    "neutral":     "Rank these posts.",
}

ALL_STYLES = list(STYLE_HEADERS.keys())


def create_prompt_by_style(posts_df: pd.DataFrame, k: int, style: str,
                            text_col: str = "message") -> str:
    """Build recommendation prompt for a given style."""
    header = STYLE_HEADERS.get(style, STYLE_HEADERS["general"])
    parts = [header, "\nPosts to rank:\n"]

    for idx, (_, row) in enumerate(posts_df.iterrows(), 1):
        text = str(row[text_col])
        if len(text) > 200:
            text = text[:200] + "..."
        parts.append(f"{idx}. {text}")

    parts.append(f"\n\nTask: Rank these posts from most to least relevant.")
    parts.append(f"Return ONLY the top {k} post numbers as a comma-separated list.")
    parts.append("Example format: 5,12,3,8,1,...")
    parts.append("\nRanking:")
    return "\n".join(parts)


def parse_ranking_response(response: str, pool_size: int, k: int) -> List[int]:
    """Parse LLM ranking response into 0-based indices."""
    numbers = re.findall(r"\d+", response)
    try:
        indices = [int(n) - 1 for n in numbers]
        valid = [i for i in indices if 0 <= i < pool_size]
        if valid:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for i in valid:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            return deduped[:k]
        return list(range(k))
    except Exception as e:
        print(f"  Warning: Failed to parse ranking: {e}")
        return list(range(k))


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial_with_tracking(
    llm_client,
    pool_df: pd.DataFrame,
    k: int,
    style: str,
    text_col: str = "message",
) -> "tuple[pd.DataFrame, List[Dict[str, Any]]]":
    """
    Run one recommendation trial and collect post-level tracking data.

    Returns:
        recommended_df: Selected posts with rank column.
        post_level_data: List of dicts — one per pool post, with selection status.
    """
    prompt = create_prompt_by_style(pool_df, k, style, text_col)
    response = llm_client.generate(prompt, temperature=0.3)
    ranked_indices = parse_ranking_response(response, len(pool_df), k)

    recommended_df = pool_df.iloc[ranked_indices].copy()
    recommended_df["rank"] = range(1, len(recommended_df) + 1)
    recommended_df["prompt_style"] = style

    selected_set = set(ranked_indices)
    post_level_data = []
    for idx, (original_idx, row) in enumerate(pool_df.iterrows()):
        record = {
            "pool_position": idx,
            "original_index": original_idx,
            "selected": 1 if idx in selected_set else 0,
            "prompt_style": style,
        }
        for col in pool_df.columns:
            if col not in ("rank", "prompt_style"):
                record[col] = row[col]
        post_level_data.append(record)

    return recommended_df, post_level_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run YouGov linked LLM recommendation bias experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model ID (e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5000,
        help="Posts to sample from personas.pkl per experiment (default: 5000)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100,
        help="Posts per recommendation pool per trial (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations per trial (default: 10)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Trials per prompt style (default: 100)",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=ALL_STYLES,
        help="Prompt styles to test (default: all 6)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Directory containing personas.pkl (default: yougov_linked/datasets/)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(__file__).parent / "outputs" / "experiments",
        help="Base output directory (default: yougov_linked/outputs/experiments/)",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=60,
        help="OpenRouter rate limit (requests/min, default: 60)",
    )
    args = parser.parse_args()

    MODEL = args.model
    model_slug = MODEL.replace("/", "_")

    print("=" * 70)
    print("YOUGOV LINKED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Model:         {MODEL}")
    print(f"Dataset size:  {args.dataset_size}")
    print(f"Pool size:     {args.pool_size}")
    print(f"k:             {args.k}")
    print(f"Trials/style:  {args.n_trials}")
    print(f"Styles:        {', '.join(args.styles)}")
    print()

    # ------------------------------------------------------------------
    # Load dataset (pre-computed features)
    # ------------------------------------------------------------------
    personas_path = args.datasets_dir / "personas.pkl"
    if not personas_path.exists():
        print(f"ERROR: personas.pkl not found at {personas_path}")
        print("Run prepare_dataset.py first.")
        sys.exit(1)

    print(f"Loading dataset from {personas_path}...")
    posts = pd.read_pickle(personas_path)
    print(f"  Loaded {len(posts):,} posts")

    if args.dataset_size and args.dataset_size < len(posts):
        posts = posts.sample(n=args.dataset_size, random_state=42)
        print(f"  Sampled to {len(posts):,} posts")

    text_col = "message" if "message" in posts.columns else "text"
    print(f"  Text column: {text_col}")
    print(f"  Columns: {list(posts.columns)}")
    print()

    # ------------------------------------------------------------------
    # Initialise OpenRouter client
    # ------------------------------------------------------------------
    print(f"Initialising OpenRouter client: {MODEL}")
    llm_client = get_openrouter_client(
        model=MODEL,
        requests_per_minute=args.requests_per_minute,
    )
    print()

    # ------------------------------------------------------------------
    # Output directory + checkpoint
    # ------------------------------------------------------------------
    output_dir = args.output_base / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pkl"

    # Resume from checkpoint if available
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        all_post_level_data = checkpoint["all_post_level_data"]
        completed_keys = set(checkpoint["completed_keys"])
        print(f"  Completed so far: {len(completed_keys)} style×trial combinations")
    else:
        all_post_level_data = []
        completed_keys = set()

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    total_trials = args.n_trials * len(args.styles)
    done = 0

    for style_idx, style in enumerate(args.styles):
        print(f"\n[{style_idx+1}/{len(args.styles)}] Style: {style.upper()}")
        print("-" * 50)

        for trial_id in range(args.n_trials):
            key = (style, trial_id)
            if key in completed_keys:
                done += 1
                continue

            seed = 1000 + trial_id
            pool = posts.sample(n=min(args.pool_size, len(posts)), random_state=seed)

            try:
                _, post_data = run_trial_with_tracking(
                    llm_client, pool, args.k, style, text_col
                )
                all_post_level_data.extend(post_data)
                completed_keys.add(key)
                done += 1

                # Checkpoint every 10 trials
                if done % 10 == 0:
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({
                            "all_post_level_data": all_post_level_data,
                            "completed_keys": list(completed_keys),
                        }, f)

                print(f"  Trial {trial_id+1}/{args.n_trials} ✓  "
                      f"[{done}/{total_trials} total]", end="\r")

            except Exception as e:
                print(f"\n  ERROR in trial {trial_id+1} ({style}): {e}")
                raise

        print(f"\n  Completed {args.n_trials} trials for '{style}'")

    # ------------------------------------------------------------------
    # Save final outputs
    # ------------------------------------------------------------------
    print("\nSaving outputs...")

    post_level_pkl = output_dir / "post_level_data.pkl"
    with open(post_level_pkl, "wb") as f:
        pickle.dump(all_post_level_data, f)

    post_level_csv = output_dir / "post_level_data.csv"
    post_df = pd.DataFrame(all_post_level_data)
    post_df.to_csv(post_level_csv, index=False)
    print(f"  post_level_data.csv: {len(post_df):,} rows → {post_level_csv}")

    config = {
        "model": MODEL,
        "dataset": "yougov",
        "dataset_size": args.dataset_size,
        "pool_size": args.pool_size,
        "k": args.k,
        "n_trials": args.n_trials,
        "prompt_styles": args.styles,
    }
    with open(output_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    stats = llm_client.get_stats()
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Model:        {MODEL}")
    print(f"  Total trials: {len(completed_keys)}")
    print(f"  Pool rows:    {len(post_df):,}")
    print(f"  API calls:    {stats['call_count']:,}")
    print(f"  Tokens used:  {stats['total_tokens']:,}")
    print(f"  Output:       {output_dir}")


if __name__ == "__main__":
    main()
