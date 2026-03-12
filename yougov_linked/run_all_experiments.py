#!/usr/bin/env python3
"""
Batch Experiment Runner — YouGov Linked Pipeline

Runs all 3 default OpenRouter models in sequence, then triggers analysis.

Usage:
    python run_all_experiments.py                    # full run
    python run_all_experiments.py --quick            # quick test (10 trials, 1000 posts)
    python run_all_experiments.py --models openai/gpt-4o-mini
    python run_all_experiments.py --skip-analysis
    python run_all_experiments.py --skip-experiments
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-5",
    "google/gemini-2.0-flash-001",
]

SCRIPT_DIR = Path(__file__).parent


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command, print timing, return True on success."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    start = time.time()

    result = subprocess.run(cmd, cwd=SCRIPT_DIR.parent)

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    status = "✓ SUCCESS" if result.returncode == 0 else "✗ FAILED"
    print(f"\n{status} ({mins}m {secs}s): {description}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all YouGov linked experiments across 3 OpenRouter models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"OpenRouter model IDs to run (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 10 trials, 1000 posts, pool-size 50, k 5",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments, only run analysis",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis, only run experiments",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials per style",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Override dataset size",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=60,
        help="OpenRouter rate limit (default: 60)",
    )
    args = parser.parse_args()

    # Determine trial / dataset parameters
    if args.quick:
        n_trials = args.n_trials or 10
        dataset_size = args.dataset_size or 1000
        pool_size = 50
        k = 5
    else:
        n_trials = args.n_trials or 100
        dataset_size = args.dataset_size or 5000
        pool_size = 100
        k = 10

    run_script = SCRIPT_DIR / "run_experiment.py"
    analysis_script = SCRIPT_DIR / "run_analysis.py"

    print("=" * 70)
    print("YOUGOV LINKED BATCH RUNNER")
    print("=" * 70)
    print(f"Models:       {', '.join(args.models)}")
    print(f"Trials/style: {n_trials}")
    print(f"Dataset size: {dataset_size}")
    print(f"Pool size:    {pool_size}")
    print(f"k:            {k}")
    print(f"Quick mode:   {args.quick}")
    print()

    failed = []

    # ------------------------------------------------------------------
    # Phase 1: Experiments
    # ------------------------------------------------------------------
    if not args.skip_experiments:
        print(f"\nPhase 1: Running experiments ({len(args.models)} models × 6 prompts)")
        for model in args.models:
            cmd = [
                sys.executable, str(run_script),
                "--model", model,
                "--n-trials", str(n_trials),
                "--dataset-size", str(dataset_size),
                "--pool-size", str(pool_size),
                "--k", str(k),
                "--requests-per-minute", str(args.requests_per_minute),
            ]
            ok = run_command(cmd, f"Experiment: {model}")
            if not ok:
                failed.append(f"Experiment: {model}")
    else:
        print("Skipping experiments (--skip-experiments)")

    # ------------------------------------------------------------------
    # Phase 2: Analysis
    # ------------------------------------------------------------------
    if not args.skip_analysis:
        print(f"\nPhase 2: Running analysis")
        cmd = [sys.executable, str(analysis_script)]
        ok = run_command(cmd, "Comprehensive analysis")
        if not ok:
            failed.append("Analysis")
    else:
        print("Skipping analysis (--skip-analysis)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    if failed:
        print(f"FAILURES ({len(failed)}):")
        for f in failed:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")
        print(f"\nOutputs: {SCRIPT_DIR}/outputs/experiments/")
        print(f"Analysis: {SCRIPT_DIR}/analysis_outputs/")


if __name__ == "__main__":
    main()
