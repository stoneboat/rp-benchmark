#!/usr/bin/env python
"""CLI entry point: run the demo benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rpbench.config import load_config
from rpbench.runners.run_demo import run_demo
from rpbench.utils.io import save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WF-001 demo benchmark")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Override output root directory",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_root:
        cfg.output_root = args.output_root

    base_seed, trial_seeds = cfg.resolve_trial_seeds()
    print(f"Running demo: {cfg.dataset}, mechanisms={cfg.mechanisms}")
    print(
        f"  epsilon_grid={cfg.epsilon_grid}, "
        f"seed_batch={{mode={cfg.seed_batch.mode}, base_seed={base_seed}, count={len(trial_seeds)}}}"
    )

    records = run_demo(cfg)

    out_path = Path(cfg.output_root) / "demo_results.jsonl"
    save_jsonl(records, out_path)
    print(f"\nSaved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
