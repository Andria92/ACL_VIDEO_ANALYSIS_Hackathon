"""Generate Milestone 4.1 dynamic reliability plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.visualisation.dynamics import (
    plot_dynamic_spike_audit,
    plot_hka_raw_vs_robust_velocity,
)


def main() -> int:
    args = parse_args()
    dynamic_df = pd.read_parquet(args.dynamic_features)
    spike_audit = pd.read_csv(args.spike_audit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_hka_raw_vs_robust_velocity(
            dynamic_df,
            output_dir / f"{args.prefix}_hka_raw_vs_robust_velocity.png",
        ),
        plot_dynamic_spike_audit(
            dynamic_df,
            spike_audit,
            output_dir / f"{args.prefix}_dynamic_spike_audit.png",
        ),
    ]
    for path in paths:
        print(path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-features", required=True)
    parser.add_argument("--spike-audit", required=True)
    parser.add_argument("--output-dir", default="data/diagnostics")
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
