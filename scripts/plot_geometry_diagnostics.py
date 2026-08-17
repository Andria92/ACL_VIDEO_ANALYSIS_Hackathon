"""Generate Milestone 3 projected-geometry diagnostic graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.visualisation.geometry import (
    plot_feature_availability,
    plot_hka_bilateral_difference,
    plot_hka_trajectories,
    plot_trunk_pelvis_profile,
    plot_upper_limb_profile,
)


def main() -> int:
    args = parse_args()
    feature_df = pd.read_parquet(args.features)
    outputs = {
        "hka": plot_hka_trajectories(feature_df, args.hka_output),
        "hka_difference": plot_hka_bilateral_difference(feature_df, args.hka_difference_output),
        "trunk_pelvis": plot_trunk_pelvis_profile(feature_df, args.trunk_pelvis_output),
        "upper_limb": plot_upper_limb_profile(feature_df, args.upper_limb_output),
        "availability": plot_feature_availability(feature_df, args.availability_output),
    }
    for name, output in outputs.items():
        print(f"Wrote {name} plot to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--hka-output", required=True)
    parser.add_argument("--hka-difference-output", required=True)
    parser.add_argument("--trunk-pelvis-output", required=True)
    parser.add_argument("--upper-limb-output", required=True)
    parser.add_argument("--availability-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
