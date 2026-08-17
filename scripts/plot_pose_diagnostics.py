"""Generate raw joint-coordinate diagnostic plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.visualisation.trajectories import (
    DEFAULT_DIAGNOSTIC_LANDMARKS,
    plot_joint_coordinate_diagnostics,
)


def main() -> int:
    args = parse_args()

    import pandas as pd

    pose_df = read_pose_table(Path(args.pose), pd)
    landmarks = tuple(args.landmarks.split(",")) if args.landmarks else DEFAULT_DIAGNOSTIC_LANDMARKS
    output = plot_joint_coordinate_diagnostics(
        pose_df,
        args.output,
        landmark_names=landmarks,
    )
    print(f"Wrote diagnostic plot to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, help="Pose .parquet or .csv file.")
    parser.add_argument("--output", required=True, help="Output image path, usually .png.")
    parser.add_argument(
        "--landmarks",
        default=None,
        help="Comma-separated landmark names. Defaults to shoulders, hips, knees, ankles.",
    )
    return parser.parse_args()


def read_pose_table(path: Path, pd):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Pose path must end in .parquet or .csv.")


if __name__ == "__main__":
    raise SystemExit(main())
