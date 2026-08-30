"""Validate a predicted pose table against manually labelled or laboratory reference joints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.validation.pose_reference import validate_pose_against_reference


def main() -> int:
    args = parse_args()
    report = validate_pose_against_reference(
        _read_table(Path(args.predicted)),
        _read_table(Path(args.reference)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote pose-reference validation report to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, help="Predicted pose .parquet or .csv.")
    parser.add_argument(
        "--reference",
        required=True,
        help=(
            "Reference .parquet or .csv with source_frame_index, landmark_name, x_px, y_px, "
            "and optional visible and normalizer_px columns."
        ),
    )
    parser.add_argument("--output", required=True, help="Output validation JSON.")
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Pose tables must use .parquet or .csv.")


if __name__ == "__main__":
    raise SystemExit(main())
