"""Compute Milestone 3 projected geometry features from processed pose."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.cases.models import InjurySide
from acl_motion.geometry.features import (
    FEATURE_SET_VERSION,
    GEOMETRY_VERSION,
    build_feature_completeness,
    compute_geometry_features,
)


def main() -> int:
    args = parse_args()
    processed_pose = pd.read_parquet(args.processed_pose)
    feature_df, normalisation = compute_geometry_features(
        processed_pose,
        injured_side=InjurySide(args.injured_side),
    )
    completeness = build_feature_completeness(feature_df)
    summary = build_summary(
        processed_pose,
        feature_df,
        completeness,
        normalisation,
        args,
    )

    output_path = Path(args.output)
    completeness_path = Path(args.completeness_output)
    summary_path = Path(args.summary_output)
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(
        ".metadata.json"
    )
    for path in (output_path, completeness_path, summary_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(output_path, index=False)
    completeness.to_csv(completeness_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(summary["run_metadata"], indent=2), encoding="utf-8")

    print(f"Wrote {len(feature_df)} feature rows to {output_path}")
    print(f"Wrote feature completeness to {completeness_path}")
    print(f"Wrote feature summary to {summary_path}")
    print(f"Wrote run metadata to {metadata_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-pose", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--completeness-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument(
        "--injured-side",
        default=InjurySide.UNKNOWN.value,
        choices=[side.value for side in InjurySide],
    )
    return parser.parse_args()


def build_summary(processed_pose, feature_df, completeness, normalisation, args: argparse.Namespace) -> dict:
    first = processed_pose.iloc[0]
    generated_at = datetime.now(UTC).isoformat()
    run_metadata = {
        "run_id": f"{first.get('case_id', 'case')}_m3_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": generated_at,
        "case_id": first.get("case_id"),
        "source_id": first.get("source_id"),
        "processed_pose_input": args.processed_pose,
        "pose_backend": first.get("backend"),
        "pose_model": _pose_model(first.get("backend_metadata")),
        "geometry_version": GEOMETRY_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "injured_side": args.injured_side,
        "normalisation_method": normalisation.method,
        "normalisation_reference": normalisation.to_metadata(),
    }
    return {
        "run_metadata": run_metadata,
        "total_feature_rows": len(feature_df),
        "feature_count": int(feature_df["feature_name"].nunique()),
        "frame_count": int(feature_df["frame_index"].nunique()),
        "status_counts": {
            str(status): int(count)
            for status, count in feature_df["status"].value_counts().sort_index().items()
        },
        "supported_counts_by_feature": {
            row["feature_name"]: int(row["supported_frames"])
            for _, row in completeness.iterrows()
        },
        "completeness_by_feature": {
            row["feature_name"]: float(row["completeness"])
            for _, row in completeness.iterrows()
        },
        "primary_rejection_reason_by_feature": {
            row["feature_name"]: row["primary_rejection_reason"]
            for _, row in completeness.iterrows()
        },
    }


def _pose_model(metadata) -> str | None:
    if isinstance(metadata, dict):
        return metadata.get("model_name") or metadata.get("model_path")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
