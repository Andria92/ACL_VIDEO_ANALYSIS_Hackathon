"""Build a Milestone 5 MovementProfile and evidence analytics outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.cases.annotations import load_event_annotation
from acl_motion.profiles.builder import (
    build_case_feature_summary,
    build_movement_profile,
    write_evidence_profile_json,
    write_profile_json,
)
from acl_motion.visualisation.profiles import (
    plot_body_region_coverage,
    plot_evidence_category_counts,
    plot_feature_reliability_overview,
    plot_movement_profile_overview,
    plot_rejection_reasons,
)


def main() -> int:
    args = parse_args()
    dynamic_df = pd.read_parquet(args.dynamic_features)
    event_annotation = load_event_annotation(args.event_annotation)
    reliability = _load_json(args.pose_reliability) if args.pose_reliability else {}
    dynamic_quality = _load_json(args.dynamic_quality) if args.dynamic_quality else {}
    geometry_summary = _load_json(args.geometry_summary) if args.geometry_summary else {}
    movement_window = _load_json(args.movement_window) if args.movement_window else {}
    profile = build_movement_profile(
        dynamic_df,
        event_annotation=event_annotation,
        pose_reliability_summary=reliability,
        dynamic_quality_summary=dynamic_quality,
        geometry_feature_summary=geometry_summary,
        manual_roi_keyframe_count=_csv_row_count(args.roi_keyframes),
        provenance={
            "dynamic_features_input": args.dynamic_features,
            "event_annotation_file": args.event_annotation,
            "pose_reliability_summary": args.pose_reliability,
            "dynamic_quality_summary": args.dynamic_quality,
            "geometry_summary": args.geometry_summary,
            "movement_window_annotation": args.movement_window,
            "movement_window": movement_window,
        },
    )
    summary = build_case_feature_summary(profile)

    write_profile_json(profile, args.profile_output)
    write_evidence_profile_json(profile.evidence_profile, args.evidence_output)
    summary_path = Path(args.case_feature_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_path, index=False)
    output_dir = Path(args.diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_movement_profile_overview(
        dynamic_df, profile, output_dir / f"{args.prefix}_movement_profile_overview.png"
    )
    plot_feature_reliability_overview(
        profile, output_dir / f"{args.prefix}_feature_reliability_overview.png"
    )
    plot_body_region_coverage(profile, output_dir / f"{args.prefix}_body_region_coverage.png")
    plot_evidence_category_counts(profile, output_dir / f"{args.prefix}_evidence_category_counts.png")
    plot_rejection_reasons(profile, output_dir / f"{args.prefix}_rejection_reasons.png")

    print(f"Wrote MovementProfile to {args.profile_output}")
    print(f"Wrote EvidenceProfile to {args.evidence_output}")
    print(f"Wrote case feature summary to {args.case_feature_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-features", required=True)
    parser.add_argument("--event-annotation", required=True)
    parser.add_argument("--pose-reliability")
    parser.add_argument("--dynamic-quality")
    parser.add_argument("--geometry-summary")
    parser.add_argument("--movement-window")
    parser.add_argument("--roi-keyframes")
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--case-feature-output", required=True)
    parser.add_argument("--diagnostics-dir", default="data/diagnostics")
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv_row_count(path: str | None) -> int | None:
    if not path:
        return None
    if not Path(path).exists():
        return None
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return max(0, len(lines) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
