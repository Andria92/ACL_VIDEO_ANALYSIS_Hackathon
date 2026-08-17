"""Run Milestone 2 pose reliability and coordinate processing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.processing.pose_processing import ProcessingConfig, process_pose_coordinates
from acl_motion.quality.pose_quality import (
    QualityConfig,
    build_target_identity_diagnostics,
    classify_pose_quality,
)
from acl_motion.quality.reliability import build_reliability_summary
from acl_motion.visualisation.quality import plot_availability_timeline, plot_raw_clean_smoothed


def main() -> int:
    args = parse_args()
    raw_pose = read_table(Path(args.raw_pose))
    quality_config = QualityConfig(
        landmark_confidence_threshold=args.landmark_confidence_threshold,
        frame_median_confidence_threshold=args.frame_median_confidence_threshold,
        core_confidence_threshold=args.core_confidence_threshold,
        temporal_jump_min_px=args.temporal_jump_min_px,
        temporal_jump_body_scale_multiplier=args.temporal_jump_body_scale_multiplier,
        suspicious_padding_frames=args.suspicious_padding_frames,
        min_valid_segment_frames=args.min_valid_segment_frames,
    )
    processing_config = ProcessingConfig(
        max_interpolation_gap_frames=args.max_interpolation_gap_frames,
        smoothing_window_frames=args.smoothing_window_frames,
    )

    frame_quality, landmark_quality = classify_pose_quality(raw_pose, quality_config)
    processed_pose = process_pose_coordinates(raw_pose, landmark_quality, processing_config)
    summary = build_reliability_summary(processed_pose, frame_quality)
    summary["quality_config"] = asdict(quality_config)
    summary["processing_config"] = asdict(processing_config)
    summary["raw_pose_path"] = args.raw_pose

    frame_quality_path = Path(args.frame_quality_output)
    landmark_quality_path = Path(args.landmark_quality_output)
    processed_path = Path(args.processed_output)
    summary_path = Path(args.summary_output)
    identity_path = Path(args.target_identity_output) if args.target_identity_output else None
    for path in (
        frame_quality_path,
        landmark_quality_path,
        processed_path,
        summary_path,
        *(filter(None, [identity_path])),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    frame_quality.to_csv(frame_quality_path, index=False)
    if identity_path is not None:
        build_target_identity_diagnostics(frame_quality).to_csv(identity_path, index=False)
    landmark_quality.to_parquet(landmark_quality_path, index=False)
    processed_pose.to_parquet(processed_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.raw_clean_plot:
        plot_raw_clean_smoothed(processed_pose, args.raw_clean_plot)
    if args.availability_plot:
        plot_availability_timeline(processed_pose, frame_quality, args.availability_plot)

    print(f"Wrote frame quality to {frame_quality_path}")
    if identity_path is not None:
        print(f"Wrote target identity diagnostics to {identity_path}")
    print(f"Wrote landmark quality to {landmark_quality_path}")
    print(f"Wrote processed pose to {processed_path}")
    print(f"Wrote reliability summary to {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pose", required=True)
    parser.add_argument("--frame-quality-output", required=True)
    parser.add_argument("--landmark-quality-output", required=True)
    parser.add_argument("--processed-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--target-identity-output", default=None)
    parser.add_argument("--raw-clean-plot", default=None)
    parser.add_argument("--availability-plot", default=None)
    parser.add_argument("--landmark-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--frame-median-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--core-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--temporal-jump-min-px", type=float, default=70.0)
    parser.add_argument("--temporal-jump-body-scale-multiplier", type=float, default=0.55)
    parser.add_argument("--suspicious-padding-frames", type=int, default=1)
    parser.add_argument("--min-valid-segment-frames", type=int, default=5)
    parser.add_argument("--max-interpolation-gap-frames", type=int, default=2)
    parser.add_argument("--smoothing-window-frames", type=int, default=5)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Raw pose path must be .parquet or .csv.")


if __name__ == "__main__":
    raise SystemExit(main())
