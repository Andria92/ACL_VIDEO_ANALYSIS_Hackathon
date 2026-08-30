"""Compute M5.9 phase-based movement story artifacts for a human case."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.annotations.registry import case_by_slug
from acl_motion.annotations.storage import load_movement_window_json
from acl_motion.semantics.phases import (
    PhaseSegmentationConfig,
    segment_movement_phases,
    write_phase_json,
)
from acl_motion.semantics.plots import plot_phase_change_score, plot_phase_timeline

TRANSITION_COLUMNS = [
    "transition_frame",
    "transition_timestamp",
    "movement_end_relative_ms",
    "from_phase_id",
    "to_phase_id",
    "change_score",
    "sustained_shift_score",
    "feature_family_contributions",
    "dominant_feature_families",
    "boundary_source",
    "refinement_depth",
    "parent_phase_id",
    "parent_phase_start_frame",
    "parent_phase_end_frame",
    "local_parent_phase_id",
    "local_parent_phase_start_frame",
    "local_parent_phase_end_frame",
    "evidence",
]


def main() -> int:
    args = parse_args()
    case = case_by_slug(args.case_slug, None)
    dynamic_df = pd.read_parquet(args.dynamic_features)
    case_summary = pd.read_parquet(args.case_feature_summary)
    path_df = pd.read_parquet(args.path_features)
    movement_window = load_movement_window_json(args.movement_window)
    result = segment_movement_phases(
        case_id=case.case_id,
        source_id=case.source_id,
        dynamic_df=dynamic_df,
        case_summary=case_summary,
        path_df=path_df,
        movement_window=movement_window,
        config=PhaseSegmentationConfig(),
    )
    phase_output = Path(args.phase_output)
    frame_map_output = Path(args.frame_map_output)
    change_score_output = Path(args.change_score_output)
    transition_output = Path(args.transition_output)
    for path in (phase_output, frame_map_output, change_score_output, transition_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_phase_json(result, phase_output)
    result.frame_map.to_parquet(frame_map_output, index=False)
    result.change_signal.to_parquet(change_score_output, index=False)
    pd.DataFrame(result.transitions, columns=TRANSITION_COLUMNS).to_csv(
        transition_output,
        index=False,
    )
    diagnostics_dir = Path(args.diagnostics_dir)
    phase_payload = result.to_json_dict()
    plot_phase_change_score(
        result.change_signal,
        phase_payload["phases"],
        phase_payload["transitions"],
        diagnostics_dir / f"{args.prefix}_multivariate_change_score.png",
    )
    plot_phase_timeline(
        phase_payload["phases"],
        diagnostics_dir / f"{args.prefix}_phase_timeline.png",
    )
    print(f"Wrote movement phases to {phase_output}")
    print(f"Wrote phase frame map to {frame_map_output}")
    print(f"Wrote movement change score to {change_score_output}")
    print(f"Wrote phase transitions to {transition_output}")
    print(json.dumps(_json_ready(_summary(result)), indent=2, allow_nan=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-slug", default="christen_press")
    parser.add_argument("--movement-window", default="data/annotations/human/christen_press_movement_window_human.json")
    parser.add_argument("--dynamic-features", default="data/dynamics/human/christen_press_dynamic_features.parquet")
    parser.add_argument(
        "--case-feature-summary",
        default="data/analytics/human/christen_press_case_feature_summary.parquet",
    )
    parser.add_argument("--path-features", default="data/path/human/christen_press_projected_movement_path.parquet")
    parser.add_argument("--phase-output", default="data/phases/human/christen_press_movement_phases.json")
    parser.add_argument("--frame-map-output", default="data/phases/human/christen_press_phase_frame_map.parquet")
    parser.add_argument(
        "--change-score-output",
        default="data/phases/human/christen_press_movement_change_score.parquet",
    )
    parser.add_argument("--transition-output", default="data/phases/human/christen_press_phase_transitions.csv")
    parser.add_argument("--diagnostics-dir", default="data/diagnostics/human")
    parser.add_argument("--prefix", default="christen_press_human")
    return parser.parse_args()


def _summary(result) -> dict:
    presentation_mode = str(result.metadata.get("presentation_mode") or "")
    evidence_interval = presentation_mode == "SUPPORTED_EVIDENCE_INTERVAL"
    return {
        "status": result.status,
        "presentation_mode": presentation_mode,
        "phase_count": 0 if evidence_interval else len(result.phases),
        "internal_segment_count": len(result.phases),
        "supported_evidence_interval": (
            {
                "start_frame": result.phases[0].start_frame,
                "end_frame": result.phases[0].end_frame,
                "duration_ms": result.phases[0].duration_ms,
                "evidence_status": result.phases[0].evidence_summary["evidence_status"],
            }
            if evidence_interval and result.phases
            else None
        ),
        "phases": [
            {
                "phase_id": phase.phase_id,
                "title": phase.title,
                "start_frame": phase.start_frame,
                "end_frame": phase.end_frame,
                "duration_ms": phase.duration_ms,
                "evidence_status": phase.evidence_summary["evidence_status"],
            }
            for phase in result.phases
        ],
        "transition_frames": [item["transition_frame"] for item in result.transitions],
        "eligible_descriptor_count": len(result.eligible_descriptors),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
