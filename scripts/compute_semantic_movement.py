"""Compute M5.7 semantic movement observations for a human case."""

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
from acl_motion.annotations.storage import load_human_annotation_session, load_movement_window_json
from acl_motion.semantics.bilateral import compute_bilateral_hka_summary
from acl_motion.semantics.builder import build_movement_observations, write_observations_json
from acl_motion.semantics.path import (
    body_center_from_processed_pose,
    build_path_frame_diagnostics,
    compensate_projected_path,
    enforce_path_validation,
    estimate_background_camera_motion,
    estimate_background_camera_motion_affine,
    path_quality_summary,
    validate_projected_path,
)
from acl_motion.semantics.plots import (
    plot_bilateral_hka_relationship,
    plot_path_validation_comparison,
    plot_projected_heading,
    plot_projected_movement_path,
    plot_projected_speed,
    plot_semantic_summary,
)
from acl_motion.semantics.vocabulary import (
    build_observable_movement_description_payload,
    write_observable_descriptions_json,
)
from acl_motion.video.roi import RoiTimeline


def main() -> int:
    args = parse_args()
    case = case_by_slug(args.case_slug, None)
    dynamic_df = pd.read_parquet(args.dynamic_features)
    case_summary = pd.read_parquet(args.case_feature_summary)
    processed_pose = pd.read_parquet(args.processed_pose)
    frame_quality = pd.read_csv(args.frame_quality)
    feature_summary = json.loads(Path(args.feature_summary).read_text(encoding="utf-8"))
    session = load_human_annotation_session(args.annotation_session)
    movement_window = load_movement_window_json(args.movement_window)
    dynamic_df = _ensure_movement_timing(dynamic_df, movement_window)
    centers = body_center_from_processed_pose(processed_pose)
    timing = dynamic_df[
        [
            "source_frame_index",
            "movement_elapsed_ms",
            "movement_end_relative_ms",
        ]
    ].drop_duplicates("source_frame_index")
    centers = centers.merge(timing, on="source_frame_index", how="left", suffixes=("", "_timing"))
    centers["movement_elapsed_ms"] = centers["movement_elapsed_ms_timing"]
    centers["movement_end_relative_ms"] = centers["movement_end_relative_ms_timing"]
    centers = centers[
        centers["source_frame_index"].between(
            movement_window.movement_start_frame,
            movement_window.movement_end_frame,
            inclusive="both",
        )
    ].copy()
    roi_timeline = RoiTimeline(tuple(session.roi_keyframes))
    translation_camera = estimate_background_camera_motion(
        case.video_path,
        centers["source_frame_index"].astype(int).tolist(),
        roi_for_frame=roi_timeline.bbox_for_frame,
    )
    affine_camera = estimate_background_camera_motion_affine(
        case.video_path,
        centers["source_frame_index"].astype(int).tolist(),
        roi_for_frame=roi_timeline.bbox_for_frame,
    )
    reference_px = feature_summary["run_metadata"]["normalisation_reference"]["reference_value_px"]
    translation_candidate = compensate_projected_path(
        centers,
        translation_camera,
        scale_reference_px=float(reference_px),
    )
    affine_candidate = compensate_projected_path(
        centers,
        affine_camera,
        scale_reference_px=float(reference_px),
    )
    translation_validation = validate_projected_path(translation_candidate)
    affine_validation = validate_projected_path(affine_candidate)
    chosen_method, chosen_candidate, chosen_validation = _choose_path_candidate(
        translation_candidate,
        translation_validation,
        affine_candidate,
        affine_validation,
    )
    path_df = enforce_path_validation(chosen_candidate, chosen_validation)
    path_summary = path_quality_summary(path_df)
    path_summary["candidate_comparison"] = {
        "translation": translation_validation,
        "affine": affine_validation,
        "chosen_method": chosen_method,
    }
    bilateral_summary = compute_bilateral_hka_summary(dynamic_df)
    phase_status = _phase_status(args.movement_phases)
    observations = build_movement_observations(
        case_id=case.case_id,
        dynamic_df=dynamic_df,
        case_summary=case_summary,
        path_summary=path_summary,
        bilateral_summary=bilateral_summary,
    )
    description_payload = build_observable_movement_description_payload(
        case_id=case.case_id,
        source_id=case.source_id,
        dynamic_df=dynamic_df,
        frame_quality=frame_quality,
        path_summary=path_summary,
        movement_window=movement_window,
        phase_status=phase_status,
    )
    path_output = Path(args.path_output)
    path_quality_output = Path(args.path_quality_output)
    observation_output = Path(args.observation_output)
    observable_description_output = Path(args.observable_description_output)
    for path in (path_output, path_quality_output, observation_output, observable_description_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    path_df.to_parquet(path_output, index=False)
    path_quality_output.write_text(
        json.dumps(_json_ready(path_summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_observations_json(
        observations,
        observation_output,
        metadata={
            "case_id": case.case_id,
            "source_id": case.source_id,
            "movement_window": movement_window.to_dict(),
            "path_output": str(path_output),
            "path_quality_summary": str(path_quality_output),
            "path_candidate_choice": chosen_method,
            "bilateral_hka_summary": bilateral_summary.to_dict(),
            "semantic_version": "m5_10_semantic_movement_with_controlled_vocabulary_v1",
            "observable_description_output": str(observable_description_output),
        },
    )
    write_observable_descriptions_json(description_payload, observable_description_output)
    diagnostics_dir = Path(args.diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    translation_candidate.to_parquet(args.translation_path_output, index=False)
    affine_candidate.to_parquet(args.affine_path_output, index=False)
    translation_camera.to_csv(args.translation_camera_output, index=False)
    affine_camera.to_csv(args.affine_camera_output, index=False)
    roi_records = _roi_records(roi_timeline, frame_quality["source_frame_index"].astype(int).tolist())
    path_diagnostics = build_path_frame_diagnostics(
        frame_quality=frame_quality,
        center_df=centers,
        roi_records=roi_records,
        translation_camera=translation_camera,
        affine_camera=affine_camera,
        translation_path=translation_candidate,
        affine_path=affine_candidate,
        final_path=path_df,
    )
    path_diagnostics.to_csv(args.path_diagnostics_output, index=False)
    plot_path_validation_comparison(
        centers,
        translation_candidate,
        affine_candidate,
        diagnostics_dir / f"{args.prefix}_path_validation_comparison.png",
    )
    plot_projected_movement_path(
        path_df,
        diagnostics_dir / f"{args.prefix}_semantic_projected_movement_path.png",
    )
    plot_projected_speed(
        path_df,
        diagnostics_dir / f"{args.prefix}_semantic_projected_speed_pattern.png",
    )
    plot_projected_heading(
        path_df,
        diagnostics_dir / f"{args.prefix}_semantic_projected_direction_change.png",
    )
    plot_bilateral_hka_relationship(
        dynamic_df,
        diagnostics_dir / f"{args.prefix}_semantic_bilateral_hka_relationship.png",
    )
    plot_semantic_summary(
        observations,
        diagnostics_dir / f"{args.prefix}_semantic_movement_summary.png",
    )
    print(f"Wrote projected movement path to {path_output}")
    print(f"Wrote path quality summary to {path_quality_output}")
    print(f"Wrote path diagnostics to {args.path_diagnostics_output}")
    print(f"Wrote semantic observations to {observation_output}")
    print(f"Wrote observable movement descriptions to {observable_description_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-slug", default="christen_press")
    parser.add_argument("--annotation-session", default="data/annotations/human/christen_press_annotation_session_human.json")
    parser.add_argument("--movement-window", default="data/annotations/human/christen_press_movement_window_human.json")
    parser.add_argument("--processed-pose", default="data/processed/human/christen_press_processed_pose.parquet")
    parser.add_argument("--frame-quality", default="data/quality/human/christen_press_frame_quality.csv")
    parser.add_argument("--dynamic-features", default="data/dynamics/human/christen_press_dynamic_features.parquet")
    parser.add_argument("--case-feature-summary", default="data/analytics/human/christen_press_case_feature_summary.parquet")
    parser.add_argument("--feature-summary", default="data/quality/human/christen_press_feature_summary.json")
    parser.add_argument("--path-output", default="data/path/human/christen_press_projected_movement_path.parquet")
    parser.add_argument(
        "--translation-path-output",
        default="data/path/human/christen_press_projected_movement_path_translation_candidate.parquet",
    )
    parser.add_argument(
        "--affine-path-output",
        default="data/path/human/christen_press_projected_movement_path_affine_candidate.parquet",
    )
    parser.add_argument(
        "--translation-camera-output",
        default="data/quality/human/christen_press_camera_motion_translation.csv",
    )
    parser.add_argument(
        "--affine-camera-output",
        default="data/quality/human/christen_press_camera_motion_affine.csv",
    )
    parser.add_argument(
        "--path-diagnostics-output",
        default="data/quality/human/christen_press_path_frame_diagnostics.csv",
    )
    parser.add_argument("--path-quality-output", default="data/quality/human/christen_press_path_quality_summary.json")
    parser.add_argument("--observation-output", default="data/semantic/human/christen_press_movement_observations.json")
    parser.add_argument(
        "--observable-description-output",
        default="data/semantics/human/christen_press_observable_movement_descriptions.json",
    )
    parser.add_argument("--movement-phases", default="data/phases/human/christen_press_movement_phases.json")
    parser.add_argument("--diagnostics-dir", default="data/diagnostics/human")
    parser.add_argument("--prefix", default="christen_press_human")
    return parser.parse_args()


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


def _choose_path_candidate(
    translation_candidate: pd.DataFrame,
    translation_validation: dict,
    affine_candidate: pd.DataFrame,
    affine_validation: dict,
) -> tuple[str, pd.DataFrame, dict]:
    if translation_validation.get("validation_status") == "SUPPORTED":
        return "translation", translation_candidate, translation_validation
    if affine_validation.get("validation_status") == "SUPPORTED":
        return "affine", affine_candidate, affine_validation
    if _supported_count(affine_candidate) > _supported_count(translation_candidate):
        return "affine_diagnostic_unavailable", affine_candidate, affine_validation
    return "translation_diagnostic_unavailable", translation_candidate, translation_validation


def _supported_count(df: pd.DataFrame) -> int:
    if df.empty or "path_status" not in df.columns:
        return 0
    return int(df["path_status"].eq("SUPPORTED").sum())


def _phase_status(path: str) -> str:
    phase_path = Path(path)
    if not phase_path.exists():
        return "UNAVAILABLE"
    try:
        payload = json.loads(phase_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "UNAVAILABLE"
    return str(payload.get("status", "UNAVAILABLE"))


def _roi_records(roi_timeline: RoiTimeline, frames: list[int]) -> pd.DataFrame:
    records = []
    for frame in sorted({int(item) for item in frames}):
        bbox = roi_timeline.bbox_for_frame(frame)
        records.append(
            {
                "source_frame_index": frame,
                "roi_center_x": bbox.x + bbox.width / 2.0,
                "roi_center_y": bbox.y + bbox.height / 2.0,
                "roi_x": bbox.x,
                "roi_y": bbox.y,
                "roi_width": bbox.width,
                "roi_height": bbox.height,
            }
        )
    return pd.DataFrame(records)


def _ensure_movement_timing(dynamic_df: pd.DataFrame, movement_window) -> pd.DataFrame:
    output = dynamic_df.copy()
    if {"movement_elapsed_ms", "movement_end_relative_ms"}.issubset(output.columns):
        return output
    start_ms = float(movement_window.movement_start_timestamp_ms)
    end_ms = float(movement_window.movement_end_timestamp_ms)
    output["movement_elapsed_ms"] = output["timestamp_ms"].astype(float) - start_ms
    output["movement_end_relative_ms"] = output["timestamp_ms"].astype(float) - end_ms
    return output


if __name__ == "__main__":
    raise SystemExit(main())
