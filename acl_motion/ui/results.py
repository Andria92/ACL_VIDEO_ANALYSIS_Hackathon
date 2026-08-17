"""First human-results experience for ACL Movement Explorer."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from acl_motion.analytics.guard import CROSS_CASE_ANALYTIC_NAMES
from acl_motion.annotations.models import AnnotationCase, MovementWindowAnnotation
from acl_motion.annotations.movement_window import movement_window_to_event_annotation
from acl_motion.annotations.propagation import propagated_bbox
from acl_motion.annotations.storage import (
    human_annotation_paths,
    load_human_annotation_session,
    load_movement_window_json,
    save_human_annotation_session,
)
from acl_motion.annotations.view_alignment import load_view_alignment
from acl_motion.segmentation.target_mask import (
    MaskPrompt,
    append_mask_prompt,
    clear_mask_prompts,
    draw_mask_prompt_overlay,
    draw_target_mask_overlay,
    load_mask_prompts,
    mask_prompt_path,
    pop_mask_prompt,
    target_mask_for_frame,
)
from acl_motion.semantics.metric_explorer import build_metric_explorer_payload
from acl_motion.semantics.visual_story import build_movement_visual_story
from acl_motion.video.io import read_video_metadata
from acl_motion.video.roi import BBox
from acl_motion.visualisation.overlay import draw_pose_overlay

HUMAN_DATA_ROOT = Path("data")

SEMANTIC_CATEGORY_LABELS = {
    "movement_path": "Movement Path",
    "hip_knee_ankle_chain": "Hip-Knee-Ankle Chain",
    "hip_thigh": "Hip / Thigh",
    "trunk_pelvis": "Trunk & Pelvis",
    "upper_body": "Upper Body",
    "bilateral_limb_relationship": "Bilateral Limb Relationship",
    "movement_timing": "Movement Timing",
    "evidence": "Evidence",
}

RESULT_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "whole_body": (
        "injured_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg",
        "right_hka_angle_2d_deg",
        "projected_trunk_axis_angle_deg",
        "right_elbow_angle_2d_deg",
        "hka_projected_bilateral_difference_deg",
    ),
    "lower_limb": (
        "injured_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg",
        "left_hka_angle_2d_deg",
        "right_hka_angle_2d_deg",
        "left_knee_line_deviation_normalized",
        "right_knee_line_deviation_normalized",
        "hka_projected_bilateral_difference_deg",
        "left_knee_ankle_x_offset_normalized",
        "right_knee_ankle_x_offset_normalized",
        "left_knee_ankle_distance_normalized",
        "right_knee_ankle_distance_normalized",
    ),
    "trunk_pelvis": (
        "projected_trunk_axis_angle_deg",
        "projected_hip_line_angle_deg",
        "projected_shoulder_line_angle_deg",
        "projected_shoulder_pelvis_orientation_difference_deg",
    ),
    "upper_body": (
        "injured_elbow_angle_2d_deg",
        "contralateral_elbow_angle_2d_deg",
        "left_elbow_angle_2d_deg",
        "right_elbow_angle_2d_deg",
        "left_upper_arm_orientation_2d_deg",
        "right_upper_arm_orientation_2d_deg",
        "left_wrist_pelvis_x_offset_normalized",
        "right_wrist_pelvis_x_offset_normalized",
        "left_wrist_pelvis_distance_normalized",
        "right_wrist_pelvis_distance_normalized",
    ),
    "bilateral": (
        "hka_projected_bilateral_difference_deg",
        "hka_projected_bilateral_absolute_difference_deg",
        "elbow_projected_bilateral_difference_deg",
        "elbow_projected_bilateral_absolute_difference_deg",
        "knee_line_deviation_bilateral_difference",
        "knee_line_deviation_normalized_bilateral_difference",
    ),
}

FEATURE_LABELS = {
    "left_hka_angle_2d_deg": "Left projected HKA",
    "right_hka_angle_2d_deg": "Right projected HKA",
    "injured_hka_angle_2d_deg": "Injured projected HKA",
    "contralateral_hka_angle_2d_deg": "Contralateral projected HKA",
    "hka_projected_bilateral_difference_deg": "Projected bilateral HKA difference",
    "hka_projected_bilateral_absolute_difference_deg": "Projected bilateral HKA absolute difference",
    "left_knee_ankle_x_offset_normalized": "Left knee-ankle x offset",
    "right_knee_ankle_x_offset_normalized": "Right knee-ankle x offset",
    "left_knee_line_deviation_normalized": "Left knee-line deviation",
    "right_knee_line_deviation_normalized": "Right knee-line deviation",
    "left_knee_ankle_distance_normalized": "Left projected segment-length diagnostic",
    "right_knee_ankle_distance_normalized": "Right projected segment-length diagnostic",
    "projected_trunk_axis_angle_deg": "Projected trunk-axis orientation",
    "projected_hip_line_angle_deg": "Projected hip-line orientation",
    "projected_shoulder_line_angle_deg": "Projected shoulder-line orientation",
    "projected_shoulder_pelvis_orientation_difference_deg": (
        "Shoulder-pelvis orientation difference"
    ),
    "left_elbow_angle_2d_deg": "Left elbow angle",
    "right_elbow_angle_2d_deg": "Right elbow angle",
    "injured_elbow_angle_2d_deg": "Injured elbow angle",
    "contralateral_elbow_angle_2d_deg": "Contralateral elbow angle",
    "left_upper_arm_orientation_2d_deg": "Left upper-arm orientation",
    "right_upper_arm_orientation_2d_deg": "Right upper-arm orientation",
    "left_wrist_pelvis_x_offset_normalized": "Left wrist-pelvis x offset",
    "right_wrist_pelvis_x_offset_normalized": "Right wrist-pelvis x offset",
    "left_wrist_pelvis_distance_normalized": "Left wrist-pelvis distance",
    "right_wrist_pelvis_distance_normalized": "Right wrist-pelvis distance",
    "elbow_projected_bilateral_difference_deg": "Projected bilateral elbow difference",
    "elbow_projected_bilateral_absolute_difference_deg": (
        "Projected bilateral elbow absolute difference"
    ),
    "knee_line_deviation_bilateral_difference": "Projected bilateral knee-line difference",
    "knee_line_deviation_normalized_bilateral_difference": (
        "Projected bilateral knee-line difference, normalized"
    ),
}

STATUS_EXPLANATIONS = {
    "SUPPORTED": "Supported by the current quality rules.",
    "LIMITED": "Partially supported; inspect the evidence before using it.",
    "UNAVAILABLE": "Unavailable under the current quality rules.",
    "VALID_TARGET": "The target athlete evidence is defensible for this frame.",
    "TARGET_IDENTITY_UNCERTAIN": "The target athlete evidence is uncertain in this frame.",
    "TARGET_NOT_FOUND": "The target athlete was not found in this frame.",
    "INVALID_TRACK_SEGMENT": "This frame is outside a valid continuous target segment.",
    "LOW_POSE_CONFIDENCE": "Pose confidence was below the quality threshold.",
    "LOW_CONFIDENCE": "One or more required landmarks had low confidence.",
    "INVALID_TARGET_FRAME": "This frame was not valid enough for this feature.",
    "MISSING_FEATURE": "The feature value is unavailable at this frame.",
    "LOW_DYNAMIC_CONFIDENCE": "The local trajectory was not stable enough for robust dynamics.",
    "INSUFFICIENT_NEIGHBORHOOD": "There were not enough stable neighboring samples.",
    "TEMPORAL_OUTLIER": "The local trajectory was inconsistent with nearby samples.",
    "NOT_DYNAMIC_FEATURE": "This descriptor is not treated as a dynamic feature.",
}


def human_results_available(case: AnnotationCase, data_root: str | Path = HUMAN_DATA_ROOT) -> bool:
    """Return whether a human Movement Profile exists for the case."""

    return _results_paths(case.slug, Path(data_root))["movement_profile"].exists()


def load_human_results_payload(
    case: AnnotationCase,
    *,
    data_root: str | Path = HUMAN_DATA_ROOT,
    case_views: tuple[AnnotationCase, ...] | None = None,
) -> dict:
    """Load the user-facing HUMAN results payload for one completed case."""

    root = Path(data_root)
    view_cases = _normalise_case_views(case, case_views)
    paths = _results_paths(case.slug, root)
    _require_human_paths(paths)
    session = load_human_annotation_session(paths["annotation_session"])
    movement_window = load_movement_window_json(paths["movement_window"])
    movement_profile = _read_json(paths["movement_profile"])
    evidence_profile = _read_json(paths["evidence_profile"])
    reliability = _read_json(paths["reliability_summary"])
    dynamic_quality = _read_json(paths["dynamic_quality_summary"])
    path_quality = _read_json(paths["path_quality_summary"])
    semantic_observations = _read_json(paths["semantic_observations"])
    observable_descriptions = _read_json(paths["observable_descriptions"])
    movement_story = _read_json(paths["movement_phases"])
    case_summary = pd.read_parquet(paths["case_feature_summary"])
    dynamic_df = pd.read_parquet(paths["dynamic_features"])
    path_df = pd.read_parquet(paths["path_features"])
    processed_pose = pd.read_parquet(paths["processed_pose"])
    phase_frame_map = pd.read_parquet(paths["phase_frame_map"])
    metric_explorer = build_metric_explorer_payload(
        dynamic_df=dynamic_df,
        path_df=path_df,
        movement_story=movement_story,
    )
    selected_features = _selected_existing_features(dynamic_df)
    feature_profiles = {
        item["feature_name"]: item for item in movement_profile.get("trajectory_summaries", [])
    }
    frames = _frame_index(dynamic_df)
    payload = {
        "case": {
            "slug": case.slug,
            "player_name": case.player_name,
            "case_id": case.case_id,
            "source_id": case.source_id,
            "view_id": case.view_id or case.source_id,
            "view_label": case.view_label,
            "subtitle": "Human-annotated movement analysis",
        },
        "view": _view_payload(case, root),
        "case_views": _case_views_payload(case, view_cases, root),
        "case_synthesis": build_case_synthesis_payload(case, view_cases, data_root=root),
        "target_annotation": {
            "label": "Human verified",
            "human_target_verified": True,
            "manual_roi_keyframes": session.manual_roi_keyframe_count,
            "annotation_session_id": session.provenance.annotation_session_id,
            "annotator_id": session.provenance.annotator_id,
        },
        "target_segmentation": _target_segmentation_payload(case.slug, root),
        "movement_window": movement_window.to_dict(),
        "header_metrics": _header_metrics(session, movement_window),
        "summary_statement": (
            "The movement window represents the observable sequence selected by the "
            "researcher. The exact timing of ACL rupture is not inferred."
        ),
        "evidence_dimensions": _evidence_dimensions(
            evidence_profile,
            reliability,
            dynamic_quality,
            case_summary,
        ),
        "body_region_evidence": _body_region_evidence(case_summary),
        "semantic_category_labels": SEMANTIC_CATEGORY_LABELS,
        "semantic_observations": semantic_observations["observations"],
        "semantic_categories": semantic_observations["categories"],
        "observable_movement_descriptions": observable_descriptions,
        "movement_story": movement_story,
        "movement_visual_story": build_movement_visual_story(
            movement_story=movement_story,
            metric_explorer=metric_explorer,
            processed_pose=processed_pose,
            laterality_mapping=_laterality_mapping(movement_profile, dynamic_df),
        ),
        "phase_frame_map": _phase_frame_map_payload(phase_frame_map),
        "metric_explorer": metric_explorer,
        "path_quality_summary": path_quality,
        "feature_groups": _feature_groups(selected_features),
        "feature_cards": _feature_cards(case_summary, feature_profiles, selected_features),
        "trajectories": _trajectories(dynamic_df, selected_features),
        "frames": frames,
        "quality_limitations": _quality_limitations(reliability, dynamic_quality, dynamic_df),
        "cross_case_analytics": {
            name: {
                "available": False,
                "label": _cross_case_label(name),
                "reason": "Additional human-validated cases are required.",
            }
            for name in CROSS_CASE_ANALYTIC_NAMES
        },
        "source_files": {key: str(value) for key, value in paths.items()},
    }
    return _json_ready(payload)


def load_result_evidence_payload(
    case: AnnotationCase,
    *,
    feature_name: str,
    source_frame_index: int,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> dict:
    """Load exact feature/frame traceability for the Results view."""

    paths = _results_paths(case.slug, Path(data_root))
    _require_human_paths(paths)
    dynamic_df = pd.read_parquet(paths["dynamic_features"])
    processed_pose = pd.read_parquet(paths["processed_pose"])
    landmark_quality = pd.read_parquet(paths["landmark_quality"])
    frame_quality = pd.read_csv(paths["frame_quality"])
    rows = dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["source_frame_index"].astype(int).eq(int(source_frame_index))
    ]
    if rows.empty:
        raise ValueError(
            f"No exact trace point for feature {feature_name} at source frame {source_frame_index}."
        )
    row = rows.iloc[0]
    landmarks = _listify(row.get("landmarks_used"))
    frame_rows = frame_quality[
        frame_quality["source_frame_index"].astype(int).eq(int(source_frame_index))
    ]
    landmark_rows = landmark_quality[
        landmark_quality["source_frame_index"].astype(int).eq(int(source_frame_index))
        & landmark_quality["landmark_name"].isin(landmarks)
    ]
    processed_rows = processed_pose[
        processed_pose["source_frame_index"].astype(int).eq(int(source_frame_index))
        & processed_pose["landmark_name"].isin(landmarks)
    ]
    return _json_ready(
        {
            "feature_name": feature_name,
            "display_label": feature_display_label(feature_name),
            "source_frame_index": int(row["source_frame_index"]),
            "analysis_frame_index": int(row["analysis_frame_index"]),
            "timestamp_ms": _optional_float(row["timestamp_ms"]),
            "movement_elapsed_ms": _optional_float(row.get("movement_elapsed_ms")),
            "movement_end_relative_ms": _optional_float(row.get("movement_end_relative_ms")),
            "feature_value": _optional_float(row["feature_value"]),
            "unit": str(row.get("unit", "")),
            "landmarks_used": landmarks,
            "frame_qc": _row_dict(frame_rows),
            "landmark_qc": [_series_dict(item) for _, item in landmark_rows.iterrows()],
            "processed_landmarks": [_series_dict(item) for _, item in processed_rows.iterrows()],
            "input_provenance": {
                "observed_or_interpolated": (
                    "interpolated" if bool(row.get("input_interpolated")) else "observed"
                ),
                "smoothed": bool(row.get("input_smoothed")),
            },
            "feature_status": str(row.get("feature_status", "")),
            "feature_status_text": explain_status(str(row.get("feature_status", ""))),
            "frame_status": str(row.get("frame_status", "")),
            "frame_status_text": explain_status(str(row.get("frame_status", ""))),
            "dynamic_status": str(row.get("dynamic_status", "")),
            "dynamic_status_text": explain_status(str(row.get("dynamic_status", ""))),
            "dynamic_rejection_reason": _clean_reason(row.get("dynamic_rejection_reason")),
            "rejection_reason": _clean_reason(row.get("rejection_reason")),
            "advanced": {
                "raw_first_difference_rate": _optional_float(row.get("raw_first_difference_rate")),
                "raw_dynamic_status": str(row.get("raw_dynamic_status", "")),
                "robust_dynamic_rate": _optional_float(row.get("robust_dynamic_rate")),
                "dynamic_quality": str(row.get("dynamic_quality", "")),
                "local_residual": _optional_float(row.get("local_residual")),
                "local_jitter_metric": _optional_float(row.get("local_jitter_metric")),
                "temporal_stability_score": _optional_float(row.get("temporal_stability_score")),
            },
        }
    )


def read_result_frame_jpeg(
    case: AnnotationCase,
    *,
    source_frame_index: int,
    show_roi: bool,
    show_pose: bool,
    show_mask: bool = False,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> bytes:
    """Read a source frame and optionally render human ROI and processed pose."""

    import cv2

    paths = _results_paths(case.slug, Path(data_root))
    _require_human_paths(paths)
    capture = cv2.VideoCapture(str(case.video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {case.video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_index))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {source_frame_index} from {case.video_path}")
        target_bbox = _bbox_for_result_frame(paths, source_frame_index)
        if show_mask:
            prompts = load_mask_prompts(mask_prompt_path(data_root, case.slug))
            mask = target_mask_for_frame(
                frame,
                bbox=target_bbox,
                prompts=prompts,
                frame_index=int(source_frame_index),
            )
            frame = draw_target_mask_overlay(frame, mask)
            frame = draw_mask_prompt_overlay(
                frame,
                prompts,
                frame_index=int(source_frame_index),
            )
        bbox = target_bbox if show_roi else None
        landmarks = _landmarks_for_result_frame(paths, source_frame_index) if show_pose else {}
        output = draw_pose_overlay(
            frame,
            landmarks,
            bbox=bbox,
            frame_label=f"source frame {source_frame_index}",
            confidence_threshold=0.0,
        )
        encoded_ok, buffer = cv2.imencode(".jpg", output)
        if not encoded_ok:
            raise ValueError("Could not encode results frame as JPEG.")
        return buffer.tobytes()
    finally:
        capture.release()


def save_result_mask_prompt(
    case: AnnotationCase,
    *,
    source_frame_index: int,
    x_px: float,
    y_px: float,
    label: str,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> dict:
    """Save one human target/non-target point prompt for visible-region refinement."""

    path = mask_prompt_path(data_root, case.slug)
    payload = append_mask_prompt(
        path,
        MaskPrompt(
            frame_index=int(source_frame_index),
            x_px=float(x_px),
            y_px=float(y_px),
            label=label,
        ),
    )
    return _json_ready({"prompt_file": str(path), **payload})


def undo_result_mask_prompt(
    case: AnnotationCase,
    *,
    source_frame_index: int,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> dict:
    """Remove the latest human target-region prompt for one frame."""

    path = mask_prompt_path(data_root, case.slug)
    payload = pop_mask_prompt(path, frame_index=int(source_frame_index))
    return _json_ready({"prompt_file": str(path), **payload})


def clear_result_mask_prompts(
    case: AnnotationCase,
    *,
    source_frame_index: int,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> dict:
    """Clear human target-region prompts for one frame."""

    path = mask_prompt_path(data_root, case.slug)
    payload = clear_mask_prompts(path, frame_index=int(source_frame_index))
    return _json_ready({"prompt_file": str(path), **payload})


def trim_human_analysis_window_and_regenerate(
    case: AnnotationCase,
    *,
    movement_end_frame: int,
    rationale: str,
    annotator_id: str = "researcher_01",
    data_root: str | Path = HUMAN_DATA_ROOT,
    python_executable: str | Path = sys.executable,
) -> dict:
    """Shorten a human analysis window and rebuild the human Results bundle."""

    root = Path(data_root)
    paths = _results_paths(case.slug, root)
    annotation_paths = human_annotation_paths(root / "annotations" / "human", case.slug)
    session = load_human_annotation_session(annotation_paths.session_json)
    if not session.roi_keyframes:
        raise ValueError("Cannot trim analysis window without human ROI keyframes.")

    metadata = read_video_metadata(case.video_path)
    start_frame = int(session.roi_keyframes[0].frame_index)
    end_frame = int(movement_end_frame)
    if end_frame < start_frame:
        raise ValueError("Movement End must be at or after Movement Start.")
    if end_frame >= metadata.frame_count:
        raise ValueError("Movement End is outside the source video.")

    previous_end = (
        session.movement_window.movement_end_frame
        if session.movement_window is not None
        else None
    )
    timestamp_ms = end_frame / metadata.fps * 1000 if metadata.fps else 0.0
    movement_window = MovementWindowAnnotation(
        movement_start_frame=start_frame,
        movement_start_timestamp_ms=start_frame / metadata.fps * 1000 if metadata.fps else 0.0,
        movement_end_frame=end_frame,
        movement_end_timestamp_ms=timestamp_ms,
        confidence=session.event_confidence_label,
        rationale=rationale,
        source="human_results_analysis_boundary",
    )
    event_annotation = movement_window_to_event_annotation(session, movement_window)
    boundary_note = _analysis_boundary_note(previous_end, end_frame, rationale)
    updated_session = session.with_changes(
        movement_window=movement_window,
        event_annotation=event_annotation,
        notes=_append_note(session.notes, boundary_note),
        finalized=True,
    )
    save_human_annotation_session(updated_session, root / "annotations" / "human", case.slug)
    decision_path = _append_analysis_boundary_decision(
        root,
        case,
        previous_end=previous_end,
        movement_end_frame=end_frame,
        rationale=rationale,
        annotator_id=annotator_id,
    )

    commands = build_human_analysis_regeneration_commands(
        case,
        movement_start_frame=start_frame,
        movement_end_frame=end_frame,
        data_root=root,
        python_executable=python_executable,
    )
    command_results = [_run_regeneration_command(command) for command in commands]
    return {
        "regenerated": True,
        "case": case.to_dict(),
        "movement_window": movement_window.to_dict(),
        "previous_movement_end_frame": previous_end,
        "analysis_boundary_decision": str(decision_path),
        "result_url": f"/results?case={case.slug}",
        "outputs": {key: str(value) for key, value in paths.items() if value.exists()},
        "commands": command_results,
    }


def build_human_analysis_regeneration_commands(
    case: AnnotationCase,
    *,
    movement_start_frame: int,
    movement_end_frame: int,
    data_root: str | Path = HUMAN_DATA_ROOT,
    python_executable: str | Path = sys.executable,
) -> list[list[str]]:
    """Return the existing script commands needed to rebuild one human analysis."""

    root = Path(data_root)
    slug = case.slug
    prefix = slug
    annotation_root = root / "annotations" / "human"
    roi = annotation_root / f"{slug}_target_roi_human.csv"
    session = annotation_root / f"{slug}_annotation_session_human.json"
    movement_window = annotation_root / f"{slug}_movement_window_human.json"
    event = annotation_root / f"{slug}_event_annotation_human.json"
    pose = root / "pose" / "human" / f"{slug}_raw_pose.parquet"
    processed = root / "processed" / "human" / f"{slug}_processed_pose.parquet"
    features = root / "features" / "human" / f"{slug}_framewise_geometry.parquet"
    dynamic = root / "dynamics" / "human" / f"{slug}_dynamic_features.parquet"
    case_summary = root / "analytics" / "human" / f"{slug}_case_feature_summary.parquet"
    path_features = root / "path" / "human" / f"{slug}_projected_movement_path.parquet"
    feature_summary = root / "quality" / "human" / f"{slug}_feature_summary.json"
    frame_quality = root / "quality" / "human" / f"{slug}_frame_quality.csv"
    annotated_injured_side = str(case.injured_side.value)
    injured_side = (
        annotated_injured_side
        if annotated_injured_side in {"left", "right"}
        else _previous_injured_side(feature_summary)
    )
    exe = str(python_executable)
    return [
        [
            exe,
            "scripts/extract_pose.py",
            "--video",
            str(case.video_path),
            "--backend",
            "yolo",
            "--model-path",
            str(root / "models" / "yolov8n-pose.pt"),
            "--yolo-selection-strategy",
            "center",
            "--roi-keyframes",
            str(roi),
            "--case-id",
            case.case_id,
            "--source-id",
            case.source_id,
            "--start-frame",
            str(movement_start_frame),
            "--end-frame",
            str(movement_end_frame),
            "--output",
            str(pose),
            "--metadata-output",
            str(root / "pose" / "human" / f"{slug}_raw_pose.metadata.json"),
        ],
        [
            exe,
            "scripts/process_pose_quality.py",
            "--raw-pose",
            str(pose),
            "--frame-quality-output",
            str(frame_quality),
            "--landmark-quality-output",
            str(root / "quality" / "human" / f"{slug}_landmark_quality.parquet"),
            "--processed-output",
            str(processed),
            "--summary-output",
            str(root / "quality" / "human" / f"{slug}_reliability_summary.json"),
            "--target-identity-output",
            str(root / "quality" / "human" / f"{slug}_target_identity_diagnostics.csv"),
            "--raw-clean-plot",
            str(root / "diagnostics" / "human" / f"{prefix}_raw_clean_smoothed.png"),
            "--availability-plot",
            str(root / "diagnostics" / "human" / f"{prefix}_availability_timeline.png"),
        ],
        [
            exe,
            "scripts/compute_geometry_features.py",
            "--processed-pose",
            str(processed),
            "--injured-side",
            injured_side,
            "--output",
            str(features),
            "--completeness-output",
            str(root / "quality" / "human" / f"{slug}_feature_completeness.csv"),
            "--summary-output",
            str(feature_summary),
            "--metadata-output",
            str(root / "features" / "human" / f"{slug}_framewise_geometry.metadata.json"),
        ],
        [
            exe,
            "scripts/plot_pose_diagnostics.py",
            "--pose",
            str(pose),
            "--output",
            str(root / "diagnostics" / "human" / f"{prefix}_joint_trajectories.png"),
        ],
        [
            exe,
            "scripts/plot_geometry_diagnostics.py",
            "--features",
            str(features),
            "--hka-output",
            str(root / "diagnostics" / "human" / f"{prefix}_hka_trajectories.png"),
            "--hka-difference-output",
            str(root / "diagnostics" / "human" / f"{prefix}_hka_bilateral_difference.png"),
            "--trunk-pelvis-output",
            str(root / "diagnostics" / "human" / f"{prefix}_trunk_pelvis_profile.png"),
            "--upper-limb-output",
            str(root / "diagnostics" / "human" / f"{prefix}_upper_limb_profile.png"),
            "--availability-output",
            str(root / "diagnostics" / "human" / f"{prefix}_feature_availability.png"),
        ],
        [
            exe,
            "scripts/compute_movement_window_features.py",
            "--feature-input",
            str(features),
            "--session",
            str(session),
            "--movement-window",
            str(movement_window),
            "--output",
            str(root / "events" / "human" / f"{slug}_movement_relative_features.parquet"),
            "--window-output",
            str(root / "summaries" / "human" / f"{slug}_movement_window_summaries.parquet"),
            "--summary-output",
            str(root / "summaries" / "human" / f"{slug}_movement_summary.json"),
        ],
        [
            exe,
            "scripts/harden_dynamic_reliability.py",
            "--event-features",
            str(root / "events" / "human" / f"{slug}_movement_relative_features.parquet"),
            "--event-annotation",
            str(event),
            "--output",
            str(dynamic),
            "--quality-output",
            str(root / "quality" / "human" / f"{slug}_dynamic_quality_summary.json"),
            "--spike-audit-output",
            str(root / "quality" / "human" / f"{slug}_dynamic_spike_audit.csv"),
            "--window-output",
            str(root / "summaries" / "human" / f"{slug}_dynamic_window_summaries.parquet"),
        ],
        [
            exe,
            "scripts/build_movement_profile.py",
            "--dynamic-features",
            str(dynamic),
            "--event-annotation",
            str(event),
            "--pose-reliability",
            str(root / "quality" / "human" / f"{slug}_reliability_summary.json"),
            "--dynamic-quality",
            str(root / "quality" / "human" / f"{slug}_dynamic_quality_summary.json"),
            "--geometry-summary",
            str(feature_summary),
            "--movement-window",
            str(movement_window),
            "--roi-keyframes",
            str(roi),
            "--profile-output",
            str(root / "profiles" / "human" / f"{slug}_movement_profile.json"),
            "--evidence-output",
            str(root / "quality" / "human" / f"{slug}_evidence_profile.json"),
            "--case-feature-output",
            str(case_summary),
            "--diagnostics-dir",
            str(root / "diagnostics" / "human"),
            "--prefix",
            prefix,
        ],
        _semantic_command(
            exe,
            case,
            root,
            prefix,
            session,
            movement_window,
            processed,
            frame_quality,
            dynamic,
            case_summary,
            feature_summary,
        ),
        [
            exe,
            "scripts/compute_movement_phases.py",
            "--case-slug",
            slug,
            "--movement-window",
            str(movement_window),
            "--dynamic-features",
            str(dynamic),
            "--case-feature-summary",
            str(case_summary),
            "--path-features",
            str(path_features),
            "--phase-output",
            str(root / "phases" / "human" / f"{slug}_movement_phases.json"),
            "--frame-map-output",
            str(root / "phases" / "human" / f"{slug}_phase_frame_map.parquet"),
            "--change-score-output",
            str(root / "phases" / "human" / f"{slug}_movement_change_score.parquet"),
            "--transition-output",
            str(root / "phases" / "human" / f"{slug}_phase_transitions.csv"),
            "--diagnostics-dir",
            str(root / "diagnostics" / "human"),
            "--prefix",
            prefix,
        ],
        _semantic_command(
            exe,
            case,
            root,
            prefix,
            session,
            movement_window,
            processed,
            frame_quality,
            dynamic,
            case_summary,
            feature_summary,
        ),
        [
            exe,
            "scripts/render_qc_overlay.py",
            "--video",
            str(case.video_path),
            "--processed-pose",
            str(processed),
            "--frame-quality",
            str(frame_quality),
            "--output",
            str(root / "diagnostics" / "human" / f"{prefix}_qc_overlay.mp4"),
            "--start-frame",
            str(movement_start_frame),
            "--end-frame",
            str(movement_end_frame),
        ],
        [
            exe,
            "scripts/plot_human_movement_diagnostics.py",
            "--session",
            str(session),
            "--movement-window",
            str(movement_window),
            "--dynamic-features",
            str(dynamic),
            "--output-dir",
            str(root / "diagnostics" / "human"),
            "--prefix",
            prefix,
        ],
    ]


def _semantic_command(
    exe: str,
    case: AnnotationCase,
    root: Path,
    prefix: str,
    session: Path,
    movement_window: Path,
    processed: Path,
    frame_quality: Path,
    dynamic: Path,
    case_summary: Path,
    feature_summary: Path,
) -> list[str]:
    slug = case.slug
    return [
        exe,
        "scripts/compute_semantic_movement.py",
        "--case-slug",
        slug,
        "--annotation-session",
        str(session),
        "--movement-window",
        str(movement_window),
        "--processed-pose",
        str(processed),
        "--frame-quality",
        str(frame_quality),
        "--dynamic-features",
        str(dynamic),
        "--case-feature-summary",
        str(case_summary),
        "--feature-summary",
        str(feature_summary),
        "--path-output",
        str(root / "path" / "human" / f"{slug}_projected_movement_path.parquet"),
        "--translation-path-output",
        str(root / "path" / "human" / f"{slug}_projected_movement_path_translation_candidate.parquet"),
        "--affine-path-output",
        str(root / "path" / "human" / f"{slug}_projected_movement_path_affine_candidate.parquet"),
        "--translation-camera-output",
        str(root / "quality" / "human" / f"{slug}_camera_motion_translation.csv"),
        "--affine-camera-output",
        str(root / "quality" / "human" / f"{slug}_camera_motion_affine.csv"),
        "--path-diagnostics-output",
        str(root / "quality" / "human" / f"{slug}_path_frame_diagnostics.csv"),
        "--path-quality-output",
        str(root / "quality" / "human" / f"{slug}_path_quality_summary.json"),
        "--observation-output",
        str(root / "semantic" / "human" / f"{slug}_movement_observations.json"),
        "--observable-description-output",
        str(root / "semantics" / "human" / f"{slug}_observable_movement_descriptions.json"),
        "--movement-phases",
        str(root / "phases" / "human" / f"{slug}_movement_phases.json"),
        "--diagnostics-dir",
        str(root / "diagnostics" / "human"),
        "--prefix",
        prefix,
    ]


def _run_regeneration_command(command: list[str]) -> dict:
    env = os.environ.copy()
    mpl_cache = Path("/private/tmp/acl_movement_explorer_mpl_cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    command_label = " ".join(command[:2])
    if result.returncode != 0:
        raise ValueError(
            f"Analysis regeneration failed during {command_label}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return {
        "command": command_label,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-3:],
    }


def _append_analysis_boundary_decision(
    root: Path,
    case: AnnotationCase,
    *,
    previous_end: int | None,
    movement_end_frame: int,
    rationale: str,
    annotator_id: str,
) -> Path:
    path = root / "annotations" / "human" / f"{case.slug}_analysis_boundary_decisions_human.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"decisions": []}
    payload.setdefault("decisions", []).append(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "case_id": case.case_id,
            "source_id": case.source_id,
            "view_id": case.view_id or case.source_id,
            "previous_movement_end_frame": previous_end,
            "movement_end_frame": movement_end_frame,
            "rationale": rationale,
            "annotator_id": annotator_id,
            "source": "results_ui_end_analysis_here",
            "preserves_source_video": True,
        }
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _analysis_boundary_note(previous_end: int | None, movement_end_frame: int, rationale: str) -> str:
    previous = "-" if previous_end is None else str(previous_end)
    return (
        f"[{datetime.now(UTC).isoformat()}] Analysis window changed from end frame "
        f"{previous} to {movement_end_frame} via Results UI. Rationale: {rationale}"
    )


def _append_note(existing: str, note: str) -> str:
    return "\n".join(item for item in (existing.strip(), note.strip()) if item)


def _previous_injured_side(feature_summary_path: Path) -> str:
    if not feature_summary_path.exists():
        return "unknown"
    try:
        payload = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return str(payload.get("run_metadata", {}).get("injured_side", "unknown") or "unknown")


def feature_display_label(feature_name: str) -> str:
    """Return concise user-facing feature text."""

    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def explain_status(status: str) -> str:
    """Translate internal status codes into user-readable wording."""

    return STATUS_EXPLANATIONS.get(status, status.replace("_", " ").lower() if status else "")


def result_frame_for_time(payload: dict, movement_end_relative_ms: float) -> int:
    """Return the exact source frame for the closest available graph time."""

    frames = payload["frames"]
    if not frames:
        raise ValueError("No result frames are available.")
    closest = min(
        frames,
        key=lambda item: abs(item["movement_end_relative_ms"] - movement_end_relative_ms),
    )
    return int(closest["source_frame_index"])


def render_results_page() -> str:
    """Return the self-contained first Results experience."""

    return SIMPLE_RESULTS_HTML


def build_case_synthesis_payload(
    current_case: AnnotationCase,
    case_views: tuple[AnnotationCase, ...] | None = None,
    *,
    data_root: str | Path = HUMAN_DATA_ROOT,
) -> dict:
    """Build a cautious case-level synthesis from saved per-view summaries only.

    This intentionally keeps projected quantities view-specific. It selects a
    preferred evidence source per feature but never averages angles or pools
    descriptive statistics across camera views.
    """

    root = Path(data_root)
    view_cases = _normalise_case_views(current_case, case_views)
    base = {
        "synthesis_version": "optional_multiview_evidence_synthesis_v1",
        "case_id": current_case.case_id,
        "current_view_slug": current_case.slug,
        "view_count": len(view_cases),
        "rules": {
            "per_view_analysis_is_authoritative": True,
            "averages_projected_angles_across_views": False,
            "pools_descriptive_statistics_across_views": False,
            "uses_3d_reconstruction": False,
        },
        "alignment": {
            "required_for_temporal_synthesis": len(view_cases) > 1,
            "manual_sync_supported": True,
            "anchors": load_view_alignment(root / "annotations" / "human", current_case.case_id)
            .to_dict()
            .get("anchors", []),
            "note": (
                "Views retain original source frame numbers. Case-level temporal alignment "
                "should use human-identified movement/event anchors such as foot plant, "
                "directional transition, loss of control, or movement end."
            ),
        },
        "human_override": {
            "supported": True,
            "note": (
                "The automatic preferred view is preserved; a researcher-selected final "
                "view can be stored alongside it without deleting the recommendation."
            ),
        },
    }
    if len(view_cases) <= 1:
        return {
            **base,
            "available": False,
            "reason": "Single-view case; normal per-view Results are authoritative.",
            "feature_sources": [],
            "corroborated": [],
            "disagreements": [],
        }

    completed: list[dict] = []
    unavailable: list[dict] = []
    for view in view_cases:
        paths = _results_paths(view.slug, root)
        if not paths["case_feature_summary"].exists():
            unavailable.append(
                {
                    **_view_payload(view, root),
                    "reason": "No saved view-level feature summary exists yet.",
                }
            )
            continue
        try:
            summary = pd.read_parquet(paths["case_feature_summary"])
        except OSError as exc:
            unavailable.append({**_view_payload(view, root), "reason": str(exc)})
            continue
        completed.append(
            {
                "view": _view_payload(view, root),
                "features": _view_feature_candidates(view, summary),
            }
        )

    if len(completed) < 2:
        return {
            **base,
            "available": False,
            "reason": "At least two completed view-level analyses are required for synthesis.",
            "completed_views": [item["view"] for item in completed],
            "unavailable_views": unavailable,
            "feature_sources": [],
            "corroborated": [],
            "disagreements": [],
        }

    by_feature: dict[str, list[dict]] = {}
    for item in completed:
        for candidate in item["features"]:
            by_feature.setdefault(candidate["feature_name"], []).append(candidate)

    feature_sources = []
    corroborated = []
    disagreements = []
    for feature_name, candidates in sorted(by_feature.items()):
        supported = [item for item in candidates if item["support_score"] > 0]
        if not supported:
            continue
        ranked = sorted(supported, key=lambda item: item["support_score"], reverse=True)
        best = ranked[0]
        status = _cross_view_status(ranked)
        source = {
            "feature_name": feature_name,
            "display_label": feature_display_label(feature_name),
            "body_region": best["body_region"],
            "automatic_preferred_view": best,
            "human_selected_view": None,
            "human_review_required": len(ranked) > 1
            and abs(ranked[0]["support_score"] - ranked[1]["support_score"]) < 0.05,
            "other_supported_views": ranked[1:],
            "cross_view_status": status["status"],
            "cross_view_note": status["note"],
            "statistics_source": (
                "Per-view statistics from the automatic preferred view; not pooled "
                "or averaged across views."
            ),
        }
        feature_sources.append(source)
        if status["status"] == "CROSS_VIEW_CORROBORATED":
            corroborated.append(source)
        elif status["status"] == "CROSS_VIEW_DISAGREEMENT":
            disagreements.append(source)

    return _json_ready(
        {
            **base,
            "available": True,
            "completed_views": [item["view"] for item in completed],
            "unavailable_views": unavailable,
            "feature_sources": feature_sources,
            "corroborated": corroborated,
            "disagreements": disagreements,
        }
    )


def _normalise_case_views(
    current_case: AnnotationCase,
    case_views: tuple[AnnotationCase, ...] | None,
) -> tuple[AnnotationCase, ...]:
    views = list(case_views or (current_case,))
    if not any(view.slug == current_case.slug for view in views):
        views.append(current_case)
    matching = [view for view in views if view.case_id == current_case.case_id]
    return tuple(sorted(matching or [current_case], key=_view_sort_key))


def _case_views_payload(
    current_case: AnnotationCase,
    case_views: tuple[AnnotationCase, ...],
    root: Path,
) -> dict:
    views = [_view_payload(view, root) for view in case_views]
    primary = next((view for view in views if view["primary_view"]), views[0])
    return {
        "case_id": current_case.case_id,
        "view_count": len(views),
        "current_view_slug": current_case.slug,
        "primary_view_slug": primary["slug"],
        "views": views,
        "single_view_fallback": len(views) == 1,
    }


def _view_payload(case: AnnotationCase, root: Path) -> dict:
    metadata = _video_metadata_payload(case)
    timing_limited = bool(case.slow_motion and case.real_time_scale is None)
    return {
        "slug": case.slug,
        "case_id": case.case_id,
        "source_id": case.source_id,
        "view_id": case.view_id or case.source_id,
        "view_label": case.view_label,
        "primary_view": bool(case.primary_view),
        "player_name": case.player_name,
        "source_video": str(case.video_path),
        "perspective": case.perspective,
        "occlusion_level": case.occlusion_level,
        "view_quality": case.view_quality,
        "slow_motion": bool(case.slow_motion),
        "cropped_or_zoomed": bool(case.cropped_or_zoomed),
        "real_time_scale": case.real_time_scale,
        "timing_measurements_limited": timing_limited,
        "timing_note": (
            "Timing-dependent measurements should not be treated as real athlete time "
            "because this view is slow motion and no real-time scale is registered."
            if timing_limited
            else ""
        ),
        "notes": case.notes,
        "metadata": metadata,
        "results_available": human_results_available(case, root),
    }


def _video_metadata_payload(case: AnnotationCase) -> dict | None:
    try:
        metadata = read_video_metadata(case.video_path)
    except ValueError:
        return None
    return {
        "fps": metadata.fps,
        "resolution": {"width": metadata.width, "height": metadata.height},
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
    }


def _view_feature_candidates(view: AnnotationCase, summary: pd.DataFrame) -> list[dict]:
    candidates = []
    for _, row in summary.iterrows():
        feature_name = str(row["feature_name"])
        candidates.append(
            {
                "view_slug": view.slug,
                "view_id": view.view_id or view.source_id,
                "view_label": view.view_label,
                "view_perspective": view.perspective,
                "feature_name": feature_name,
                "body_region": str(row.get("body_region", "")),
                "quality_category": str(row.get("quality_category", "")),
                "analytics_eligibility": str(row.get("analytics_eligibility", "")),
                "geometry_completeness": _optional_float(row.get("geometry_completeness")),
                "dynamic_completeness": _optional_float(row.get("dynamic_completeness")),
                "mean": _optional_float(row.get("mean")),
                "minimum": _optional_float(row.get("minimum")),
                "maximum": _optional_float(row.get("maximum")),
                "range": _optional_float(row.get("range")),
                "pre_late_change": _optional_float(row.get("pre_late_change")),
                "primary_rejection_reason": _readable_reason(row.get("primary_rejection_reason")),
                "support_score": _view_feature_support_score(row),
                "open_view_url": f"/results?case={view.slug}",
            }
        )
    return candidates


def _view_feature_support_score(row: pd.Series) -> float:
    quality = str(row.get("quality_category", "")).upper()
    eligibility = str(row.get("analytics_eligibility", "")).upper()
    geometry = _optional_float(row.get("geometry_completeness")) or 0.0
    dynamic = _optional_float(row.get("dynamic_completeness")) or 0.0
    quality_score = {"SUPPORTED": 1.0, "GOOD": 1.0, "LIMITED": 0.5}.get(quality, 0.0)
    eligibility_score = 0.4 if "READY" in eligibility or bool(row.get("analytics_eligible")) else 0.0
    return quality_score + eligibility_score + geometry * 0.4 + dynamic * 0.2


def _cross_view_status(candidates: list[dict]) -> dict:
    if len(candidates) < 2:
        return {
            "status": "SINGLE_VIEW_EVIDENCE",
            "note": "Only one completed view supports this feature.",
        }
    directions = [
        _movement_direction(item["pre_late_change"], item["feature_name"])
        for item in candidates
    ]
    directions = [direction for direction in directions if direction != "unknown"]
    if len(set(directions)) > 1:
        return {
            "status": "CROSS_VIEW_DISAGREEMENT",
            "note": (
                "Supported views show materially different projected changes. This may "
                "reflect camera projection, occlusion, timing alignment, or landmark "
                "quality; it is not automatically a model error."
            ),
        }
    return {
        "status": "CROSS_VIEW_CORROBORATED",
        "note": "Multiple completed views support this qualitative feature evidence.",
    }


def _movement_direction(change: float | None, feature_name: str) -> str:
    if change is None:
        return "unknown"
    threshold = 5.0 if feature_name.endswith("_deg") else 0.05
    if abs(change) < threshold:
        return "stable"
    return "increase" if change > 0 else "decrease"


def _view_sort_key(case: AnnotationCase) -> tuple[int, str, str]:
    return (0 if case.primary_view else 1, case.view_label.lower(), case.slug)


def _results_paths(slug: str, root: Path) -> dict[str, Path]:
    return {
        "annotation_session": root / "annotations" / "human" / f"{slug}_annotation_session_human.json",
        "movement_window": root / "annotations" / "human" / f"{slug}_movement_window_human.json",
        "roi_keyframes": root / "annotations" / "human" / f"{slug}_target_roi_human.csv",
        "movement_profile": root / "profiles" / "human" / f"{slug}_movement_profile.json",
        "evidence_profile": root / "quality" / "human" / f"{slug}_evidence_profile.json",
        "reliability_summary": root / "quality" / "human" / f"{slug}_reliability_summary.json",
        "dynamic_quality_summary": (
            root / "quality" / "human" / f"{slug}_dynamic_quality_summary.json"
        ),
        "case_feature_summary": (
            root / "analytics" / "human" / f"{slug}_case_feature_summary.parquet"
        ),
        "dynamic_features": root / "dynamics" / "human" / f"{slug}_dynamic_features.parquet",
        "path_features": root / "path" / "human" / f"{slug}_projected_movement_path.parquet",
        "path_quality_summary": (
            root / "quality" / "human" / f"{slug}_path_quality_summary.json"
        ),
        "semantic_observations": (
            root / "semantic" / "human" / f"{slug}_movement_observations.json"
        ),
        "observable_descriptions": (
            root / "semantics" / "human" / f"{slug}_observable_movement_descriptions.json"
        ),
        "movement_phases": root / "phases" / "human" / f"{slug}_movement_phases.json",
        "phase_frame_map": root / "phases" / "human" / f"{slug}_phase_frame_map.parquet",
        "movement_change_score": (
            root / "phases" / "human" / f"{slug}_movement_change_score.parquet"
        ),
        "phase_transitions": root / "phases" / "human" / f"{slug}_phase_transitions.csv",
        "processed_pose": root / "processed" / "human" / f"{slug}_processed_pose.parquet",
        "frame_quality": root / "quality" / "human" / f"{slug}_frame_quality.csv",
        "landmark_quality": root / "quality" / "human" / f"{slug}_landmark_quality.parquet",
    }


def _require_human_paths(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ValueError(f"Human results are incomplete. Missing: {missing}")
    for path in paths.values():
        if "/human/" not in path.as_posix() and not path.name.endswith("_human.csv"):
            raise ValueError(f"Results view only accepts HUMAN namespace files: {path}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _header_metrics(session, movement_window) -> dict:
    return {
        "target": "Human verified",
        "movement_start_frame": movement_window.movement_start_frame,
        "movement_end_frame": movement_window.movement_end_frame,
        "movement_duration_seconds": movement_window.duration_ms / 1000.0,
        "roi_keyframes": session.manual_roi_keyframe_count,
    }


def _evidence_dimensions(
    evidence_profile: dict,
    reliability: dict,
    dynamic_quality: dict,
    case_summary: pd.DataFrame,
) -> list[dict]:
    geometry_coverage = float(case_summary["geometry_completeness"].mean())
    dynamic_feature_coverage = float(case_summary["dynamic_completeness"].mean())
    robust_sample_completeness = _optional_float(dynamic_quality.get("robust_dynamic_completeness"))
    return [
        {
            "name": "Target evidence",
            "label": coverage_label(reliability.get("target_tracking_coverage")),
            "value": _optional_float(reliability.get("target_tracking_coverage")),
            "explanation": "Valid target frames divided by frames inside the human Movement Window.",
        },
        {
            "name": "Pose availability",
            "label": coverage_label(reliability.get("pose_frame_coverage")),
            "value": _optional_float(reliability.get("pose_frame_coverage")),
            "explanation": "Frames where the pose backend returned pose evidence for the selected target ROI.",
        },
        {
            "name": "Geometry evidence",
            "label": coverage_label(geometry_coverage),
            "value": geometry_coverage,
            "explanation": "Mean supported-frame fraction across projected 2D geometry features.",
        },
        {
            "name": "Dynamic feature coverage",
            "label": coverage_label(dynamic_feature_coverage),
            "value": dynamic_feature_coverage,
            "explanation": "Mean robust-dynamic completeness across feature-level Movement Profile summaries.",
        },
        {
            "name": "Robust dynamic sample completeness",
            "label": coverage_label(robust_sample_completeness),
            "value": robust_sample_completeness,
            "explanation": (
                "Supported robust dynamic samples divided by all eligible dynamic samples; "
                "this denominator excludes descriptors that are not dynamic features."
            ),
        },
        {
            "name": "Movement Profile evidence",
            "label": evidence_profile["evidence_overview"]["overall_quality_label"],
            "value": None,
            "explanation": (
                f"{evidence_profile['supported_feature_count']} supported, "
                f"{evidence_profile['limited_feature_count']} limited, "
                f"{evidence_profile['unavailable_feature_count']} unavailable features."
            ),
        },
    ]


def coverage_label(value: Any) -> str:
    """Convert a completeness fraction into a transparent UI label."""

    numeric = _optional_float(value)
    if numeric is None:
        return "UNAVAILABLE"
    if numeric >= 0.9:
        return "HIGH"
    if numeric >= 0.7:
        return "GOOD"
    if numeric >= 0.4:
        return "LIMITED"
    return "LOW"


def _body_region_evidence(case_summary: pd.DataFrame) -> list[dict]:
    rows = []
    for region, group in case_summary.groupby("body_region", sort=True):
        counts = group["quality_category"].value_counts().to_dict()
        rows.append(
            {
                "body_region": str(region),
                "display_label": _region_label(str(region)),
                "supported": int(counts.get("SUPPORTED", 0)),
                "limited": int(counts.get("LIMITED", 0)),
                "unavailable": int(counts.get("UNAVAILABLE", 0)),
                "total": len(group),
                "geometry_completeness": float(group["geometry_completeness"].mean()),
                "dynamic_completeness": float(group["dynamic_completeness"].mean()),
            }
        )
    return rows


def _target_segmentation_payload(slug: str, data_root: Path) -> dict:
    prompt_path = mask_prompt_path(data_root, slug)
    prompts = load_mask_prompts(prompt_path)
    return {
        "method": "human_roi_seeded_grabcut_with_human_point_prompts",
        "mask_version": "m5_9_target_mask_grabcut_prompt_v1",
        "prompt_file": str(prompt_path),
        "prompt_count": len(prompts),
        "labels": ["target", "opponent"],
        "human_evidence_types": ["visible_target_region", "visible_non_target_region"],
        "note": (
            "The generated mask is seeded by rectangular target ROI plus optional human "
            "visible target/non-target region prompts; it is not proof that hidden joints "
            "were observed."
        ),
    }


def _laterality_mapping(movement_profile: dict, dynamic_df: pd.DataFrame) -> dict[str, str]:
    """Return the injured/contralateral side mapping preserved by M3 metadata."""

    run_metadata = (
        movement_profile.get("provenance", {})
        .get("geometry_feature_summary", {})
        .get("run_metadata", {})
    )
    injured_side = str(run_metadata.get("injured_side") or "").lower()
    if injured_side in {"left", "right"}:
        return {
            "injured": injured_side,
            "contralateral": "left" if injured_side == "right" else "right",
        }
    for metadata in dynamic_df.get("metadata", pd.Series(dtype=object)).dropna():
        if isinstance(metadata, dict):
            mapping = metadata.get("laterality_mapping")
            if isinstance(mapping, dict) and mapping.get("injured") in {"left", "right"}:
                return {
                    "injured": str(mapping["injured"]),
                    "contralateral": str(mapping.get("contralateral") or ""),
                }
    return {}


def _selected_existing_features(dynamic_df: pd.DataFrame) -> tuple[str, ...]:
    existing = {str(name) for name in dynamic_df["feature_name"].unique()}
    selected: list[str] = []
    for names in RESULT_FEATURE_GROUPS.values():
        for name in names:
            if name in existing and name not in selected:
                selected.append(name)
    return tuple(selected)


def _feature_groups(selected_features: tuple[str, ...]) -> dict:
    selected = set(selected_features)
    return {
        group: [
            {"feature_name": name, "display_label": feature_display_label(name)}
            for name in names
            if name in selected
        ]
        for group, names in RESULT_FEATURE_GROUPS.items()
    }


def _feature_cards(
    case_summary: pd.DataFrame,
    feature_profiles: dict[str, dict],
    selected_features: tuple[str, ...],
) -> dict[str, dict]:
    indexed = case_summary.set_index("feature_name")
    cards: dict[str, dict] = {}
    for feature_name in selected_features:
        if feature_name not in indexed.index:
            continue
        row = indexed.loc[feature_name]
        profile = feature_profiles.get(feature_name, {})
        dynamic_status = (
            "SUPPORTED"
            if bool(row["dynamic_analytics_eligible"])
            else ("UNAVAILABLE" if float(row["dynamic_completeness"]) == 0 else "LIMITED")
        )
        cards[feature_name] = {
            "feature_name": feature_name,
            "display_label": feature_display_label(feature_name),
            "canonical_name": feature_name,
            "body_region": str(row["body_region"]),
            "body_region_label": _region_label(str(row["body_region"])),
            "unit": str(profile.get("unit") or ""),
            "sequence_evidence": str(row["quality_category"]),
            "geometry_completeness": _optional_float(row["geometry_completeness"]),
            "dynamic_evidence": dynamic_status,
            "dynamic_completeness": _optional_float(row["dynamic_completeness"]),
            "at_movement_end": _optional_float(row["value_at_t0"]),
            "at_movement_end_status": str(row["t0_status"]),
            "why_limited": _readable_reason(
                row.get("primary_rejection_reason") or row.get("eligibility_reason")
            ),
            "landmarks_used": _listify(profile.get("landmarks_used")),
            "analytics_eligibility": str(row["analytics_eligibility"]),
            "notes": str(profile.get("notes", "")),
        }
    return cards


def _trajectories(dynamic_df: pd.DataFrame, selected_features: tuple[str, ...]) -> dict:
    output: dict[str, list[dict]] = {}
    filtered = dynamic_df[dynamic_df["feature_name"].isin(selected_features)]
    for feature_name, rows in filtered.groupby("feature_name", sort=False):
        points = []
        for _, row in rows.sort_values("source_frame_index").iterrows():
            feature_status = str(row.get("feature_status", ""))
            points.append(
                {
                    "source_frame_index": int(row["source_frame_index"]),
                    "analysis_frame_index": int(row["analysis_frame_index"]),
                    "timestamp_ms": _optional_float(row["timestamp_ms"]),
                    "movement_elapsed_ms": _optional_float(row.get("movement_elapsed_ms")),
                    "movement_end_relative_ms": _optional_float(row.get("movement_end_relative_ms")),
                    "value": (
                        _optional_float(row["feature_value"])
                        if feature_status == "SUPPORTED"
                        else None
                    ),
                    "unit": str(row.get("unit", "")),
                    "feature_status": feature_status,
                    "feature_status_text": explain_status(feature_status),
                    "dynamic_status": str(row.get("dynamic_status", "")),
                    "dynamic_status_text": explain_status(str(row.get("dynamic_status", ""))),
                    "rejection_reason": _readable_reason(row.get("rejection_reason")),
                    "dynamic_rejection_reason": _readable_reason(row.get("dynamic_rejection_reason")),
                }
            )
        output[str(feature_name)] = points
    return output


def _phase_frame_map_payload(phase_frame_map: pd.DataFrame) -> list[dict]:
    columns = [
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "movement_elapsed_ms",
        "movement_end_relative_ms",
        "phase_id",
        "phase_index",
        "phase_title",
        "change_score",
        "smoothed_change_score",
        "contributing_descriptors",
        "candidate_boundary",
        "selected_boundary",
        "sustained_shift_score",
    ]
    existing = [column for column in columns if column in phase_frame_map.columns]
    return [
        _json_ready(row)
        for row in phase_frame_map[existing]
        .sort_values("source_frame_index")
        .to_dict(orient="records")
    ]


def _frame_index(dynamic_df: pd.DataFrame) -> list[dict]:
    columns = [
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "movement_elapsed_ms",
        "movement_end_relative_ms",
    ]
    return [
        _json_ready(row)
        for row in dynamic_df[columns]
        .drop_duplicates("source_frame_index")
        .sort_values("source_frame_index")
        .to_dict(orient="records")
    ]


def _quality_limitations(reliability: dict, dynamic_quality: dict, dynamic_df: pd.DataFrame) -> list[dict]:
    limitations: list[dict] = []
    frame_counts = reliability.get("frame_status_counts", {})
    for status, count in frame_counts.items():
        if status != "VALID_TARGET" and count:
            limitations.append(
                {
                    "label": explain_status(status),
                    "source": "Frame QC",
                    "count": int(count),
                    "internal_status": status,
                }
            )
    for status, count in dynamic_quality.get("dynamic_status_counts", {}).items():
        if status not in {"SUPPORTED", "NOT_DYNAMIC_FEATURE"} and count:
            limitations.append(
                {
                    "label": explain_status(status),
                    "source": "Dynamic QC",
                    "count": int(count),
                    "internal_status": status,
                }
            )
    reasons = (
        dynamic_df["dynamic_rejection_reason"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(5)
    )
    for reason, count in reasons.items():
        limitations.append(
            {
                "label": _readable_reason(reason),
                "source": "Why data was withheld",
                "count": int(count),
                "internal_status": str(reason),
            }
        )
    return limitations


def _bbox_for_result_frame(paths: dict[str, Path], source_frame_index: int) -> BBox:
    session = load_human_annotation_session(paths["annotation_session"])
    return propagated_bbox(session.roi_keyframes, int(source_frame_index))


def _landmarks_for_result_frame(paths: dict[str, Path], source_frame_index: int) -> dict:
    processed_pose = pd.read_parquet(paths["processed_pose"])
    rows = processed_pose[processed_pose["source_frame_index"].astype(int).eq(int(source_frame_index))]
    landmarks = {}
    for _, row in rows.iterrows():
        x = row.get("smoothed_x")
        y = row.get("smoothed_y")
        if _optional_float(x) is None or _optional_float(y) is None:
            x = row.get("clean_x")
            y = row.get("clean_y")
        landmarks[str(row["landmark_name"])] = {
            "x_px": _optional_float(x),
            "y_px": _optional_float(y),
            "confidence": _optional_float(row.get("confidence")),
            "observed": not bool(row.get("rejected")) and _optional_float(x) is not None,
        }
    return landmarks


def _row_dict(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    return _series_dict(df.iloc[0])


def _series_dict(row: pd.Series) -> dict:
    return {str(key): _json_ready(value) for key, value in row.to_dict().items()}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if pd.isna(value) if not isinstance(value, list | tuple | dict) else False:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item is not None]
    return []


def _clean_reason(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _readable_reason(value: Any) -> str:
    reason = _clean_reason(value)
    if not reason:
        return ""
    return STATUS_EXPLANATIONS.get(reason, reason.replace("_", " "))


def _region_label(region: str) -> str:
    return {
        "lower_limb": "Lower Limb",
        "trunk_pelvis": "Trunk & Pelvis",
        "upper_body": "Upper Body",
        "bilateral": "Bilateral",
        "whole_body": "Whole Body",
    }.get(region, region.replace("_", " ").title())


def _cross_case_label(name: str) -> str:
    return {
        "similarity": "Similar documented cases",
        "umap": "Movement Landscape",
        "clustering": "Exploratory clustering",
        "association_rules": "Association rules",
    }[name]


SIMPLE_RESULTS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Explorer - Results</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2a33;
      --muted: #627181;
      --line: #d7dfe7;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --accent: #215f9a;
      --accent-soft: #e9f2fb;
      --warn: #9a6400;
      --bad: #9d2735;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 14px;
    }
    .header {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .header-line {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-weight: 800;
      align-items: center;
    }
    .header-line span:not(:last-child)::after {
      content: "|";
      color: var(--muted);
      margin-left: 8px;
      font-weight: 500;
    }
    .header-line span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .subtle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      margin: 4px 0 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
    }
    .video-frame {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101820;
      border-radius: 8px;
      display: block;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    button, select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      font: inherit;
      font-weight: 750;
      min-height: 38px;
      padding: 7px 10px;
    }
    button {
      cursor: pointer;
      min-width: 50px;
    }
    button:hover, select:hover { border-color: var(--accent); }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      min-width: 96px;
    }
    button.warn {
      background: #fff8e6;
      border-color: #e3b24f;
      color: #744500;
    }
    button[disabled] {
      cursor: wait;
      opacity: 0.7;
    }
    .readout {
      margin-left: auto;
      color: var(--muted);
      font-weight: 750;
    }
    .boundary-control {
      border: 1px solid #f0d89b;
      border-radius: 8px;
      background: #fffaf0;
      padding: 9px 10px;
      margin-top: 10px;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .boundary-control strong {
      color: #744500;
    }
    .boundary-status {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      min-width: min(100%, 360px);
    }
    .selector-grid {
      display: grid;
      grid-template-columns: minmax(180px, 240px) minmax(260px, 1fr);
      gap: 10px;
      align-items: end;
    }
    .view-row {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 10px;
    }
    .view-row select {
      min-width: 220px;
    }
    .view-count {
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
      padding-bottom: 9px;
    }
    .synthesis {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 10px;
    }
    label {
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 4px;
      text-transform: uppercase;
    }
    select { width: 100%; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { font-size: 18px; margin-bottom: 0; }
    h2 { font-size: 18px; margin-bottom: 6px; }
    .description {
      color: var(--ink);
      font-size: 15px;
      margin-bottom: 4px;
    }
    .technical-label {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      margin-bottom: 10px;
    }
    .info-icon {
      color: var(--accent);
      cursor: help;
      font-size: 15px;
      margin-left: 4px;
      vertical-align: 1px;
    }
    .measurement-heading {
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 12px;
    }
    .measurement-heading h3 {
      font-size: 18px;
      margin-bottom: 5px;
    }
    .section-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin: 10px 0 6px;
      text-transform: uppercase;
    }
    .trajectory-interpretation {
      border-left: 4px solid var(--accent);
      margin: 9px 0 0;
      padding: 7px 0 7px 11px;
      line-height: 1.4;
    }
    .headline-values {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .value-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 72px;
      background: #fbfcfd;
    }
    .value-card span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 750;
      margin-bottom: 5px;
    }
    .value-card strong {
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }
    .chart {
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    .filmstrip {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .thumb {
      min-width: 0;
      padding: 0;
      text-align: left;
      overflow: hidden;
      background: #fff;
    }
    .thumb img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      background: #101820;
    }
    .thumb span {
      display: block;
      padding: 6px 7px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.25;
    }
    details {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    summary {
      cursor: pointer;
      font-weight: 850;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .phase-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .phase-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .phase-card span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 5px;
      text-transform: uppercase;
    }
    .phase-card h3 {
      font-size: 15px;
      margin-bottom: 6px;
    }
    .phase-card p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      margin-bottom: 8px;
    }
    .whole-movement-summary {
      border-left: 4px solid var(--accent);
      padding: 8px 0 8px 12px;
      margin-top: 10px;
      max-width: 980px;
    }
    .whole-movement-summary h3 {
      font-size: 14px;
      margin-bottom: 5px;
    }
    .whole-movement-summary p {
      line-height: 1.45;
      margin-bottom: 0;
    }
    .phase-explanation {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      margin-top: 8px;
    }
    .phase-explanation strong {
      color: var(--ink);
      display: block;
      font-size: 12px;
      margin-bottom: 3px;
    }
    .story-change-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .story-change {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      min-width: 0;
    }
    .story-change-heading {
      display: flex;
      gap: 7px;
      justify-content: space-between;
      align-items: flex-start;
    }
    .story-change p {
      margin: 5px 0 0;
    }
    .movement-glyph-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
    }
    .movement-glyph-legend strong {
      color: var(--ink);
    }
    .glyph-line-key {
      display: inline-block;
      width: 22px;
      border-top: 3px solid #176d4d;
      margin-right: 5px;
      vertical-align: 3px;
    }
    .glyph-line-key.start {
      border-top: 2px dashed #215f9a;
    }
    .phase-mini-visual {
      width: 100%;
      height: 132px;
      display: block;
      background: #f4f7fa;
      border-radius: 6px;
      margin-top: 7px;
    }
    .phase-mini-values {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 2px 10px;
      align-items: baseline;
      margin-top: 7px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }
    .phase-mini-values strong {
      color: var(--ink);
      font-size: 12px;
    }
    .phase-mini-values span {
      display: block;
      margin: 0;
      font-weight: 700;
      text-transform: none;
    }
    .phase-mini-values .phase-mini-delta {
      grid-column: 2;
      text-align: right;
    }
    .phase-mini-values.multi {
      border-top: 1px solid var(--line);
      padding-top: 6px;
    }
    .phase-mini-values.multi strong {
      font-size: 11px;
      text-align: right;
    }
    .phase-snapshots {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
      margin-top: 10px;
    }
    .phase-snapshot {
      min-width: 0;
      padding: 0;
      text-align: left;
      overflow: hidden;
      background: #fff;
    }
    .phase-snapshot img {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      display: block;
      background: #101820;
    }
    .phase-snapshot span {
      display: block;
      padding: 6px 7px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }
    .phase-snapshot.change-lower { border-color: #b8d9b4; }
    .phase-snapshot.change-larger { border-color: #e7d98d; }
    .phase-snapshot.change-largest { border-color: #e2a2a0; }
    .movement-change-legend {
      color: var(--muted);
      font-size: 11px;
      margin: 7px 0 0;
    }
    .operator-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .operator-card-wide {
      grid-column: 1 / -1;
    }
    .operator-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .operator-card h3 {
      font-size: 15px;
      margin-bottom: 8px;
    }
    .operator-card ul {
      margin: 0;
      padding-left: 18px;
    }
    .operator-card li {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      margin-bottom: 7px;
    }
    .support-overview {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      align-items: center;
      margin-top: 8px;
    }
    .support-overview strong {
      font-size: 15px;
    }
    .support-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      margin: 0;
    }
    .support-details {
      margin: 10px 0 0;
      background: #fbfcfd;
    }
    .gap-item {
      border-top: 1px solid var(--line);
      padding: 9px 0;
    }
    .gap-item:first-child {
      border-top: 0;
    }
    .gap-item p {
      margin: 4px 0 0;
    }
    .operator-row {
      border-top: 1px solid var(--line);
      padding-top: 7px;
      margin-top: 7px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .compact-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .compact-table th,
    .compact-table td {
      border-top: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
      vertical-align: top;
    }
    .compact-table th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .advanced-details {
      margin-top: 10px;
      margin-bottom: 0;
      background: #fff;
    }
    .audit-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .audit-section {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      min-width: 0;
    }
    .audit-section h3 {
      font-size: 15px;
      margin-bottom: 7px;
    }
    .audit-section-wide {
      grid-column: 1 / -1;
    }
    .technical {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .unavailable {
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      min-height: 220px;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-weight: 800;
      text-align: center;
      padding: 20px;
    }
    .unavailable[hidden] {
      border: 0;
      min-height: 0;
      padding: 0;
      display: none !important;
    }
    .status {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 850;
      font-size: 12px;
    }
    .status.unavailable {
      background: #fff4e0;
      color: var(--warn);
      border: 0;
      min-height: 0;
      display: inline-block;
      padding: 2px 8px;
    }
    @media (max-width: 760px) {
      main { padding: 8px; }
      .selector-grid, .headline-values, .stats-grid, .phase-grid, .operator-grid, .filmstrip,
      .story-change-grid, .phase-snapshots, .audit-grid {
        grid-template-columns: 1fr;
      }
      .audit-section-wide { grid-column: auto; }
      .view-row { align-items: stretch; }
      .readout { margin-left: 0; width: 100%; }
      .chart { height: 260px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="header">
      <div class="header-line" id="compactHeader">
        <span>Loading</span>
      </div>
      <p class="subtle" id="headerNote"></p>
    </section>

    <section class="panel" id="viewPanel" hidden>
      <div class="view-row">
        <div>
          <label for="viewSelect">View</label>
          <select id="viewSelect"></select>
        </div>
        <span class="view-count" id="viewCount"></span>
        <button type="button" id="synthesisButton">Compare / synthesise views</button>
      </div>
      <div id="caseSynthesisPanel" class="synthesis" hidden></div>
    </section>

    <section class="panel">
      <img id="videoFrame" class="video-frame" alt="Selected source frame" />
      <div class="controls" aria-label="Frame navigation">
        <button id="backFiveButton" type="button">-5</button>
        <button id="backOneButton" type="button">-1</button>
        <button id="restartButton" type="button">Restart</button>
        <button id="playPauseButton" class="primary" type="button">Play</button>
        <button id="forwardOneButton" type="button">+1</button>
        <button id="forwardFiveButton" type="button">+5</button>
        <span id="frameReadout" class="readout">Frame</span>
      </div>
      <div class="boundary-control">
        <div>
          <strong>Human analysis boundary</strong>
          <p class="subtle">If the target switches or disappears after this frame, end analysis here and rebuild the results.</p>
        </div>
        <button id="trimAnalysisButton" class="warn" type="button">End analysis here + regenerate</button>
        <span id="trimAnalysisStatus" class="boundary-status"></span>
      </div>
    </section>

    <section class="panel" id="phaseStoryPanel">
      <h2>Movement Story</h2>
      <p class="subtle" id="phaseStorySummary"></p>
      <div class="whole-movement-summary" id="wholeMovementSummary"></div>
      <div class="movement-glyph-legend" aria-label="Movement diagram legend">
        <strong>Movement diagrams</strong>
        <span><i class="glyph-line-key start"></i>Blue dashed = phase start</span>
        <span><i class="glyph-line-key"></i>Green solid = phase end</span>
        <span>Arrow = direction of change</span>
        <span>Arc = projected angle</span>
      </div>
      <div class="phase-grid" id="phaseStoryGrid"></div>
    </section>

    <section class="panel" id="featurePanel">
      <h2>Selected Measurement</h2>
      <div class="selector-grid">
        <div>
          <label for="featureCategorySelect">Feature category</label>
          <select id="featureCategorySelect"></select>
        </div>
        <div>
          <label for="featureSelect">Measurement</label>
          <select id="featureSelect"></select>
        </div>
      </div>
      <p class="subtle" id="categoryAvailabilityNote"></p>
      <div class="measurement-heading">
        <h3 id="featureTitle">Feature</h3>
        <p class="description" id="featureDescription"></p>
        <p class="technical-label" id="featureTechnicalLabel"></p>
      </div>
      <p class="section-label">Movement change</p>
      <div class="headline-values" id="headlineValues"></div>
      <canvas id="featureGraph" class="chart" width="1100" height="260"></canvas>
      <div id="unavailableVisual" class="unavailable" hidden></div>
      <p class="trajectory-interpretation" id="trajectoryInterpretation"></p>
      <div id="filmstrip" class="filmstrip"></div>
      <details id="moreStatistics">
        <summary>Descriptive statistics</summary>
        <div class="stats-grid" id="moreStatisticsGrid"></div>
      </details>
      <div class="measurement-heading">
        <h3>Phase comparison</h3>
        <div id="phaseComparisonPanel"></div>
      </div>
    </section>

    <section class="panel" id="operatorAnalyticsPanel">
      <h2>Measurement Support</h2>
      <p class="subtle">How much of the selected trajectory is supported by defensible target and landmark evidence.</p>
      <div id="supportPanel"></div>
      <details class="support-details" id="unsupportedIntervalDetails">
        <summary id="unsupportedIntervalSummary">Why are some frames unsupported?</summary>
        <div id="gapReasonPanel"></div>
      </details>
    </section>

    <details id="advancedEvidenceDetails">
      <summary>Advanced Evidence Details</summary>
      <p class="subtle">Technical measurement dependencies, frame-level QC, raw status codes, and provenance.</p>
      <div class="audit-grid">
        <section class="audit-section">
          <h3>Measurement provenance</h3>
          <div id="dependencyPanel"></div>
        </section>
        <section class="audit-section">
          <h3>Support aggregation</h3>
          <p id="evidenceSummary" class="subtle"></p>
        </section>
        <section class="audit-section">
          <h3>Selected-frame QC and raw status</h3>
          <pre id="technicalText" class="technical"></pre>
        </section>
        <section class="audit-section">
          <h3>Case and source provenance</h3>
          <pre id="provenanceText" class="technical"></pre>
        </section>
      </div>
    </details>
  </main>

<script>
const params = new URLSearchParams(window.location.search);
const caseSlug = params.get('case') || 'christen_press';
const $ = (id) => document.getElementById(id);

let result = null;
let currentFrame = 0;
let playTimer = null;
let selectedCategory = 'LOWER LIMB';
let selectedFeatureId = 'injured_hka';

const FEATURE_CATEGORIES = {
  'LOWER LIMB': [
    {
      id: 'injured_hka',
      label: 'Injured projected HKA',
      metrics: ['injured_hka_angle_2d_deg'],
      description: 'The projected injured-side lower-limb configuration across supported measurement frames.',
      visual: 'line'
    },
    {
      id: 'contralateral_hka',
      label: 'Contralateral projected HKA',
      metrics: ['contralateral_hka_angle_2d_deg'],
      description: 'The projected contralateral lower-limb configuration across supported measurement frames.',
      visual: 'line'
    },
    {
      id: 'injured_hka_rate_change',
      label: 'Supported HKA change',
      metrics: ['injured_hka_angle_2d_deg'],
      description: 'Supported change in the injured-side projected hip-knee-ankle angle. Unsupported intervals remain gaps.',
      visual: 'line'
    },
    {
      id: 'knee_line_deviation',
      label: 'Knee-line deviation',
      metrics: ['right_knee_line_deviation_normalized'],
      description: 'Projected knee position relative to the hip-ankle line, normalized to body scale.',
      visual: 'line'
    },
    {
      id: 'knee_ankle_offset',
      label: 'Knee-ankle offset',
      metrics: ['right_knee_ankle_x_offset_normalized'],
      description: 'Projected knee-to-ankle x-offset, normalized to body scale.',
      visual: 'line'
    }
  ],
  'BILATERAL': [
    {
      id: 'left_right_hka_pair',
      label: 'Left/right projected HKA',
      metrics: ['left_hka_angle_2d_deg', 'right_hka_angle_2d_deg'],
      description: 'Paired left/right projected lower-limb configuration across supported frames. This does not infer injury side.',
      visual: 'paired'
    },
    {
      id: 'left_right_knee_line',
      label: 'Left/right knee-line deviation',
      metrics: ['left_knee_line_deviation_normalized', 'right_knee_line_deviation_normalized'],
      description: 'Paired left/right projected knee position relative to each hip-ankle line, normalized to body scale.',
      visual: 'paired'
    },
    {
      id: 'hka_pair',
      label: 'Injured vs contralateral HKA',
      metrics: ['injured_hka_angle_2d_deg', 'contralateral_hka_angle_2d_deg'],
      description: 'Paired projected lower-limb configuration for the injured and contralateral sides.',
      visual: 'paired'
    },
    {
      id: 'hka_signed_difference',
      label: 'Signed HKA difference',
      metrics: ['hka_projected_bilateral_difference_deg'],
      description: 'Projected injured-minus-contralateral HKA difference through supported frames.',
      visual: 'line'
    },
    {
      id: 'hka_absolute_difference',
      label: 'Absolute HKA difference',
      metrics: ['hka_projected_bilateral_absolute_difference_deg'],
      description: 'Absolute projected difference between injured and contralateral HKA.',
      visual: 'line'
    },
    {
      id: 'bilateral_change',
      label: 'Bilateral change through time',
      metrics: ['hka_projected_bilateral_difference_deg'],
      description: 'How the signed projected bilateral HKA relationship changes across supported frames.',
      visual: 'line'
    },
    {
      id: 'bilateral_knee_line',
      label: 'Bilateral knee-line difference',
      metrics: ['knee_line_deviation_normalized_bilateral_difference'],
      description: 'Projected injured-minus-contralateral knee-line relationship, normalized to body scale.',
      visual: 'line'
    }
  ],
  'TRUNK & PELVIS': [
    {
      id: 'trunk_orientation',
      label: 'Trunk orientation',
      metrics: ['projected_trunk_axis_angle_deg'],
      description: 'Projected trunk-axis orientation across supported frames.',
      visual: 'line'
    },
    {
      id: 'pelvis_orientation',
      label: 'Hip/pelvis-line orientation',
      metrics: ['projected_hip_line_angle_deg'],
      description: 'Projected hip-line orientation across supported frames.',
      visual: 'line'
    },
    {
      id: 'shoulder_orientation',
      label: 'Shoulder orientation',
      metrics: ['projected_shoulder_line_angle_deg'],
      description: 'Projected shoulder-line orientation across supported frames.',
      visual: 'line'
    },
    {
      id: 'shoulder_pelvis_relationship',
      label: 'Shoulder-pelvis relationship',
      metrics: ['projected_shoulder_pelvis_orientation_difference_deg'],
      description: 'Projected shoulder-line orientation relative to hip-line orientation.',
      visual: 'line'
    }
  ],
  'UPPER BODY': [
    {
      id: 'elbow_pair',
      label: 'Left/right elbow angle',
      metrics: ['left_elbow_angle_2d_deg', 'right_elbow_angle_2d_deg'],
      description: 'Paired projected elbow angles across supported frames.',
      visual: 'paired'
    },
    {
      id: 'upper_arm_pair',
      label: 'Left/right upper-arm orientation',
      metrics: ['left_upper_arm_orientation_2d_deg', 'right_upper_arm_orientation_2d_deg'],
      description: 'Paired projected upper-arm orientation across supported frames.',
      visual: 'paired'
    },
    {
      id: 'wrist_relationship',
      label: 'Wrist-pelvis relationship',
      metrics: ['right_wrist_pelvis_x_offset_normalized'],
      description: 'Projected wrist-pelvis x-offset, normalized to body scale.',
      visual: 'line'
    }
  ],
  'TIMING': [
    {
      id: 'hka_extrema_timing',
      label: 'Timing of supported HKA extrema',
      metrics: ['injured_hka_angle_2d_deg'],
      description: 'Source-frame positions where the supported injured-side projected HKA reaches its minimum and maximum. This is not an injury timestamp.',
      visual: 'timing_extrema'
    },
    {
      id: 'hka_peak_change_timing',
      label: 'Timing of robust HKA change',
      metrics: ['injured_hka_angle_2d_deg'],
      description: 'Source-frame position of the largest supported frame-to-frame projected HKA change. Unsupported gaps are ignored.',
      visual: 'timing_change'
    }
  ]
};

fetch('/api/results?case=' + encodeURIComponent(caseSlug))
  .then((response) => {
    if (!response.ok) throw new Error('Results are unavailable.');
    return response.json();
  })
  .then((payload) => {
    result = payload;
    currentFrame = Number(result.movement_window?.movement_start_frame ?? frameBounds().start);
    initialiseFeatureSelection();
    initialiseControls();
    renderAll();
  })
  .catch((error) => {
    document.body.innerHTML = '<main><section class="panel"><h1>Results unavailable</h1><p>' + escapeHtml(error.message) + '</p></section></main>';
  });

function initialiseControls() {
  const categories = availableCategories();
  $('featureCategorySelect').innerHTML = categories
    .map((category) => '<option value="' + category + '">' + category + '</option>')
    .join('');
  $('featureCategorySelect').disabled = categories.length === 0;
  $('featureCategorySelect').value = selectedCategory;
  $('featureCategorySelect').onchange = () => {
    selectedCategory = $('featureCategorySelect').value;
    const feature = firstAvailableFeature(FEATURE_CATEGORIES[selectedCategory] || []);
    selectedFeatureId = feature?.id || null;
    renderFeatureOptions();
    renderSelectedFeature();
  };
  $('featureSelect').onchange = () => {
    selectedFeatureId = $('featureSelect').value;
    renderSelectedFeature();
  };
  $('restartButton').onclick = () => {
    if (playTimer) togglePlayback();
    setFrame(frameBounds().start);
  };
  $('backFiveButton').onclick = () => stepFrame(-5);
  $('backOneButton').onclick = () => stepFrame(-1);
  $('forwardOneButton').onclick = () => stepFrame(1);
  $('forwardFiveButton').onclick = () => stepFrame(5);
  $('playPauseButton').onclick = togglePlayback;
  $('trimAnalysisButton').onclick = trimAnalysisWindowAtCurrentFrame;
  $('featureGraph').addEventListener('click', graphClickToFrame);
  window.addEventListener('resize', () => renderSelectedFeature());
  renderFeatureOptions();
  renderCategoryAvailabilityNote();
}

function initialiseFeatureSelection() {
  const categories = availableCategories();
  if (!categories.length) {
    selectedCategory = Object.keys(FEATURE_CATEGORIES)[0] || '';
    selectedFeatureId = null;
    return;
  }
  if (!categories.includes(selectedCategory)) selectedCategory = categories[0];
  const feature = firstAvailableFeature(FEATURE_CATEGORIES[selectedCategory] || []);
  selectedFeatureId = feature?.id || null;
}

async function trimAnalysisWindowAtCurrentFrame() {
  const bounds = frameBounds();
  if (currentFrame >= bounds.end) {
    $('trimAnalysisStatus').textContent = 'The selected frame is already the Movement End.';
    return;
  }
  const defaultReason = 'Post-injury occlusion or wrong-player reacquisition after this frame.';
  const rationale = window.prompt(
    'Why should analysis end at source frame ' + currentFrame + '?',
    defaultReason
  );
  if (rationale === null) {
    $('trimAnalysisStatus').textContent = 'Analysis boundary change cancelled.';
    return;
  }
  const confirmed = window.confirm(
    'Regenerate this analysis using frames '
      + bounds.start + '-' + currentFrame
      + '? The source video is preserved; only the human analysis window changes.'
  );
  if (!confirmed) {
    $('trimAnalysisStatus').textContent = 'Analysis boundary change cancelled.';
    return;
  }
  if (playTimer) togglePlayback();
  const button = $('trimAnalysisButton');
  button.disabled = true;
  $('trimAnalysisStatus').textContent = 'Regenerating analysis through source frame '
    + currentFrame + '. This can take about a minute.';
  try {
    const response = await fetch('/api/results/trim-analysis-window', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        case: caseSlug,
        frame: currentFrame,
        rationale: rationale.trim() || defaultReason,
        annotator_id: 'researcher_01'
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Regeneration failed.');
    $('trimAnalysisStatus').textContent = 'Regenerated. Reloading results...';
    window.location.href = data.result_url + '&updated=' + Date.now();
  } catch (error) {
    button.disabled = false;
    $('trimAnalysisStatus').textContent = error.message;
  }
}

function renderAll() {
  renderViewControls();
  renderHeader();
  renderPhaseStory();
  setFrame(currentFrame, {redrawFeature: false});
  renderSelectedFeature();
}

function renderViewControls() {
  const views = result?.case_views?.views || [];
  const panel = $('viewPanel');
  if (!views.length || views.length <= 1) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $('viewSelect').innerHTML = views.map((view) => (
    '<option value="' + escapeHtml(view.slug) + '">' + escapeHtml(view.view_label || view.view_id) + '</option>'
  )).join('');
  $('viewSelect').value = result.case_views.current_view_slug || result.case?.slug || caseSlug;
  $('viewSelect').onchange = () => {
    const nextSlug = $('viewSelect').value;
    if (nextSlug && nextSlug !== caseSlug) {
      window.location.href = '/results?case=' + encodeURIComponent(nextSlug);
    }
  };
  $('viewCount').textContent = views.length + ' views available | current: '
    + (result.view?.perspective || 'unknown') + ' perspective';
  $('synthesisButton').onclick = () => {
    const synthesis = $('caseSynthesisPanel');
    synthesis.hidden = !synthesis.hidden;
  };
  $('caseSynthesisPanel').innerHTML = caseSynthesisHtml();
}

function caseSynthesisHtml() {
  const synthesis = result?.case_synthesis || {};
  const views = result?.case_views?.views || [];
  const viewRows = views.map((view) => (
    '<tr><td><strong>' + escapeHtml(view.view_label || view.view_id) + '</strong>'
    + (view.primary_view ? '<br /><span class="status">Primary</span>' : '')
    + '</td><td>' + escapeHtml(view.perspective || 'unknown')
    + '</td><td>' + escapeHtml(view.results_available ? 'Analysed' : 'Not analysed yet')
    + '</td><td><a href="/results?case=' + encodeURIComponent(view.slug) + '">Open view</a></td></tr>'
  )).join('');
  if (!synthesis.available) {
    return '<h3>Multi-view movement synthesis</h3>'
      + '<p class="subtle">' + escapeHtml(synthesis.reason || 'Synthesis is unavailable.') + '</p>'
      + '<table class="compact-table"><thead><tr><th>View</th><th>Perspective</th><th>Status</th><th>Inspect</th></tr></thead><tbody>'
      + viewRows + '</tbody></table>';
  }
  const featureRows = (synthesis.feature_sources || []).slice(0, 8).map((item) => {
    const best = item.automatic_preferred_view || {};
    return '<tr><td><strong>' + escapeHtml(item.display_label || item.feature_name) + '</strong>'
      + '<br />' + escapeHtml(item.body_region || '')
      + '</td><td>' + escapeHtml(best.view_label || best.view_id || '')
      + '<br />' + escapeHtml(best.view_perspective || 'unknown')
      + '</td><td>' + percent(best.geometry_completeness)
      + ' geometry<br />' + percent(best.dynamic_completeness) + ' dynamic'
      + '</td><td>' + escapeHtml(item.cross_view_status || '')
      + '<br /><a href="' + escapeHtml(best.open_view_url || '#') + '">Open supporting view</a></td></tr>';
  }).join('');
  const statusLine = (synthesis.corroborated || []).length + ' corroborated feature(s), '
    + (synthesis.disagreements || []).length + ' disagreement(s).';
  return '<h3>Multi-view movement synthesis</h3>'
    + '<p class="subtle">Case-level synthesis selects supporting views feature by feature. It does not average projected angles or pool descriptive statistics across camera views.</p>'
    + '<p><span class="status">' + escapeHtml(statusLine) + '</span></p>'
    + '<table class="compact-table"><thead><tr><th>Feature</th><th>Preferred evidence view</th><th>Support</th><th>Status</th></tr></thead><tbody>'
    + featureRows + '</tbody></table>'
    + '<details class="advanced-details"><summary>View list and alignment note</summary>'
    + '<p class="subtle">' + escapeHtml(synthesis.alignment?.note || '') + '</p>'
    + '<table class="compact-table"><thead><tr><th>View</th><th>Perspective</th><th>Status</th><th>Inspect</th></tr></thead><tbody>'
    + viewRows + '</tbody></table></details>';
}

function renderPhaseStory() {
  const story = result?.movement_story || {};
  const phases = story.phases || [];
  const panel = $('phaseStoryPanel');
  if (!phases.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $('phaseStorySummary').innerHTML = '<span class="status">' + escapeHtml(story.status || 'SUPPORTED') + '</span> '
    + phases.length + ' evidence-backed phase' + (phases.length === 1 ? '' : 's') + ' across source frames '
    + result.movement_window.movement_start_frame + '-' + result.movement_window.movement_end_frame + '.';
  $('wholeMovementSummary').innerHTML = '<h3>Whole Movement Summary</h3><p>'
    + escapeHtml(wholeMovementSummary(story, phases)) + '</p>';
  $('phaseStoryGrid').innerHTML = phases.map((phase) => {
    const evidence = phase.evidence_summary?.evidence_status || 'EVIDENCE';
    const drivers = phaseDrivers(phase)
      .filter((driver) => !['movement_timing', 'movement_path'].includes(driver.key))
      .slice(0, 4);
    const visualStory = visualStoryForPhase(phase.phase_id);
    const snapshots = phaseSalientFrames(phase, visualStory);
    return '<article class="phase-card">'
      + '<span>Phase ' + phase.phase_index + ' | frames ' + phase.start_frame + '-' + phase.end_frame + '</span>'
      + '<h3>' + escapeHtml(phase.title) + '</h3>'
      + '<p><strong class="status">' + escapeHtml(evidence) + '</strong></p>'
      + '<div class="phase-explanation"><strong>What happened?</strong><p>'
      + escapeHtml(phaseWhatHappened(phase, drivers)) + '</p></div>'
      + '<div class="phase-explanation"><strong>What defines this phase?</strong><p>'
      + escapeHtml(phaseBoundaryExplanation(phase, drivers)) + '</p></div>'
      + '<div class="phase-explanation"><strong>Main movement changes</strong>'
      + phaseChangesHtml(phase, drivers, visualStory) + '</div>'
      + phaseSnapshotsHtml(phase, snapshots)
      + '</article>';
  }).join('');
  [...document.querySelectorAll('[data-phase-frame]')].forEach((button) => {
    button.onclick = () => setFrame(Number(button.dataset.phaseFrame));
  });
  requestAnimationFrame(drawPhaseStoryVisuals);
}

function wholeMovementSummary(story, phases) {
  if (!phases.length) return 'A whole-movement description is unavailable for this case.';
  const first = phases[0];
  const last = phases[phases.length - 1];
  const sequence = phases.length === 1
    ? 'The observable sequence remained within one supported phase: ' + first.title.toLowerCase() + '.'
    : 'The observable sequence developed across ' + phases.length + ' supported phases, beginning with '
      + first.title.toLowerCase() + ' and ending with ' + last.title.toLowerCase() + '.';
  if (phases.length === 1) return sequence;
  const transition = phases[1];
  const categories = naturalList(phaseDrivers(transition)
    .filter((driver) => !['movement_timing', 'movement_path'].includes(driver.key))
    .slice(0, 3)
    .map((driver) => storyCategoryPlainLabel(driver.key)));
  const change = categories
    ? 'The clearest phase transition occurs near source frame ' + transition.start_frame + ', where ' + categories + ' change together.'
    : 'The clearest phase transition occurs near source frame ' + transition.start_frame + '.';
  return sequence + ' ' + change;
}

function phaseWhatHappened(phase, drivers) {
  const actions = drivers.slice(0, 3).map((driver) => storyCategoryAction(driver.key)).filter(Boolean);
  if (!actions.length) {
    return 'The supported projected movement followed a consistent pattern through this interval.';
  }
  return 'During this interval, ' + naturalList(actions) + '.';
}

function phaseBoundaryExplanation(phase, drivers) {
  const categories = naturalList(drivers.slice(0, 3).map((driver) => storyCategoryPlainLabel(driver.key)));
  if (Number(phase.phase_index) === 1) {
    return 'This is the opening continuous interval before the next supported multidimensional movement change'
      + (categories ? ', led by ' + categories : '') + '.';
  }
  return 'This phase begins at source frame ' + phase.start_frame
    + ', where the supported movement-change pattern separates from the preceding interval'
    + (categories ? ', led by ' + categories : '') + '.';
}

function phaseChangesHtml(phase, drivers, visualStory) {
  if (!drivers.length) return '<p>No supported movement family is available for this phase.</p>';
  return '<div class="story-change-grid">' + drivers.map((driver) => {
    const support = storyObservationSupport(visualStory, driver.key);
    const status = support?.evidence_status || driver.status || 'EVIDENCE';
    const fraction = support?.support?.supported_fraction;
    const percentage = Number.isFinite(Number(fraction)) ? ' · ' + Math.round(Number(fraction) * 100) + '%' : '';
    const hasVisual = phaseVisualAvailable(visualStory, driver.key);
    return '<div class="story-change">'
      + '<div class="story-change-heading"><strong>' + escapeHtml(storyCategoryPlainLabel(driver.key)) + '</strong>'
      + '<span class="status">' + escapeHtml(status) + escapeHtml(percentage) + '</span></div>'
      + '<p>' + escapeHtml(storyCategoryChangeSummary(driver.key, phase)) + '</p>'
      + (hasVisual ? '<canvas class="phase-mini-visual" data-phase-mini="' + escapeHtml(phase.phase_id)
        + '" data-story-category="' + escapeHtml(driver.key) + '" width="360" height="132"></canvas>' : '')
      + phaseMeasurementSummaryHtml(phase, driver.key)
      + '</div>';
  }).join('') + '</div>';
}

function phaseSnapshotsHtml(phase, snapshots) {
  if (!snapshots.length) return '';
  const hasChangeColour = snapshots.some((snapshot) => snapshot.change_intensity);
  const cards = snapshots.map((snapshot) => (
    '<button class="phase-snapshot' + (snapshot.change_intensity ? ' change-' + snapshot.change_intensity : '')
    + '" type="button" data-phase-frame="' + snapshot.source_frame_index + '">'
    + '<img alt="' + escapeHtml(snapshot.label) + ' source frame ' + snapshot.source_frame_index
    + '" src="' + frameUrl(snapshot.source_frame_index, true) + '" />'
    + '<span><strong>' + escapeHtml(snapshot.label) + '</strong><br />source frame '
    + snapshot.source_frame_index + '</span></button>'
  )).join('');
  return '<div class="phase-explanation"><strong>Important frames</strong><div class="phase-snapshots">'
    + cards + '</div>'
    + (hasChangeColour
      ? '<p class="movement-change-legend">Green/yellow/red indicates relative movement-change magnitude only, not evidence quality, severity, or risk.</p>'
      : '')
    + '</div>';
}

function phaseSalientFrames(phase, visualStory) {
  const snapshots = visualStory?.snapshot_frames || [];
  const chosen = snapshots.filter((snapshot) => (
    snapshot.change_intensity
    || ['Phase start', 'Phase end'].includes(snapshot.label)
    || !['25%', '50%', '75%', 'Mid-phase'].includes(snapshot.label)
  ));
  const fallback = [
    {label: 'Phase start', source_frame_index: phase.start_frame},
    {label: 'Phase end', source_frame_index: phase.end_frame},
  ];
  const unique = [];
  [...chosen, ...fallback].forEach((snapshot) => {
    const frame = Number(snapshot.source_frame_index);
    if (!Number.isFinite(frame) || unique.some((item) => item.source_frame_index === frame)) return;
    unique.push({...snapshot, source_frame_index: frame});
  });
  return unique.sort((a, b) => a.source_frame_index - b.source_frame_index);
}

function visualStoryForPhase(phaseId) {
  return (result?.movement_visual_story?.phases || []).find((item) => item.phase_id === phaseId) || null;
}

function storyObservationSupport(visualStory, category) {
  return (visualStory?.observations || []).find((item) => item.category === category) || null;
}

function phaseVisualAvailable(visualStory, category) {
  if (!visualStory) return false;
  if (category === 'bilateral_limb_relationship') {
    const phase = phaseForStory(visualStory);
    return phase !== null
      && phaseMetricPoints('injured_hka_angle_2d_deg', phase).some((point) => point.value !== null)
      && phaseMetricPoints('contralateral_hka_angle_2d_deg', phase).some((point) => point.value !== null);
  }
  if (category === 'hip_knee_ankle_chain') {
    const phase = phaseForStory(visualStory);
    const stat = phaseMetricStat(phase, 'injured_hka_angle_2d_deg');
    return [stat?.start_value, stat?.end_value].every((value) => Number.isFinite(Number(value)));
  }
  const phase = phaseForStory(visualStory);
  if (category === 'trunk_pelvis') return canonicalTorsoState(phase, 'start_value', 360, 132) !== null
    && canonicalTorsoState(phase, 'end_value', 360, 132) !== null;
  if (category === 'upper_body') return upperBodySide(phase) !== null;
  return false;
}

function storyCategoryPlainLabel(category) {
  return {
    movement_path: 'Movement path',
    hip_knee_ankle_chain: 'Injured-side hip–knee–ankle configuration',
    bilateral_limb_relationship: 'Injured vs opposite-side HKA',
    trunk_pelvis: 'Trunk & pelvis',
    upper_body: 'Upper body',
    movement_timing: 'Movement timing',
  }[category] || categoryLabel(category);
}

function storyCategoryAction(category) {
  return {
    movement_path: 'the on-screen travel direction and speed relative to body size changed',
    hip_knee_ankle_chain: 'the visible hip–knee–ankle configuration changed',
    bilateral_limb_relationship: 'the projected relationship between the two lower limbs changed',
    trunk_pelvis: 'the trunk, shoulder, and hip/pelvis lines reoriented',
    upper_body: 'the projected arm positions changed',
    movement_timing: 'the timing of the supported movement changed',
  }[category] || '';
}

function storyCategoryChangeSummary(category, phase) {
  if (category === 'hip_knee_ankle_chain') {
    const stat = phaseMetricStat(phase, 'injured_hka_angle_2d_deg');
    const change = Number(stat?.change);
    if (!Number.isFinite(change)) {
      return 'The supported injured-side projected hip-knee-ankle configuration is shown for this phase.';
    }
    if (Math.abs(change) < 8) {
      return 'The injured-side projected hip-knee-ankle configuration remained relatively stable through the phase.';
    }
    const magnitude = Math.abs(change) >= 20 ? ' substantially' : '';
    return change > 0
      ? 'The injured-side projected hip-knee-ankle configuration opened' + magnitude + ' through the phase.'
      : 'The injured-side projected hip-knee-ankle configuration became'
        + (magnitude ? ' substantially' : '') + ' more closed through the phase.';
  }
  if (category === 'trunk_pelvis') {
    const changes = torsoMetricDefinitions().map((item) => (
      Math.abs(Number(phaseMetricStat(phase, item.metric)?.change))
    )).filter(Number.isFinite);
    if (changes.length && Math.max(...changes) < 8) {
      return 'The projected trunk, shoulder, and hip lines remained relatively stable through the phase.';
    }
    return 'The projected trunk, shoulder, and hip lines reoriented through the phase.';
  }
  if (category === 'upper_body') {
    const side = upperBodySide(phase);
    if (!side) return 'Supported projected arm geometry is shown for this phase.';
    const changes = upperBodyMetricDefinitions(side).map((item) => (
      Math.abs(Number(phaseMetricStat(phase, item.metric)?.change))
    )).filter(Number.isFinite);
    if (changes.length && Math.max(...changes) < 8) {
      return 'The ' + side + ' projected arm configuration remained relatively stable through the phase.';
    }
    return 'The ' + side + ' projected arm orientation and elbow configuration changed through the phase.';
  }
  return {
    movement_path: 'On-screen travel direction changed while speed relative to body size also shifted.',
    bilateral_limb_relationship: 'The injured and opposite-side projected hip–knee–ankle angles are compared through the phase.',
    movement_timing: 'The supported interval defines when this movement pattern was visible.',
  }[category] || 'This supported movement component changed through the phase.';
}

function naturalList(items) {
  const values = items.filter(Boolean);
  if (!values.length) return '';
  if (values.length === 1) return values[0];
  if (values.length === 2) return values[0] + ' and ' + values[1];
  return values.slice(0, -1).join(', ') + ', and ' + values[values.length - 1];
}

function drawPhaseStoryVisuals() {
  [...document.querySelectorAll('[data-phase-mini]')].forEach((canvas) => {
    const visualStory = visualStoryForPhase(canvas.dataset.phaseMini);
    drawPhaseStoryVisual(canvas, visualStory, canvas.dataset.storyCategory);
  });
}

function drawPhaseStoryVisual(canvas, visualStory, category) {
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(220, canvas.clientWidth || 360);
  const height = 132;
  canvas.width = Math.floor(width * scale);
  canvas.height = Math.floor(height * scale);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = '#f4f7fa';
  ctx.fillRect(0, 0, width, height);
  if (category === 'bilateral_limb_relationship') {
    drawPhaseBilateralMini(ctx, width, height, visualStory);
    return;
  }
  if (category === 'hip_knee_ankle_chain') {
    drawPhaseInjuredHkaMini(ctx, width, height, visualStory);
    return;
  }
  if (category === 'trunk_pelvis') {
    drawPhaseTorsoMini(ctx, width, height, visualStory);
    return;
  }
  if (category === 'upper_body') {
    drawPhaseUpperBodyMini(ctx, width, height, visualStory);
    return;
  }
  drawPhasePoseMini(ctx, width, height, visualStory, category);
}

function phaseForStory(visualStory) {
  return (result?.movement_story?.phases || []).find((phase) => (
    phase.phase_id === visualStory?.phase_id
  )) || null;
}

function phaseMetricPoints(metric, phase) {
  if (!phase) return [];
  return supportedSeries(metric).filter((point) => (
    point.frame >= Number(phase.start_frame) && point.frame <= Number(phase.end_frame)
  ));
}

function phaseMeasurementSummaryHtml(phase, category) {
  if (category === 'hip_knee_ankle_chain') {
    const stat = phaseMetricStat(phase, 'injured_hka_angle_2d_deg');
    if (![stat?.start_value, stat?.end_value, stat?.change].every((value) => Number.isFinite(Number(value)))) {
      return '<div class="phase-mini-values"><span>Projected HKA angle</span><strong>Endpoint unavailable</strong></div>';
    }
    return '<div class="phase-mini-values">'
      + '<span>Projected HKA angle</span><strong>' + formatDegrees(stat.start_value) + ' &rarr; '
      + formatDegrees(stat.end_value) + '</strong>'
      + '<span class="phase-mini-delta">Change ' + formatSignedDegrees(stat.change) + '</span>'
      + '</div>';
  }
  if (category === 'trunk_pelvis') {
    return phaseMetricRowsHtml(phase, torsoMetricDefinitions());
  }
  if (category === 'upper_body') {
    const side = upperBodySide(phase);
    return side ? phaseMetricRowsHtml(phase, upperBodyMetricDefinitions(side)) : '';
  }
  return '';
}

function phaseMetricRowsHtml(phase, definitions) {
  const rows = definitions.map((item) => ({...item, stat: phaseMetricStat(phase, item.metric)}))
    .filter((item) => [item.stat?.start_value, item.stat?.end_value, item.stat?.change]
      .every((value) => Number.isFinite(Number(value))));
  if (!rows.length) return '';
  return '<div class="phase-mini-values multi">' + rows.map((item) => (
    '<span>' + escapeHtml(item.label) + '</span><strong>'
    + formatDegrees(item.stat.start_value) + ' &rarr; ' + formatDegrees(item.stat.end_value)
    + ' &middot; Change ' + formatSignedDegrees(item.stat.change) + '</strong>'
  )).join('') + '</div>';
}

function torsoMetricDefinitions() {
  return [
    {metric: 'projected_trunk_axis_angle_deg', label: 'Trunk axis'},
    {metric: 'projected_hip_line_angle_deg', label: 'Hip line'},
    {metric: 'projected_shoulder_line_angle_deg', label: 'Shoulder line'},
    {
      metric: 'projected_shoulder_pelvis_orientation_difference_deg',
      label: 'Shoulder-hip relationship',
    },
  ];
}

function upperBodyMetricDefinitions(side) {
  const label = side.charAt(0).toUpperCase() + side.slice(1);
  return [
    {metric: `${side}_elbow_angle_2d_deg`, label: label + ' elbow angle'},
    {metric: `${side}_upper_arm_orientation_2d_deg`, label: label + ' upper-arm orientation'},
  ];
}

function upperBodySide(phase) {
  const candidates = ['left', 'right'].map((side) => {
    const stats = upperBodyMetricDefinitions(side).map((item) => phaseMetricStat(phase, item.metric));
    const valid = stats.every((stat) => [stat?.start_value, stat?.end_value, stat?.change]
      .every((value) => Number.isFinite(Number(value))));
    return {
      side,
      valid,
      magnitude: valid ? Math.max(...stats.map((stat) => Math.abs(Number(stat.change)))) : -1,
    };
  }).filter((item) => item.valid).sort((a, b) => b.magnitude - a.magnitude);
  return candidates[0]?.side || null;
}

function formatDegrees(value) {
  return Number(value).toFixed(1) + '&deg;';
}

function formatSignedDegrees(value) {
  const number = Number(value);
  return (number > 0 ? '+' : '') + number.toFixed(1) + '&deg;';
}

function injuredSide() {
  const side = String(result?.movement_visual_story?.laterality_mapping?.injured || '').toLowerCase();
  return ['left', 'right'].includes(side) ? side : 'right';
}

function drawPhaseInjuredHkaMini(ctx, width, height, visualStory) {
  const phase = phaseForStory(visualStory);
  const stat = phaseMetricStat(phase, 'injured_hka_angle_2d_deg');
  const side = injuredSide();
  if (![stat?.start_value, stat?.end_value].every((value) => Number.isFinite(Number(value)))) return;
  const start = canonicalHkaState(Number(stat.start_value), side, width, height);
  const end = canonicalHkaState(Number(stat.end_value), side, width, height);
  drawArticulatedState(ctx, start, '#215f9a', true, 2);
  drawArticulatedState(ctx, end, '#176d4d', false, 3);
  drawAngleArc(ctx, end[1], end[0], end[2], '#176d4d');
  drawChangeArrow(ctx, start[2], end[2]);
  drawJointLabels(ctx, end, ['H', 'K', 'A']);
  ctx.fillStyle = '#627181';
  ctx.font = '10px sans-serif';
  ctx.fillText(side.charAt(0).toUpperCase() + side.slice(1) + ' injured limb', 9, 13);
}

function canonicalHkaState(angleDegrees, side, width, height) {
  const direction = side === 'left' ? -1 : 1;
  const knee = {
    x: width * (side === 'left' ? 0.56 : 0.44),
    y: height * 0.48,
  };
  const hipLength = Math.min(42, height * 0.32);
  const ankleLength = Math.min(46, height * 0.35);
  const hip = {x: knee.x, y: knee.y - hipLength};
  const ankleAngle = -Math.PI / 2 + direction * Number(angleDegrees) * Math.PI / 180;
  const ankle = {
    x: knee.x + Math.cos(ankleAngle) * ankleLength,
    y: knee.y + Math.sin(ankleAngle) * ankleLength,
  };
  return [hip, knee, ankle];
}

function orientationVector(angleDegrees, length) {
  const angle = Number(angleDegrees) * Math.PI / 180;
  return {x: Math.cos(angle) * length, y: -Math.sin(angle) * length};
}

function canonicalTorsoState(phase, endpoint, width, height) {
  const trunk = phaseMetricStat(phase, 'projected_trunk_axis_angle_deg');
  const hipLine = phaseMetricStat(phase, 'projected_hip_line_angle_deg');
  const shoulderLine = phaseMetricStat(phase, 'projected_shoulder_line_angle_deg');
  if (![trunk?.[endpoint], hipLine?.[endpoint], shoulderLine?.[endpoint]]
    .every((value) => Number.isFinite(Number(value)))) return null;
  const pelvisMid = {x: width * 0.50, y: height * 0.68};
  const trunkVector = orientationVector(trunk[endpoint], Math.min(43, height * 0.34));
  const shoulderMid = {x: pelvisMid.x + trunkVector.x, y: pelvisMid.y + trunkVector.y};
  const hipVector = orientationVector(hipLine[endpoint], Math.min(31, width * 0.10));
  const shoulderVector = orientationVector(shoulderLine[endpoint], Math.min(35, width * 0.11));
  return {
    pelvisMid,
    shoulderMid,
    leftHip: {x: pelvisMid.x - hipVector.x, y: pelvisMid.y - hipVector.y},
    rightHip: {x: pelvisMid.x + hipVector.x, y: pelvisMid.y + hipVector.y},
    leftShoulder: {x: shoulderMid.x - shoulderVector.x, y: shoulderMid.y - shoulderVector.y},
    rightShoulder: {x: shoulderMid.x + shoulderVector.x, y: shoulderMid.y + shoulderVector.y},
  };
}

function drawPhaseTorsoMini(ctx, width, height, visualStory) {
  const phase = phaseForStory(visualStory);
  const start = canonicalTorsoState(phase, 'start_value', width, height);
  const end = canonicalTorsoState(phase, 'end_value', width, height);
  if (!start || !end) return;
  drawTorsoState(ctx, start, '#215f9a', true, 2);
  drawTorsoState(ctx, end, '#176d4d', false, 3);
  const candidates = [
    [start.shoulderMid, end.shoulderMid],
    [start.rightShoulder, end.rightShoulder],
    [start.rightHip, end.rightHip],
  ].sort((a, b) => pointDistance(b[0], b[1]) - pointDistance(a[0], a[1]));
  drawChangeArrow(ctx, candidates[0][0], candidates[0][1]);
  ctx.fillStyle = '#425466';
  ctx.font = '9px sans-serif';
  ctx.fillText('S shoulder · T trunk · H hip', 9, 13);
  ctx.font = 'bold 9px sans-serif';
  ctx.fillText('S', end.rightShoulder.x + 5, end.rightShoulder.y - 4);
  ctx.fillText('T', end.shoulderMid.x + 5, (end.pelvisMid.y + end.shoulderMid.y) / 2);
  ctx.fillText('H', end.rightHip.x + 5, end.rightHip.y + 10);
}

function drawTorsoState(ctx, state, color, dashed, lineWidth) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dashed ? [6, 5] : []);
  ctx.lineCap = 'round';
  [
    [state.leftShoulder, state.rightShoulder],
    [state.leftHip, state.rightHip],
    [state.pelvisMid, state.shoulderMid],
  ].forEach(([first, second]) => {
    ctx.beginPath();
    ctx.moveTo(first.x, first.y);
    ctx.lineTo(second.x, second.y);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  [state.leftShoulder, state.rightShoulder, state.leftHip, state.rightHip].forEach((point) => {
    ctx.fillStyle = '#f4f7fa';
    ctx.beginPath();
    ctx.arc(point.x, point.y, dashed ? 2.5 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.stroke();
  });
  ctx.restore();
}

function canonicalArmState(phase, endpoint, side, width, height) {
  const elbowStat = phaseMetricStat(phase, `${side}_elbow_angle_2d_deg`);
  const upperArmStat = phaseMetricStat(phase, `${side}_upper_arm_orientation_2d_deg`);
  if (![elbowStat?.[endpoint], upperArmStat?.[endpoint]]
    .every((value) => Number.isFinite(Number(value)))) return null;
  const shoulder = {x: width * 0.50, y: height * 0.48};
  const upperLength = Math.min(36, height * 0.28);
  const forearmLength = Math.min(40, height * 0.31);
  const upperVector = orientationVector(upperArmStat[endpoint], upperLength);
  const elbow = {x: shoulder.x + upperVector.x, y: shoulder.y + upperVector.y};
  const direction = side === 'left' ? -1 : 1;
  const wristAngle = Number(upperArmStat[endpoint]) + 180
    + direction * Number(elbowStat[endpoint]);
  const forearmVector = orientationVector(wristAngle, forearmLength);
  const wrist = {x: elbow.x + forearmVector.x, y: elbow.y + forearmVector.y};
  return [shoulder, elbow, wrist];
}

function drawPhaseUpperBodyMini(ctx, width, height, visualStory) {
  const phase = phaseForStory(visualStory);
  const side = upperBodySide(phase);
  if (!side) return;
  const start = canonicalArmState(phase, 'start_value', side, width, height);
  const end = canonicalArmState(phase, 'end_value', side, width, height);
  if (!start || !end) return;
  drawArticulatedState(ctx, start, '#215f9a', true, 2);
  drawArticulatedState(ctx, end, '#176d4d', false, 3);
  drawAngleArc(ctx, end[1], end[0], end[2], '#176d4d');
  const movingJoint = pointDistance(start[2], end[2]) >= pointDistance(start[1], end[1]) ? 2 : 1;
  drawChangeArrow(ctx, start[movingJoint], end[movingJoint]);
  drawJointLabels(ctx, end, ['S', 'E', 'W']);
  ctx.fillStyle = '#627181';
  ctx.font = '10px sans-serif';
  ctx.fillText(side.charAt(0).toUpperCase() + side.slice(1) + ' arm', 9, 13);
}

function pointDistance(first, second) {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function drawArticulatedState(ctx, points, color, dashed, lineWidth) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dashed ? [6, 5] : []);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  points.forEach((point, index) => (
    index === 0 ? ctx.moveTo(point.x, point.y) : ctx.lineTo(point.x, point.y)
  ));
  ctx.stroke();
  ctx.setLineDash([]);
  points.forEach((point) => {
    ctx.fillStyle = '#f4f7fa';
    ctx.beginPath();
    ctx.arc(point.x, point.y, dashed ? 3 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = dashed ? 1.5 : 2;
    ctx.stroke();
  });
  ctx.restore();
}

function drawJointLabels(ctx, points, labels) {
  ctx.save();
  ctx.fillStyle = '#425466';
  ctx.font = 'bold 9px sans-serif';
  points.forEach((point, index) => ctx.fillText(labels[index], point.x + 6, point.y - 5));
  ctx.restore();
}

function drawAngleArc(ctx, vertex, first, second, color) {
  const start = Math.atan2(first.y - vertex.y, first.x - vertex.x);
  const finish = Math.atan2(second.y - vertex.y, second.x - vertex.x);
  let delta = finish - start;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  const radius = 15;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(vertex.x, vertex.y, radius, start, start + delta, delta < 0);
  ctx.stroke();
  ctx.restore();
}

function drawChangeArrow(ctx, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length < 10) return;
  const ux = dx / length;
  const uy = dy / length;
  const arrowStart = {x: start.x + ux * 5, y: start.y + uy * 5};
  const arrowEnd = {x: end.x - ux * 7, y: end.y - uy * 7};
  ctx.save();
  ctx.strokeStyle = '#627181';
  ctx.fillStyle = '#627181';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(arrowStart.x, arrowStart.y);
  ctx.lineTo(arrowEnd.x, arrowEnd.y);
  ctx.stroke();
  const angle = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(arrowEnd.x, arrowEnd.y);
  ctx.lineTo(arrowEnd.x - 7 * Math.cos(angle - Math.PI / 6), arrowEnd.y - 7 * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(arrowEnd.x - 7 * Math.cos(angle + Math.PI / 6), arrowEnd.y - 7 * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawPhaseBilateralMini(ctx, width, height, visualStory) {
  const phase = phaseForStory(visualStory);
  const injured = phaseMetricPoints('injured_hka_angle_2d_deg', phase);
  const opposite = phaseMetricPoints('contralateral_hka_angle_2d_deg', phase);
  const values = [...injured, ...opposite]
    .filter((point) => point.value !== null)
    .map((point) => Number(point.value));
  if (!phase || values.length < 2) return;
  const mapping = result?.movement_visual_story?.laterality_mapping || {};
  const bilateral = phase?.category_summaries?.bilateral_limb_relationship?.metrics || {};
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const left = 12;
  const top = 30;
  const plotWidth = width - 24;
  const plotHeight = height - 50;
  const project = (point) => ({
    x: left + ((point.frame - Number(phase.start_frame))
      / Math.max(1, Number(phase.end_frame) - Number(phase.start_frame))) * plotWidth,
    y: top + (1 - ((Number(point.value) - minValue) / Math.max(1, maxValue - minValue))) * plotHeight,
  });
  drawPhaseMetricLine(ctx, injured, project, '#215f9a');
  drawPhaseMetricLine(ctx, opposite, project, '#176d4d');
  [injured, opposite].forEach((series, index) => {
    const finite = series.filter((point) => point.value !== null);
    if (!finite.length) return;
    drawTrajectoryPoint(ctx, project(finite[0]), index === 0 ? '#215f9a' : '#176d4d', 2.5);
    drawTrajectoryPoint(ctx, project(finite[finite.length - 1]), index === 0 ? '#215f9a' : '#176d4d', 2.5);
  });
  const peakFrame = Number(bilateral.source_frame_of_maximum);
  const injuredPeak = injured.find((point) => point.frame === peakFrame && point.value !== null);
  const oppositePeak = opposite.find((point) => point.frame === peakFrame && point.value !== null);
  if (injuredPeak && oppositePeak) {
    const firstPeak = project(injuredPeak);
    const secondPeak = project(oppositePeak);
    ctx.save();
    ctx.strokeStyle = '#627181';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(firstPeak.x, firstPeak.y);
    ctx.lineTo(secondPeak.x, secondPeak.y);
    ctx.stroke();
    ctx.restore();
    drawTrajectoryPoint(ctx, firstPeak, '#215f9a', 3.5);
    drawTrajectoryPoint(ctx, secondPeak, '#176d4d', 3.5);
  }
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#215f9a';
  ctx.fillText('Injured limb (' + (mapping.injured || 'side') + ')', 12, 13);
  ctx.fillStyle = '#176d4d';
  ctx.fillText('Opposite limb (' + (mapping.contralateral || 'side') + ')', Math.max(width / 2, 112), 13);
  ctx.fillStyle = '#627181';
  ctx.fillText('HKA degrees', 12, 25);
  ctx.fillText('Phase start', 12, height - 5);
  const endLabel = 'Phase end';
  ctx.fillText(endLabel, width - ctx.measureText(endLabel).width - 12, height - 5);
  if (Number.isFinite(Number(bilateral.maximum_absolute_hka_difference_deg))) {
    const callout = (width < 320 ? 'Peak ' : 'Peak gap ')
      + Number(bilateral.maximum_absolute_hka_difference_deg).toFixed(1) + '° · f' + peakFrame;
    ctx.fillText(callout, Math.max(70, (width - ctx.measureText(callout).width) / 2), height - 5);
  }
}

function drawTrajectoryPoint(ctx, point, color, radius) {
  ctx.fillStyle = '#f4f7fa';
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawPhaseMetricLine(ctx, points, project, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.25;
  let drawing = false;
  let previousFrame = null;
  ctx.beginPath();
  points.forEach((point) => {
    if (point.value === null) {
      drawing = false;
      previousFrame = null;
      return;
    }
    const item = project(point);
    if (!drawing || point.frame !== previousFrame + 1) ctx.moveTo(item.x, item.y);
    else ctx.lineTo(item.x, item.y);
    drawing = true;
    previousFrame = point.frame;
  });
  ctx.stroke();
}

function drawPhasePathMini(ctx, width, height, visualStory) {
  const points = (visualStory?.visuals || []).find((item) => item.visual_type === 'projected_path')?.points || [];
  if (points.length < 2) return;
  const xs = points.map((point) => Number(point.x));
  const ys = points.map((point) => Number(point.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const project = (point) => ({
    x: 16 + ((Number(point.x) - minX) / Math.max(1e-9, maxX - minX)) * (width - 32),
    y: 14 + ((Number(point.y) - minY) / Math.max(1e-9, maxY - minY)) * (height - 28),
  });
  ctx.strokeStyle = '#215f9a';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach((point, index) => {
    const item = project(point);
    if (index === 0) ctx.moveTo(item.x, item.y);
    else ctx.lineTo(item.x, item.y);
  });
  ctx.stroke();
  [points[0], points[points.length - 1]].forEach((point, index) => {
    const item = project(point);
    ctx.fillStyle = index === 0 ? '#215f9a' : '#176d4d';
    ctx.beginPath();
    ctx.arc(item.x, item.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPhasePoseMini(ctx, width, height, visualStory, category) {
  const snapshots = visualStory?.snapshot_frames || [];
  const first = snapshots[0]?.landmarks || {};
  const last = snapshots[snapshots.length - 1]?.landmarks || {};
  drawMiniPoseHalf(ctx, first, {x: 8, y: 8, width: width / 2 - 12, height: height - 16}, category, '#215f9a');
  drawMiniPoseHalf(ctx, last, {x: width / 2 + 4, y: 8, width: width / 2 - 12, height: height - 16}, category, '#176d4d');
  ctx.fillStyle = '#627181';
  ctx.font = '11px sans-serif';
  ctx.fillText('Start', 12, 14);
  ctx.fillText('End', width / 2 + 8, 14);
}

function drawMiniPoseHalf(ctx, landmarks, rect, category, color) {
  const names = category === 'upper_body'
    ? ['left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
    : category === 'trunk_pelvis'
      ? ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
      : ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle'];
  const points = names.map((name) => landmarks[name]).filter((point) => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
  if (!points.length) return;
  const minX = Math.min(...points.map((point) => Number(point.x)));
  const maxX = Math.max(...points.map((point) => Number(point.x)));
  const minY = Math.min(...points.map((point) => Number(point.y)));
  const maxY = Math.max(...points.map((point) => Number(point.y)));
  const project = (name) => {
    const point = landmarks[name];
    if (!Number.isFinite(Number(point?.x)) || !Number.isFinite(Number(point?.y))) return null;
    return {
      x: rect.x + 12 + ((Number(point.x) - minX) / Math.max(1, maxX - minX)) * (rect.width - 24),
      y: rect.y + 16 + ((Number(point.y) - minY) / Math.max(1, maxY - minY)) * (rect.height - 24),
    };
  };
  const chains = category === 'upper_body'
    ? [['left_shoulder', 'left_elbow', 'left_wrist'], ['right_shoulder', 'right_elbow', 'right_wrist']]
    : category === 'trunk_pelvis'
      ? [['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'], ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip']]
      : [['left_hip', 'left_knee', 'left_ankle'], ['right_hip', 'right_knee', 'right_ankle']];
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  chains.forEach((chain) => {
    const projected = chain.map(project);
    if (!projected.every(Boolean)) return;
    ctx.beginPath();
    projected.forEach((point, index) => index === 0 ? ctx.moveTo(point.x, point.y) : ctx.lineTo(point.x, point.y));
    ctx.stroke();
  });
}

function renderHeader() {
  const coverage = evidenceCoverage();
  const supported = coverage.supported_source_ranges?.[0];
  const supportedText = supported
    ? 'Supported measurements: frames ' + supported.start_frame + '-' + supported.end_frame
    : 'Supported measurements: unavailable';
  const movementSeconds = Number(result.header_metrics?.movement_duration_seconds ?? 0).toFixed(2);
  $('compactHeader').innerHTML = [
    result.case?.player_name || 'Christen Press',
    result.target_annotation?.label || 'Human verified',
    'Movement ' + movementSeconds + ' s',
    supportedText
  ].map((item) => '<span>' + escapeHtml(item) + '</span>').join('');
  const post = coverage.post_supported_frame_range;
  $('headerNote').textContent = post
    ? 'Frames ' + post.start_frame + '-' + post.end_frame + ' withheld because target identity is unreliable during overlap/occlusion.'
    : '';
}

function renderFeatureOptions() {
  const features = availableFeatures(FEATURE_CATEGORIES[selectedCategory] || []);
  const featureSelect = $('featureSelect');
  if (!features.length) {
    selectedFeatureId = null;
    featureSelect.disabled = true;
    featureSelect.innerHTML = '<option value="">No measurements available</option>';
    return;
  }
  featureSelect.disabled = false;
  $('featureSelect').innerHTML = features
    .map((feature) => '<option value="' + feature.id + '">' + escapeHtml(featureOptionLabel(feature)) + '</option>')
    .join('');
  if (!features.some((feature) => feature.id === selectedFeatureId)) {
    selectedFeatureId = features[0].id;
  }
  $('featureSelect').value = selectedFeatureId;
}

function renderSelectedFeature() {
  const feature = selectedFeature();
  if (!feature) {
    const stats = unavailableStats();
    $('featureTitle').textContent = 'Measurement unavailable';
    $('featureDescription').textContent = 'No supported measurements are available for this category in the current case.';
    $('featureTechnicalLabel').textContent = '';
    $('featureGraph').hidden = true;
    $('unavailableVisual').hidden = false;
    $('unavailableVisual').style.display = 'grid';
    $('unavailableVisual').textContent = 'No supported measurements are available for this category.';
    $('headlineValues').innerHTML = headlineHtml(stats);
    $('trajectoryInterpretation').textContent = 'No supported trajectory is available for interpretation.';
    $('filmstrip').innerHTML = '';
    $('moreStatisticsGrid').innerHTML = '';
    $('evidenceSummary').innerHTML = '<span class="status unavailable">Unavailable</span> This category has no supported feature data for the selected case.';
    $('technicalText').textContent = 'No selected feature.';
    renderOperatorAnalytics({label: 'Measurement unavailable', metrics: []}, stats);
    return;
  }
  const presentation = featurePresentation(feature);
  $('featureTitle').innerHTML = escapeHtml(presentation.name)
    + '<span class="info-icon" title="' + escapeHtml(presentation.limitation) + '" aria-label="Measurement information">ⓘ</span>';
  $('featureDescription').textContent = presentation.definition;
  $('featureTechnicalLabel').textContent = presentation.technical;

  if (feature.visual === 'unavailable') {
    $('featureGraph').hidden = true;
    $('unavailableVisual').hidden = false;
    $('unavailableVisual').style.display = 'grid';
    $('unavailableVisual').textContent = 'Movement path graph unavailable in this panel. Check the phase narrative for supported path evidence.';
    $('headlineValues').innerHTML = headlineHtml(unavailableStats());
    $('trajectoryInterpretation').textContent = 'A chart-level movement-path trajectory is unavailable in this measurement panel.';
    $('filmstrip').innerHTML = '';
    $('moreStatisticsGrid').innerHTML = statsGridHtml(feature, unavailableStats());
    $('evidenceSummary').innerHTML = '<span class="status unavailable">Unavailable</span> Graph-level movement path is held back in this simplified measurement panel.';
    $('technicalText').textContent = technicalDetails(feature, unavailableStats());
    renderOperatorAnalytics(feature, unavailableStats());
    return;
  }

  const stats = featureStats(feature);
  $('featureGraph').hidden = false;
  $('unavailableVisual').hidden = true;
  $('unavailableVisual').style.display = 'none';
  $('unavailableVisual').textContent = '';
  $('headlineValues').innerHTML = headlineHtml(stats);
  $('moreStatisticsGrid').innerHTML = statsGridHtml(feature, stats);
  $('evidenceSummary').innerHTML = evidenceHtml(stats);
  $('technicalText').textContent = technicalDetails(feature, stats);
  renderFilmstrip(feature, stats);
  drawFeatureGraph(feature, stats);
  $('trajectoryInterpretation').textContent = trajectoryInterpretation(feature, stats);
  renderOperatorAnalytics(feature, stats);
}

function selectedFeature() {
  return availableFeatures(FEATURE_CATEGORIES[selectedCategory] || [])
    .find((feature) => feature.id === selectedFeatureId) || firstAvailableFeature(FEATURE_CATEGORIES[selectedCategory] || []);
}

function featureOptionLabel(feature) {
  const presentation = featurePresentation(feature);
  const variants = {
    injured_hka: 'injured side',
    contralateral_hka: 'opposite side',
    injured_hka_rate_change: 'supported change',
    left_right_hka_pair: 'left / right',
    hka_pair: 'injured / opposite',
    hka_signed_difference: 'signed difference',
    hka_absolute_difference: 'absolute difference',
    bilateral_change: 'difference through time',
    left_right_knee_line: 'left / right',
    bilateral_knee_line: 'injured / opposite difference',
    elbow_pair: 'left / right',
    upper_arm_pair: 'left / right',
  };
  return presentation.name + (variants[feature.id] ? ' · ' + variants[feature.id] : '');
}

function featurePresentation(feature) {
  const presentations = {
    injured_hka: hkaPresentation('Projected injured-side HKA angle (°)'),
    contralateral_hka: hkaPresentation('Projected contralateral HKA angle (°)'),
    injured_hka_rate_change: hkaPresentation('Supported projected injured-side HKA change (°)'),
    left_right_hka_pair: hkaPresentation('Left/right projected HKA angle (°)'),
    hka_pair: hkaPresentation('Injured/contralateral projected HKA angle (°)'),
    hka_signed_difference: bilateralHkaPresentation('Projected injured-minus-contralateral HKA difference (°)'),
    hka_absolute_difference: bilateralHkaPresentation('Absolute projected HKA difference (°)'),
    bilateral_change: bilateralHkaPresentation('Projected bilateral HKA difference through time (°)'),
    knee_line_deviation: kneeRelationshipPresentation('Projected knee-line deviation · body-scale units'),
    left_right_knee_line: kneeRelationshipPresentation('Left/right projected knee-line deviation · body-scale units'),
    bilateral_knee_line: kneeRelationshipPresentation('Projected bilateral knee-line difference · body-scale units'),
    knee_ankle_offset: {
      name: 'Knee–ankle horizontal relationship',
      definition: 'How the knee\'s projected horizontal position changes relative to the ankle through the movement.',
      technical: 'Projected knee–ankle x-offset · body-scale units',
      limitation: 'A video-derived image-plane relationship. It is not a true three-dimensional joint displacement.',
    },
    trunk_orientation: {
      name: 'Upper-body orientation',
      definition: 'How the projected direction of the player\'s trunk changes through the movement.',
      technical: 'Projected trunk-axis orientation (°)',
      limitation: 'A projected image-plane orientation, not true three-dimensional trunk flexion or lean.',
    },
    pelvis_orientation: {
      name: 'Hip / pelvic orientation',
      definition: 'How the projected orientation of the hip line changes through the movement.',
      technical: 'Projected hip-line orientation (°)',
      limitation: 'A projected image-plane orientation, not three-dimensional pelvic rotation.',
    },
    shoulder_orientation: {
      name: 'Shoulder-line orientation',
      definition: 'How the visible line between the shoulders reorients through the movement.',
      technical: 'Projected shoulder-line orientation (°)',
      limitation: 'A projected image-plane orientation, not three-dimensional torso rotation.',
    },
    shoulder_pelvis_relationship: {
      name: 'Shoulder–pelvis relationship',
      definition: 'How the projected shoulder line changes relative to the projected hip line.',
      technical: 'Projected shoulder–pelvis orientation difference (°)',
      limitation: 'This does not measure spinal or lumbar rotation.',
    },
    elbow_pair: {
      name: 'Elbow configuration',
      definition: 'How open or bent each visible shoulder–elbow–wrist chain appears through the movement.',
      technical: 'Left/right projected elbow angle (°)',
      limitation: 'A projected two-dimensional angle; camera perspective affects its appearance.',
    },
    upper_arm_pair: {
      name: 'Upper-arm orientation',
      definition: 'How each projected shoulder-to-elbow direction changes through the movement.',
      technical: 'Left/right projected upper-arm orientation (°)',
      limitation: 'A projected image-plane orientation, not three-dimensional shoulder rotation.',
    },
    wrist_relationship: {
      name: 'Wrist–pelvis relationship',
      definition: 'How the wrist\'s projected horizontal position changes relative to the pelvis.',
      technical: 'Projected wrist–pelvis x-offset · body-scale units',
      limitation: 'A body-scale-normalized image-plane relationship.',
    },
    movement_path_unavailable: {
      name: 'Projected movement path',
      definition: 'How the athlete\'s camera-compensated on-screen path changes through the movement.',
      technical: 'Camera-compensated projected path',
      limitation: 'Path evidence may be available in the Movement Story even when this chart view is unavailable.',
    },
    hka_extrema_timing: timingPresentation('Supported HKA minimum and maximum frames'),
    hka_change_timing: timingPresentation('Largest supported frame-to-frame HKA change'),
  };
  return presentations[feature?.id] || {
    name: feature?.label || 'Selected measurement',
    definition: feature?.description || 'A supported video-derived projected measurement.',
    technical: (feature?.metrics || []).join(', '),
    limitation: 'Interpret this as projected image-plane evidence only.',
  };
}

function hkaPresentation(technical) {
  return {
    name: 'Hip–knee–ankle configuration',
    definition: 'How open or bent the visible hip–knee–ankle chain appears in the video.',
    technical,
    limitation: 'A projected two-dimensional angle. It is not automatically knee flexion, valgus, or a true three-dimensional joint angle.',
  };
}

function bilateralHkaPresentation(technical) {
  return {
    name: 'Lower-limb comparison',
    definition: 'How the projected hip–knee–ankle configurations differ between the two limbs.',
    technical,
    limitation: 'A projected bilateral difference, not a clinical asymmetry or risk score.',
  };
}

function kneeRelationshipPresentation(technical) {
  return {
    name: 'Knee–ankle relationship',
    definition: 'How the knee\'s projected position changes relative to the hip–ankle line.',
    technical,
    limitation: 'This projected relationship must not be interpreted automatically as true valgus or varus.',
  };
}

function timingPresentation(technical) {
  return {
    name: 'Timing of supported movement',
    definition: 'Where an important supported measurement moment occurs in source-frame time.',
    technical,
    limitation: 'This is measurement timing, not an estimate of the ACL injury instant.',
  };
}

function firstAvailableFeature(features) {
  return availableFeatures(features)[0] || features[0];
}

function availableCategories() {
  return Object.keys(FEATURE_CATEGORIES)
    .filter((category) => availableFeatures(FEATURE_CATEGORIES[category]).length > 0);
}

function availableFeatures(features) {
  return features.filter((feature) => {
    if (feature.visual === 'unavailable') return true;
    return feature.metrics.every((metric) => metricSeries(metric).length);
  });
}

function renderCategoryAvailabilityNote() {
  const note = $('categoryAvailabilityNote');
  if (!note) return;
  const visible = new Set(availableCategories());
  const hidden = Object.keys(FEATURE_CATEGORIES).filter((category) => !visible.has(category));
  if (!hidden.length) {
    note.textContent = '';
    note.hidden = true;
    return;
  }
  note.hidden = false;
  const reason = hidden.includes('BILATERAL')
    ? 'Injured/contralateral comparison needs injury-side metadata; neutral left/right measurements remain available when generated.'
    : 'Some categories need metadata or supported samples that are not available for this case.';
  note.textContent = 'Unavailable categories hidden: ' + hidden.join(', ') + '. ' + reason;
}

function metricSeries(metric) {
  const series = result.metric_explorer?.series?.[metric] || result.trajectories?.[metric] || [];
  return series.map((point) => ({
    frame: Number(point.source_frame_index),
    time: Number(point.timestamp_ms ?? 0),
    value: Number.isFinite(Number(point.value)) ? Number(point.value) : null,
    unit: point.unit || metricSpec(metric)?.unit || '',
    status: point.evidence_status || point.feature_status || 'UNAVAILABLE',
    reason: point.quality_reason || point.rejection_reason || ''
  }));
}

function supportedSeries(metric) {
  return metricSeries(metric).map((point) => {
    if (!isSupportedPoint(point) || !isInSupportedEvidenceRange(point.frame)) {
      return {...point, value: null};
    }
    return point;
  });
}

function isSupportedPoint(point) {
  return point.value !== null && ['SUPPORTED', 'VALID_TARGET'].includes(String(point.status));
}

function isInSupportedEvidenceRange(frame) {
  const ranges = evidenceCoverage().supported_source_ranges || [];
  if (!ranges.length) return true;
  return ranges.some((range) => frame >= Number(range.start_frame) && frame <= Number(range.end_frame));
}

function featureStats(feature) {
  const primaryMetric = feature.metrics[0];
  const primary = supportedSeries(primaryMetric);
  const supported = primary.filter((point) => point.value !== null);
  const values = supported.map((point) => point.value);
  const unit = displayUnit(supported[0]?.unit || metricSpec(primaryMetric)?.unit || '', primaryMetric);
  if (!values.length) return unavailableStats(unit, primaryMetric);
  const start = supported[0];
  const end = supported[supported.length - 1];
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce((total, value) => total + Math.pow(value - mean, 2), 0) / values.length;
  return {
    status: values.length >= 8 ? 'Supported' : 'Limited',
    unit,
    start: start.value,
    end: end.value,
    change: end.value - start.value,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean,
    median: quantile(sorted, 0.5),
    sd: Math.sqrt(variance),
    q1: quantile(sorted, 0.25),
    q3: quantile(sorted, 0.75),
    iqr: quantile(sorted, 0.75) - quantile(sorted, 0.25),
    range: sorted[sorted.length - 1] - sorted[0],
    supportedN: values.length,
    relevantN: relevantFrameCount(),
    completeness: values.length / Math.max(1, relevantFrameCount()),
    startFrame: start.frame,
    endFrame: end.frame,
    minFrame: supported.find((point) => point.value === sorted[0])?.frame,
    maxFrame: supported.find((point) => point.value === sorted[sorted.length - 1])?.frame,
    metric: primaryMetric
  };
}

function unavailableStats(unit = '', metric = '') {
  return {
    status: 'Unavailable',
    unit: displayUnit(unit, metric),
    start: null,
    end: null,
    change: null,
    min: null,
    max: null,
    mean: null,
    median: null,
    sd: null,
    q1: null,
    q3: null,
    iqr: null,
    range: null,
    supportedN: 0,
    relevantN: relevantFrameCount(),
    completeness: 0,
    metric
  };
}

function headlineHtml(stats) {
  return [
    ['Start', formatValue(stats.start, stats.unit)],
    ['End', formatValue(stats.end, stats.unit)],
    ['Signed change', formatSigned(stats.change, stats.unit)],
    ['Absolute change', formatValue(Math.abs(stats.change), stats.unit)]
  ].map(([label, value]) => '<div class="value-card"><span>' + label + '</span><strong>' + value + '</strong></div>').join('');
}

function statsGridHtml(feature, stats) {
  return [
    ['Mean', formatValue(stats.mean, stats.unit)],
    ['Median', formatValue(stats.median, stats.unit)],
    ['SD', formatValue(stats.sd, stats.unit)],
    ['Q1', formatValue(stats.q1, stats.unit)],
    ['Q3', formatValue(stats.q3, stats.unit)],
    ['IQR', formatValue(stats.iqr, stats.unit)],
    ['Minimum', formatValue(stats.min, stats.unit)],
    ['Maximum', formatValue(stats.max, stats.unit)],
    ['Range', formatValue(stats.range, stats.unit)],
  ].map(([label, value]) => '<div class="value-card"><span>' + label + '</span><strong>' + value + '</strong></div>').join('');
}

function trajectoryInterpretation(feature, stats) {
  const metric = feature.metrics?.[0];
  if (!metric || stats.status === 'Unavailable') {
    return 'No supported trajectory is available for interpretation.';
  }
  const primary = supportedSeries(metric).filter((point) => point.value !== null);
  if (!primary.length) return 'No supported trajectory is available for interpretation.';
  if (feature.metrics.length > 1) {
    const secondary = supportedSeries(feature.metrics[1]).filter((point) => point.value !== null);
    const common = primary.map((point) => {
      const other = secondary.find((item) => item.frame === point.frame);
      return other ? {frame: point.frame, difference: point.value - other.value} : null;
    }).filter(Boolean);
    if (common.length >= 2) {
      const startDifference = Math.abs(common[0].difference);
      const endDifference = Math.abs(common[common.length - 1].difference);
      const tolerance = Math.max(1e-6, Math.abs(stats.range || 0) * 0.1);
      const relationship = endDifference > startDifference + tolerance
        ? 'moved farther apart'
        : endDifference + tolerance < startDifference
          ? 'moved closer together'
          : 'remained broadly similar in separation';
      return 'The paired projected trajectories ' + relationship + ' between the first and last supported frames. '
        + gapSignalSentence(metric);
    }
  }
  const direction = Number(stats.change) > 0 ? 'increased' : Number(stats.change) < 0 ? 'decreased' : 'remained stable';
  const peak = largestChangeFrame(metric);
  return 'Across supported frames, this projected measurement generally ' + direction
    + (Number.isFinite(Number(peak)) ? '; its largest supported frame-to-frame change occurs near source frame ' + peak : '')
    + '. ' + gapSignalSentence(metric);
}

function gapSignalSentence(metric) {
  const count = gapIntervalsForMetric(metric).length;
  if (!count) return 'The trajectory is supported across the displayed interval.';
  return count + ' unsupported interval' + (count === 1 ? ' remains' : 's remain') + ' visible as a gap' + (count === 1 ? '' : 's') + '.';
}

function evidenceHtml(stats) {
  const labelClass = stats.status === 'Unavailable' ? 'status unavailable' : 'status';
  const fraction = percent(stats.completeness);
  return '<span class="' + labelClass + '">' + stats.status + '</span> '
    + fraction + ' of supported-interval frames contained measurements. Yellow chart bands mark missing or unsupported frames; lines do not connect across those gaps.';
}

function renderOperatorAnalytics(feature, stats) {
  $('supportPanel').innerHTML = supportHtml(feature, stats);
  const intervals = feature.metrics?.[0] ? gapIntervalsForMetric(feature.metrics[0]) : [];
  $('unsupportedIntervalDetails').hidden = !intervals.length;
  $('unsupportedIntervalSummary').textContent = intervals.length === 1
    ? 'Why is 1 interval unsupported?'
    : 'Why are ' + intervals.length + ' intervals unsupported?';
  $('gapReasonPanel').innerHTML = gapReasonHtml(feature, intervals);
  $('dependencyPanel').innerHTML = dependencyHtml(feature);
  $('phaseComparisonPanel').innerHTML = phaseComparisonHtml(feature, stats);
  $('provenanceText').textContent = provenanceDetails();
}

function movementNarrativeHtml(feature, stats) {
  const story = result?.movement_story || {};
  const currentPhase = phaseForFrame(currentFrame) || (story.phases || [])[0];
  const firstPhase = (story.phases || [])[0];
  const finalPhase = (story.phases || [])[story.phases.length - 1];
  const overall = story.sequence_summary
    || ((story.phases || []).length
      ? 'The observable movement is divided into ' + story.phases.length + ' evidence-backed phases from source frame '
        + result.movement_window.movement_start_frame + ' to ' + result.movement_window.movement_end_frame + '.'
      : 'A whole-video movement narrative is not available for this case.');
  const phaseSentence = currentPhase
    ? 'At source frame ' + currentFrame + ', the active phase is Phase ' + currentPhase.phase_index + ' (' + currentPhase.title
      + '), spanning frames ' + currentPhase.start_frame + '-' + currentPhase.end_frame + '.'
    : 'No phase is selected for the current frame.';
  const sequenceSentence = firstPhase && finalPhase
    ? 'The sequence starts with ' + firstPhase.title.toLowerCase() + ' and ends with ' + finalPhase.title.toLowerCase() + '.'
    : '';
  const regionRows = currentPhase ? narrativeRegionRows(currentPhase, feature, stats) : [];
  return '<p>' + escapeHtml(friendlyObservation(overall)) + '</p>'
    + '<p class="subtle">' + escapeHtml(phaseSentence + (sequenceSentence ? ' ' + sequenceSentence : '')) + '</p>'
    + (regionRows.length
      ? '<table class="compact-table"><thead><tr><th>Evidence area</th><th>What changed in this phase</th><th>Support</th></tr></thead><tbody>'
        + regionRows.join('') + '</tbody></table>'
      : '<p class="subtle">No detailed regional narrative is available for this phase.</p>')
    + '<p class="subtle">' + escapeHtml(timingNarrative(feature, stats)) + '</p>'
    + '<p class="subtle">Selected measurement: <strong>' + escapeHtml(feature.label)
    + '</strong>; supported frames: <strong>' + (stats.supportedN ?? 0) + '/' + (stats.relevantN ?? 0)
    + '</strong> (' + percent(stats.completeness) + ').</p>';
}

function narrativeRegionRows(phase, feature, stats) {
  const rows = [
    narrativeRow('Lower limb', lowerLimbNarrative(phase), regionalSupport(phase, [
      'left_hka_angle_2d_deg',
      'right_hka_angle_2d_deg',
      'injured_hka_angle_2d_deg',
      'contralateral_hka_angle_2d_deg'
    ])),
    narrativeRow('Bilateral', bilateralNarrative(phase), regionalSupport(phase, [
      'hka_projected_bilateral_difference_deg',
      'left_hka_angle_2d_deg',
      'right_hka_angle_2d_deg'
    ])),
    narrativeRow('Trunk & pelvis', trunkPelvisNarrative(phase), phaseCategorySupport(phase, 'trunk_pelvis')),
    narrativeRow('Upper body', upperBodyNarrative(phase), phaseCategorySupport(phase, 'upper_body')),
    narrativeRow('Movement path', movementPathNarrative(phase), phaseCategorySupport(phase, 'movement_path')),
  ];
  const selected = selectedFrameNarrative(feature, stats);
  if (selected) rows.push(narrativeRow('Selected frame', selected, stats.status));
  return rows.filter(Boolean);
}

function narrativeRow(label, text, support) {
  if (!text) return '';
  const status = support || 'EVIDENCE';
  const statusClass = String(status).toLowerCase().includes('unavailable') ? 'status unavailable' : 'status';
  return '<tr><td><strong>' + escapeHtml(label) + '</strong></td><td>'
    + escapeHtml(friendlyObservation(text)) + '</td><td><span class="' + statusClass + '">'
    + escapeHtml(String(status)) + '</span></td></tr>';
}

function lowerLimbNarrative(phase) {
  const injured = phaseMetricSentence(phase, 'injured_hka_angle_2d_deg', 'Injured projected HKA');
  const contra = phaseMetricSentence(phase, 'contralateral_hka_angle_2d_deg', 'Contralateral projected HKA');
  if (injured || contra) return [injured, contra].filter(Boolean).join(' ');
  const left = phaseMetricSentence(phase, 'left_hka_angle_2d_deg', 'Left projected HKA');
  const right = phaseMetricSentence(phase, 'right_hka_angle_2d_deg', 'Right projected HKA');
  const kneeLine = strongestAvailableSentence(phase, [
    ['left_knee_line_deviation_normalized', 'Left knee-line deviation'],
    ['right_knee_line_deviation_normalized', 'Right knee-line deviation']
  ]);
  return [left, right, kneeLine].filter(Boolean).join(' ');
}

function bilateralNarrative(phase) {
  const explicit = phaseMetricSentence(
    phase,
    'hka_projected_bilateral_difference_deg',
    'Injured-minus-contralateral HKA difference'
  );
  if (explicit) return explicit;
  const neutral = pairDifferenceSentence(
    phase,
    'left_hka_angle_2d_deg',
    'right_hka_angle_2d_deg',
    'left-minus-right projected HKA difference'
  );
  if (neutral) {
    return 'Injury-side mapping is unavailable for this case, so the UI uses neutral left/right evidence: ' + neutral;
  }
  return 'Bilateral comparison is unavailable because paired lower-limb measurements are not supported for this phase.';
}

function trunkPelvisNarrative(phase) {
  const summaries = phase.category_summaries?.trunk_pelvis?.metrics || {};
  const preferred = [
    ['projected_trunk_axis_angle_deg', 'Trunk axis'],
    ['projected_hip_line_angle_deg', 'Hip line'],
    ['projected_shoulder_line_angle_deg', 'Shoulder line'],
    ['projected_shoulder_pelvis_orientation_difference_deg', 'Shoulder-pelvis relationship']
  ].map(([metric, label]) => {
    if (summaries[metric]) return phaseMetricObjectSentence(summaries[metric], label);
    return phaseMetricSentence(phase, metric, label);
  });
  return preferred.filter(Boolean).join(' ');
}

function upperBodyNarrative(phase) {
  return [
    phaseMetricSentence(phase, 'left_elbow_angle_2d_deg', 'Left elbow'),
    phaseMetricSentence(phase, 'right_elbow_angle_2d_deg', 'Right elbow'),
    phaseMetricSentence(phase, 'left_upper_arm_orientation_2d_deg', 'Left upper-arm orientation'),
    phaseMetricSentence(phase, 'right_upper_arm_orientation_2d_deg', 'Right upper-arm orientation')
  ].filter(Boolean).join(' ');
}

function movementPathNarrative(phase) {
  const path = phase.category_summaries?.movement_path;
  const metrics = path?.metrics || {};
  if (path?.summary) return path.summary;
  if (Number.isFinite(Number(metrics.heading_change_deg))) {
    return 'Projected travel heading changed by ' + formatSigned(metrics.heading_change_deg, 'degrees')
      + ', with normalized projected speed changing by '
      + formatSigned(metrics.speed_change_normalized_per_s, 'body scale/s') + '.';
  }
  return 'Movement path graph is unavailable in this panel; no phase-level path statement is available for this interval.';
}

function timingNarrative(feature, stats) {
  const parts = [];
  if (Number.isFinite(Number(stats.minFrame))) parts.push('minimum at frame ' + stats.minFrame);
  if (Number.isFinite(Number(stats.maxFrame))) parts.push('maximum at frame ' + stats.maxFrame);
  const peak = feature?.metrics?.[0] ? largestChangeFrame(feature.metrics[0]) : null;
  if (Number.isFinite(Number(peak))) parts.push('largest supported frame-to-frame change near frame ' + peak);
  const detail = parts.length ? ' For the selected measurement, this means ' + parts.join(', ') + '.' : '';
  return 'Timing means the source-frame position of supported extrema or the largest supported change; it is not a claim about the ACL injury instant.' + detail;
}

function selectedFrameNarrative(feature, stats) {
  if (!feature?.metrics?.length) return '';
  const metric = feature.metrics[0];
  const point = supportedSeries(metric).find((item) => item.frame === currentFrame);
  if (!point || point.value === null) {
    return 'At source frame ' + currentFrame + ', ' + feature.label + ' is not supported for plotting.';
  }
  return 'At source frame ' + currentFrame + ', ' + feature.label + ' is '
    + formatValue(point.value, displayUnit(point.unit || stats.unit, metric)) + '.';
}

function strongestAvailableSentence(phase, metricLabels) {
  const candidates = metricLabels
    .map(([metric, label]) => ({metric, label, stat: phaseMetricStat(phase, metric)}))
    .filter((item) => item.stat)
    .sort((a, b) => Math.abs(Number(b.stat.change || 0)) - Math.abs(Number(a.stat.change || 0)));
  const best = candidates[0];
  return best ? phaseMetricObjectSentence(best.stat, best.label) : '';
}

function phaseMetricSentence(phase, metric, label) {
  const stat = phaseMetricStat(phase, metric);
  return stat ? phaseMetricObjectSentence(stat, label) : '';
}

function phaseMetricStat(phase, metric) {
  const rows = result.metric_explorer?.phase_statistics?.[metric] || [];
  return rows.find((row) => Number(row.phase_index) === Number(phase.phase_index));
}

function phaseMetricObjectSentence(stat, label) {
  const unit = displayUnit(stat.unit || '', '');
  const startFrame = stat.start_frame ?? stat.source_frame_start;
  const endFrame = stat.end_frame ?? stat.source_frame_end;
  if (!Number.isFinite(Number(stat.start_value)) || !Number.isFinite(Number(stat.end_value))) return '';
  const minFrame = stat.minimum_frame ? ', min f' + stat.minimum_frame : '';
  const maxFrame = stat.maximum_frame ? ', max f' + stat.maximum_frame : '';
  return label + ': ' + formatValue(stat.start_value, unit) + ' at f' + startFrame
    + ' to ' + formatValue(stat.end_value, unit) + ' at f' + endFrame
    + ' (' + formatSigned(stat.change, unit) + ')' + minFrame + maxFrame + '.';
}

function pairDifferenceSentence(phase, leftMetric, rightMetric, label) {
  const left = phaseMetricStat(phase, leftMetric);
  const right = phaseMetricStat(phase, rightMetric);
  if (!left || !right) return '';
  if (
    !Number.isFinite(Number(left.start_value))
    || !Number.isFinite(Number(right.start_value))
    || !Number.isFinite(Number(left.end_value))
    || !Number.isFinite(Number(right.end_value))
  ) return '';
  const unit = displayUnit(left.unit || right.unit || '', leftMetric);
  const start = Number(left.start_value) - Number(right.start_value);
  const end = Number(left.end_value) - Number(right.end_value);
  return label + ': ' + formatValue(start, unit) + ' at f' + (left.start_frame ?? left.source_frame_start)
    + ' to ' + formatValue(end, unit) + ' at f' + (left.end_frame ?? left.source_frame_end)
    + ' (' + formatSigned(end - start, unit) + ').';
}

function phaseCategorySupport(phase, key) {
  return phase.category_summaries?.[key]?.evidence_status || 'Unavailable';
}

function regionalSupport(phase, metrics) {
  const supported = metrics
    .map((metric) => phaseMetricStat(phase, metric))
    .filter(Boolean)
    .map((stat) => Number(stat.supported_fraction ?? stat.completeness ?? 0));
  if (!supported.length) return 'Unavailable';
  const coverage = supported.reduce((total, value) => total + value, 0) / supported.length;
  if (coverage >= 0.75) return 'GOOD';
  if (coverage >= 0.45) return 'LIMITED';
  return 'LOW';
}

function supportHtml(feature, stats) {
  const intervals = feature.metrics?.[0] ? gapIntervalsForMetric(feature.metrics[0]) : [];
  const level = measurementSupportLevel(stats);
  const intervalText = intervals.length === 1
    ? '1 unsupported or withheld interval'
    : intervals.length + ' unsupported or withheld intervals';
  const graphSentence = intervals.length
    ? intervalText + ' ' + (intervals.length === 1 ? 'is' : 'are') + ' shown as yellow gaps on the trajectory.'
    : 'No unsupported intervals are present in the displayed trajectory.';
  return '<div class="support-overview"><span class="status">' + escapeHtml(level) + ' · '
    + percent(stats.completeness) + ' supported</span>'
    + '<strong>' + (stats.supportedN ?? 0) + ' / ' + (stats.relevantN ?? 0) + ' relevant frames</strong></div>'
    + '<p class="support-copy">' + escapeHtml(graphSentence) + '</p>';
}

function measurementSupportLevel(stats) {
  if (stats.status === 'Unavailable' || !Number(stats.supportedN || 0)) return 'UNAVAILABLE';
  const coverage = Number(stats.completeness || 0);
  if (coverage >= 0.80) return 'GOOD';
  if (coverage >= 0.50) return 'MODERATE';
  return 'LIMITED';
}

function whyPhaseHtml() {
  const phase = phaseForFrame(currentFrame) || (result?.movement_story?.phases || [])[0];
  if (!phase) return '<p class="subtle">No supported phase story is available for this frame.</p>';
  const drivers = phaseDrivers(phase).slice(0, 3);
  const primary = drivers[0];
  const compact = primary ? phaseDefinitionSentence(phase, primary) : phaseFallbackSentence(phase);
  const details = drivers.length
    ? '<details class="advanced-details"><summary>Show measured details</summary><ul>' + drivers.map((driver) => (
      '<li><strong>' + escapeHtml(driver.label) + '</strong>: '
      + escapeHtml(friendlyObservation(driver.summary))
      + ' <span class="status">' + escapeHtml(driver.status) + '</span></li>'
    )).join('') + '</ul></details>'
    : '';
  return '<p><strong>Phase ' + phase.phase_index + ': ' + escapeHtml(phase.title) + '</strong></p>'
    + '<p class="subtle">Frames ' + phase.start_frame + '-' + phase.end_frame + ' are grouped because '
    + escapeHtml(compact) + '</p>'
    + '<p class="subtle">This explanation is about the whole frame interval, so it stays the same while you inspect different measurements.</p>'
    + details;
}

function phaseDefinitionSentence(phase, primary) {
  const category = primary.key || '';
  const summaries = phase.category_summaries || {};
  if (category === 'movement_path') {
    const metrics = summaries.movement_path?.metrics || {};
    if (Number.isFinite(Number(metrics.heading_change_deg))) {
      return 'the athlete\'s on-screen travel direction changed by '
        + formatSigned(metrics.heading_change_deg, 'degrees')
        + ', with on-screen speed relative to body size changing by '
        + formatSigned(metrics.speed_change_normalized_per_s, 'body scale/s') + '.';
    }
  }
  if (category === 'trunk_pelvis') {
    return largestMetricChangeSentence(summaries.trunk_pelvis?.metrics, 'the trunk/pelvis relationship changed most clearly');
  }
  if (category === 'upper_body') {
    return largestMetricChangeSentence(summaries.upper_body?.metrics, 'the arm positions changed most clearly');
  }
  return friendlyObservation(primary.summary || phaseFallbackSentence(phase));
}

function phaseFallbackSentence(phase) {
  return 'the supported movement evidence follows a consistent pattern across this interval.';
}

function largestMetricChangeSentence(metrics, fallback) {
  const entries = Object.entries(metrics || {})
    .filter(([, stat]) => Number.isFinite(Number(stat?.change)))
    .sort((a, b) => Math.abs(Number(b[1].change)) - Math.abs(Number(a[1].change)));
  const strongest = entries[0];
  const second = entries[1];
  if (!strongest) return fallback + '.';
  const main = readableMetricName(strongest[0]) + ' changed by '
    + formatSigned(strongest[1].change, displayUnit(strongest[1].unit || '', strongest[0]));
  const supporting = second
    ? ', while ' + readableMetricName(second[0]) + ' changed by '
      + formatSigned(second[1].change, displayUnit(second[1].unit || '', second[0]))
    : '';
  return main + supporting + '.';
}

function phaseDrivers(phase) {
  const summaries = phase.category_summaries || {};
  return Object.entries(summaries)
    .filter(([, item]) => item && item.summary)
    .map(([key, item]) => ({
      key,
      label: categoryLabel(key),
      summary: item.summary || '',
      status: item.evidence_status || 'EVIDENCE',
      score: evidenceRank(item.evidence_status) + supportedSampleHint(item) / 1000,
    }))
    .sort((a, b) => b.score - a.score);
}

function gapReasonHtml(feature, suppliedIntervals = null) {
  const metric = feature.metrics?.[0];
  if (!metric) return '<p class="subtle">No plotted metric is available for gap drilldown.</p>';
  const intervals = suppliedIntervals || gapIntervalsForMetric(metric);
  if (!intervals.length) {
    return '<p class="subtle">No missing intervals for <strong>' + escapeHtml(feature.label)
      + '</strong>. The selected measurement is supported across the displayed frame window.</p>';
  }
  const rows = intervals.slice(0, 6).map((interval) => (
    '<div class="gap-item"><strong>' + frameRangeLabel(interval.start, interval.end) + '</strong>'
    + '<p>' + escapeHtml(interval.reason) + '</p>'
    + '<p class="technical">Technical status: ' + escapeHtml(interval.technical || 'feature sample unavailable') + '</p>'
    + '<button type="button" data-gap-frame="' + interval.start + '">Show frame</button></div>'
  )).join('');
  const extra = intervals.length > 6
    ? '<p class="subtle">' + (intervals.length - 6) + ' additional gap interval(s) are hidden in this compact view.</p>'
    : '';
  setTimeout(() => {
    [...document.querySelectorAll('[data-gap-frame]')].forEach((button) => {
      button.onclick = () => {
        setFrame(Number(button.dataset.gapFrame));
        $('videoFrame').scrollIntoView({behavior: 'smooth', block: 'center'});
      };
    });
  }, 0);
  return '<p class="subtle">These intervals correspond to the yellow gaps in <strong>'
    + escapeHtml(featurePresentation(feature).name) + '</strong>.</p>' + rows + extra;
}

function gapIntervalsForMetric(metric) {
  const rows = supportedSeries(metric);
  const intervals = [];
  let current = null;
  rows.forEach((point) => {
    if (point.value !== null) {
      if (current) {
        intervals.push({
          ...current,
          reason: summarizeReasons(current.reasons),
          technical: summarizeReasons(current.technicalReasons),
        });
        current = null;
      }
      return;
    }
    if (!current) current = {start: point.frame, end: point.frame, reasons: [], technicalReasons: []};
    current.end = point.frame;
    current.reasons.push(pointGapReason(point));
    current.technicalReasons.push(pointTechnicalGapReason(point));
  });
  if (current) {
    intervals.push({
      ...current,
      reason: summarizeReasons(current.reasons),
      technical: summarizeReasons(current.technicalReasons),
    });
  }
  return intervals;
}

function pointGapReason(point) {
  const raw = pointTechnicalGapReason(point);
  const upper = raw.toUpperCase();
  if (upper.includes('TARGET_IDENTITY_UNCERTAIN') || upper.includes('TARGET OVERLAP')) {
    return 'Target identity was uncertain because the annotated athlete overlapped another player or could not be distinguished reliably.';
  }
  if (upper.includes('INVALID_TRACK_SEGMENT')) {
    return 'The target track could not be maintained reliably across this interval.';
  }
  if (upper.includes('TARGET_NOT_FOUND')) {
    return 'The annotated athlete could not be located reliably in this interval.';
  }
  if (upper.includes('LOW_CONFIDENCE') || upper.includes('LOW POSE CONFIDENCE') || upper.includes('REQUIRED LANDMARK')) {
    return 'One or more landmarks required for this measurement were not reliable enough.';
  }
  if (upper.includes('OUTSIDE THE SUPPORTED EVIDENCE INTERVAL')) {
    return 'This interval falls outside the human-accepted evidence window.';
  }
  return 'The evidence required for this measurement was unavailable or insufficient.';
}

function pointTechnicalGapReason(point) {
  const reasons = [];
  if (point.reason) reasons.push(point.reason);
  if (point.status && point.status !== 'SUPPORTED') reasons.push(point.status);
  if (!isInSupportedEvidenceRange(point.frame)) reasons.push('outside the supported evidence interval');
  if (!reasons.length) reasons.push('required landmark or feature evidence is missing');
  return reasons.join(' | ');
}

function summarizeReasons(reasons) {
  const counts = new Map();
  reasons.forEach((reason) => counts.set(reason, (counts.get(reason) || 0) + 1));
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([reason, count]) => {
      const clean = String(reason).replace(/[.;]+$/, '');
      return count > 1 ? clean + ' (' + count + ' frames)' : clean;
    })
    .join('; ');
}

function dependencyHtml(feature) {
  const metrics = feature.metrics || [];
  if (!metrics.length) return '<p class="subtle">No landmark dependency map is available for this feature.</p>';
  const cards = metrics.map((metric) => {
    const card = result.feature_cards?.[metric] || {};
    const landmarks = (card.landmarks_used || []).map((name) => name.replaceAll('_', ' '));
    const unit = displayUnit(card.unit || metricSpec(metric)?.unit || '', metric);
    const status = card.sequence_evidence || metricSpec(metric)?.evidence_note || 'available';
    const limitation = card.why_limited || 'No primary limitation reported for this measurement.';
    return '<div class="operator-row"><strong>' + escapeHtml(card.display_label || shortMetricLabel(metric)) + '</strong>'
      + '<br />Needs: ' + escapeHtml(landmarks.length ? landmarks.join(', ') : 'feature metadata unavailable')
      + '<br />Unit: ' + escapeHtml(unit || 'relative / unitless')
      + '<br />Overall evidence: ' + escapeHtml(status)
      + '<br />Known unsupported-sample reason: ' + escapeHtml(cleanGapReason(limitation)) + '</div>';
  }).join('');
  return cards;
}

function phaseComparisonHtml(feature, stats) {
  const metric = feature.metrics?.[0] || stats.metric;
  if (!metric) return '<p class="subtle">No selected metric is available for phase comparison.</p>';
  const phaseStats = result.metric_explorer?.phase_statistics?.[metric] || [];
  if (!phaseStats.length) return '<p class="subtle">Phase-level statistics are unavailable for this measurement.</p>';
  const unit = displayUnit(stats.unit || metricSpec(metric)?.unit || '', metric);
  return '<table class="compact-table"><thead><tr><th>Phase</th><th>Change</th><th>Mean</th><th>Support</th></tr></thead><tbody>'
    + phaseStats.map((phase) => (
      '<tr><td><strong>P' + phase.phase_index + '</strong><br />'
      + escapeHtml(phase.phase_title || '') + '</td><td>'
      + formatSigned(phase.change, unit) + '</td><td>'
      + formatValue(phase.mean, unit) + '</td><td>'
      + (phase.supported_n ?? 0) + '/' + (phase.relevant_n ?? 0)
      + '<br />' + percent(phase.completeness) + '</td></tr>'
    )).join('') + '</tbody></table>';
}

function phaseForFrame(frame) {
  return (result?.movement_story?.phases || []).find((phase) => (
    Number(frame) >= Number(phase.start_frame) && Number(frame) <= Number(phase.end_frame)
  ));
}

function supportedSampleHint(item) {
  const metrics = item.metrics || {};
  if (Number.isFinite(Number(metrics.supported_samples))) return Number(metrics.supported_samples);
  return Math.max(...Object.values(metrics).map((value) => Number(value?.supported_samples || 0)), 0);
}

function evidenceRank(status) {
  return {GOOD: 4, HIGH: 4, SUPPORTED: 3, MODERATE: 2, LIMITED: 1}[String(status || '').toUpperCase()] || 0;
}

function categoryLabel(key) {
  return {
    movement_path: 'Movement path',
    hip_knee_ankle_chain: 'Hip-knee-ankle chain',
    bilateral_limb_relationship: 'Bilateral limb relationship',
    trunk_pelvis: 'Trunk & pelvis',
    upper_body: 'Upper body',
    movement_timing: 'Movement timing',
  }[key] || key.replaceAll('_', ' ');
}

function humanStatus(status) {
  return String(status || '')
    .replaceAll('_', ' ')
    .toLowerCase();
}

function friendlyObservation(text) {
  return String(text || '')
    .replaceAll('Camera-compensated projected heading', 'On-screen travel direction')
    .replaceAll('camera-compensated projected heading', 'on-screen travel direction')
    .replaceAll('Body-scale-normalized projected speed', 'On-screen speed relative to body size')
    .replaceAll('body-scale-normalized projected speed', 'on-screen speed relative to body size')
    .replaceAll('projected trunk axis', 'trunk line')
    .replaceAll('projected hip line', 'hip/pelvis line')
    .replaceAll('projected shoulder line', 'shoulder line')
    .replaceAll('projected shoulder-pelvis orientation difference', 'shoulder-pelvis relationship')
    .replace(/\bdeg\b/g, 'degrees')
    .replace(/\bunits\/s\b/g, 'body scale/s');
}

function readableMetricName(metric) {
  return {
    left_hka_angle_2d_deg: 'left projected HKA',
    right_hka_angle_2d_deg: 'right projected HKA',
    injured_hka_angle_2d_deg: 'injured-side projected HKA',
    contralateral_hka_angle_2d_deg: 'contralateral projected HKA',
    hka_projected_bilateral_difference_deg: 'injured-minus-contralateral HKA difference',
    projected_trunk_axis_angle_deg: 'trunk line',
    projected_hip_line_angle_deg: 'hip/pelvis line',
    projected_shoulder_line_angle_deg: 'shoulder line',
    projected_shoulder_pelvis_orientation_difference_deg: 'shoulder-pelvis relationship',
    left_elbow_angle_2d_deg: 'left elbow angle',
    right_elbow_angle_2d_deg: 'right elbow angle',
    left_upper_arm_orientation_2d_deg: 'left upper-arm direction',
    right_upper_arm_orientation_2d_deg: 'right upper-arm direction',
  }[metric] || shortMetricLabel(metric);
}

function cleanGapReason(text) {
  const pieces = String(text || '')
    .split('|')
    .map((piece) => piece.trim())
    .filter(Boolean)
    .map((piece) => piece
      .replaceAll('TARGET_IDENTITY_UNCERTAIN', 'target identity uncertain')
      .replaceAll('INVALID_TRACK_SEGMENT', 'invalid track segment')
      .replaceAll('LOW_POSE_CONFIDENCE', 'low pose confidence')
      .replaceAll('UNAVAILABLE', 'unavailable')
      .replaceAll('_', ' ')
      .replace(/^Frame status is /i, '')
      .replace(/^unavailable$/i, '')
      .replace(/\.$/, '')
      .trim())
    .filter(Boolean);
  const unique = [...new Set(pieces)];
  if (!unique.length) return 'measurement unavailable for this frame';
  return friendlyObservation(unique.slice(0, 2).join('; '));
}

function frameRangeLabel(start, end) {
  return Number(start) === Number(end) ? 'Frame ' + start : 'Frames ' + start + '-' + end;
}

function technicalDetails(feature, stats) {
  const lines = [];
  lines.push('Selected feature: ' + feature.label);
  lines.push('Canonical metric(s): ' + (feature.metrics.length ? feature.metrics.join(', ') : 'none'));
  lines.push('Source landmarks: ' + sourceLandmarks(feature).join(', '));
  lines.push('Selected frame: ' + currentFrame);
  lines.push('Selected frame status: ' + framePointStatus(feature));
  lines.push('Processing version: ' + (result.observable_movement_descriptions?.metadata?.description_version || 'unavailable'));
  lines.push('Path quality status: ' + (result.path_quality_summary?.overall_status || 'unavailable'));
  lines.push('Supported N: ' + stats.supportedN + ' / ' + stats.relevantN);
  lines.push('Completeness: ' + percent(stats.completeness));
  lines.push('Rejected/unsupported samples remain missing; unsupported gaps are not connected.');
  return lines.join('\n');
}

function provenanceDetails() {
  const lines = [];
  lines.push('Case ID: ' + (result.case?.case_id || result.case?.slug || 'unavailable'));
  lines.push('Source ID: ' + (result.case?.source_id || 'unavailable'));
  lines.push('View ID: ' + (result.case?.view_id || result.view?.view_id || 'unavailable'));
  lines.push('View perspective: ' + (result.view?.perspective || 'unknown'));
  lines.push('Target annotation: ' + (result.target_annotation?.label || 'unavailable'));
  lines.push('Current source frame: ' + currentFrame);
  lines.push('Movement window: ' + frameBounds().start + '-' + frameBounds().end);
  const sources = Object.entries(result.source_files || {});
  if (sources.length) {
    lines.push('Source files:');
    sources.forEach(([name, path]) => lines.push('  ' + name + ': ' + path));
  }
  return lines.join('\n');
}

function sourceLandmarks(feature) {
  const names = [];
  feature.metrics.forEach((metric) => {
    const card = result.feature_cards?.[metric];
    if (card?.landmarks_used) {
      card.landmarks_used.forEach((name) => {
        if (!names.includes(name)) names.push(name);
      });
    }
  });
  return names.length ? names : ['available in feature metadata'];
}

function framePointStatus(feature) {
  if (!feature.metrics.length) return 'unavailable';
  return feature.metrics.map((metric) => {
    const point = metricSeries(metric).find((item) => item.frame === currentFrame);
    return metric + ': ' + (point?.status || 'UNAVAILABLE') + (point?.reason ? ' - ' + point.reason : '');
  }).join('\n');
}

function drawFeatureGraph(feature, stats) {
  const canvas = $('featureGraph');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(760, Math.floor(rect.width * window.devicePixelRatio));
  const cssHeight = Number.parseFloat(getComputedStyle(canvas).height) || 260;
  canvas.height = Math.floor(cssHeight * window.devicePixelRatio);
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  const width = canvas.width / window.devicePixelRatio;
  const height = canvas.height / window.devicePixelRatio;
  ctx.clearRect(0, 0, width, height);
  if (feature.visual === 'timing_extrema' || feature.visual === 'timing_change') {
    drawPlotFrame(ctx, width, height, axisLabelForFeature(feature, stats));
    drawTimingGraph(ctx, width, height, feature, stats);
    drawCursor(ctx, width, height);
    return;
  }
  const metrics = feature.metrics;
  const plotted = metrics.map((metric) => ({
    metric,
    label: shortMetricLabel(metric),
    color: metric === metrics[0] ? '#215f9a' : '#9d2735',
    series: supportedSeries(metric)
  }));
  const values = plotted.flatMap((item) => item.series.map((point) => point.value).filter((value) => value !== null));
  if (!values.length) {
    drawPlotFrame(ctx, width, height, axisLabelForFeature(feature, stats));
    drawUnavailableMessage(ctx, width, height, 'No supported samples for this feature.');
    return;
  }
  const yMin = Math.min(...values);
  const yMax = Math.max(...values);
  drawPlotFrame(ctx, width, height, axisLabelForFeature(feature, stats), yMin, yMax);
  drawMissingIntervals(ctx, plotted[0].series, width, height);
  plotted.forEach((item) => drawLine(ctx, item.series, item.color, yMin, yMax, width, height));
  drawLegend(ctx, plotted, width);
  drawCursor(ctx, width, height);
}

function drawPlotFrame(ctx, width, height, yLabel = 'Measurement value', yMin = null, yMax = null) {
  const plot = plotBox(width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#d7dfe7';
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.left, plot.top, plot.width, plot.height);
  ctx.save();
  ctx.translate(20, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#1f2a33';
  ctx.font = '700 14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.fillStyle = '#627181';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#1f2a33';
  ctx.font = '700 14px sans-serif';
  ctx.fillText('Frame number', plot.left + plot.width / 2, height - 10);
  ctx.fillStyle = '#627181';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  frameTicks(frameBounds().start, frameBounds().end, 9).forEach((frame) => {
    const x = plot.left + ((frame - frameBounds().start) / Math.max(1, frameBounds().end - frameBounds().start)) * plot.width;
    ctx.strokeStyle = '#d7dfe7';
    ctx.beginPath();
    ctx.moveTo(x, plot.top + plot.height);
    ctx.lineTo(x, plot.top + plot.height + 5);
    ctx.stroke();
    ctx.fillStyle = '#627181';
    ctx.fillText(String(frame), x, plot.top + plot.height + 20);
  });
  drawYAxisTicks(ctx, plot, yLabel, yMin, yMax);
  ctx.textAlign = 'left';
}

function drawYAxisTicks(ctx, plot, yLabel, yMin, yMax) {
  if (yMin === null || yMax === null || !Number.isFinite(Number(yMin)) || !Number.isFinite(Number(yMax))) return;
  const yPad = Number(yMax) === Number(yMin) ? 1 : (Number(yMax) - Number(yMin)) * 0.08;
  const low = Number(yMin) - yPad;
  const high = Number(yMax) + yPad;
  const ticks = valueTicks(low, high, 5);
  ctx.textAlign = 'right';
  ctx.font = '12px sans-serif';
  ticks.forEach((value) => {
    const y = plot.top + (1 - ((value - low) / Math.max(0.000001, high - low))) * plot.height;
    ctx.strokeStyle = '#edf1f5';
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.left + plot.width, y);
    ctx.stroke();
    ctx.strokeStyle = '#d7dfe7';
    ctx.beginPath();
    ctx.moveTo(plot.left - 5, y);
    ctx.lineTo(plot.left, y);
    ctx.stroke();
    ctx.fillStyle = '#627181';
    ctx.fillText(axisTickLabel(value, yLabel), plot.left - 8, y + 4);
  });
  ctx.textAlign = 'left';
}

function valueTicks(min, max, count) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const ticks = [];
  for (let index = 0; index < count; index += 1) {
    ticks.push(min + (index / Math.max(1, count - 1)) * (max - min));
  }
  return ticks;
}

function axisTickLabel(value, yLabel) {
  if (yLabel === 'Degrees') return Math.round(value) + '°';
  if (yLabel === 'Body scale') return Number(value).toFixed(2);
  if (Math.abs(Number(value)) >= 10) return Number(value).toFixed(0);
  return Number(value).toFixed(2);
}

function drawMissingIntervals(ctx, series, width, height) {
  const intervals = missingIntervals(series);
  if (!intervals.length) return;
  const plot = plotBox(width, height);
  const bounds = frameBounds();
  intervals.forEach((interval) => {
    const x1 = frameToPlotX(interval.start, bounds, plot);
    const x2 = frameToPlotX(interval.end, bounds, plot);
    ctx.fillStyle = 'rgba(255, 208, 80, 0.30)';
    ctx.fillRect(Math.min(x1, x2), plot.top, Math.max(Math.abs(x2 - x1), 6), plot.height);
  });
}

function missingIntervals(series) {
  const intervals = [];
  let current = null;
  series.forEach((point) => {
    if (point.value === null) {
      if (!current) current = {start: point.frame, end: point.frame};
      current.end = point.frame;
      return;
    }
    if (current) {
      intervals.push(current);
      current = null;
    }
  });
  if (current) intervals.push(current);
  return intervals;
}

function frameToPlotX(frame, bounds, plot) {
  return plot.left + ((frame - bounds.start) / Math.max(1, bounds.end - bounds.start)) * plot.width;
}

function drawLine(ctx, series, color, yMin, yMax, width, height) {
  const bounds = frameBounds();
  const plot = plotBox(width, height);
  const yPad = yMax === yMin ? 1 : (yMax - yMin) * 0.08;
  const low = yMin - yPad;
  const high = yMax + yPad;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  let active = false;
  let previousFrame = null;
  series.forEach((point) => {
    if (point.value === null) {
      active = false;
      previousFrame = null;
      return;
    }
    const x = frameToPlotX(point.frame, bounds, plot);
    const y = plot.top + (1 - ((point.value - low) / Math.max(0.000001, high - low))) * plot.height;
    if (!active || previousFrame === null || point.frame !== previousFrame + 1) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    active = true;
    previousFrame = point.frame;
  });
  ctx.stroke();
}

function drawTimingGraph(ctx, width, height, feature, stats) {
  const plot = plotBox(width, height);
  const bounds = frameBounds();
  ctx.strokeStyle = '#215f9a';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.top + plot.height / 2);
  ctx.lineTo(plot.left + plot.width, plot.top + plot.height / 2);
  ctx.stroke();
  const markers = feature.visual === 'timing_extrema'
    ? [['Minimum', stats.minFrame], ['Maximum', stats.maxFrame]]
    : [['Largest supported change', largestChangeFrame(feature.metrics[0])]];
  markers.forEach(([label, frame], index) => {
    if (frame === null || frame === undefined) return;
    const x = frameToPlotX(frame, bounds, plot);
    ctx.fillStyle = index === 0 ? '#215f9a' : '#9d2735';
    ctx.beginPath();
    ctx.arc(x, plot.top + plot.height / 2, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#1f2a33';
    ctx.font = '12px sans-serif';
    ctx.fillText(label + ': frame ' + frame, Math.min(x + 10, width - 180), plot.top + 42 + index * 22);
  });
}

function largestChangeFrame(metric) {
  const supported = supportedSeries(metric).filter((point) => point.value !== null);
  let bestFrame = null;
  let bestChange = -Infinity;
  for (let index = 1; index < supported.length; index += 1) {
    if (supported[index].frame !== supported[index - 1].frame + 1) continue;
    const change = Math.abs(supported[index].value - supported[index - 1].value);
    if (change > bestChange) {
      bestChange = change;
      bestFrame = supported[index].frame;
    }
  }
  return bestFrame;
}

function drawLegend(ctx, plotted, width) {
  let x = plotBox(width, 260).left;
  const y = 28;
  plotted.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y, 14, 3);
    ctx.fillStyle = '#627181';
    ctx.font = '12px sans-serif';
    ctx.fillText(item.label, x + 18, y + 5);
    x += Math.min(220, item.label.length * 7 + 38);
  });
  ctx.fillStyle = 'rgba(255, 208, 80, 0.70)';
  ctx.fillRect(x, y - 3, 14, 9);
  ctx.strokeStyle = '#9a6400';
  ctx.strokeRect(x, y - 3, 14, 9);
  ctx.fillStyle = '#627181';
  ctx.fillText('missing / unsupported', x + 18, y + 5);
  ctx.fillStyle = '#9d2735';
  ctx.fillRect(width - 150, y, 14, 3);
  ctx.fillStyle = '#627181';
  ctx.fillText('selected frame', width - 132, y + 5);
}

function drawCursor(ctx, width, height) {
  const plot = plotBox(width, height);
  const bounds = frameBounds();
  const x = frameToPlotX(currentFrame, bounds, plot);
  ctx.strokeStyle = '#9d2735';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, plot.top);
  ctx.lineTo(x, plot.top + plot.height);
  ctx.stroke();
}

function drawUnavailableMessage(ctx, width, height, text) {
  ctx.fillStyle = '#627181';
  ctx.font = '700 15px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = 'left';
}

function plotBox(width, height) {
  return {left: 64, top: 52, width: width - 92, height: height - 92};
}

function renderFilmstrip(feature, stats) {
  const metric = feature.metrics[0];
  const frames = supportedSeries(metric).filter((point) => point.value !== null).map((point) => point.frame);
  const selected = filmstripFrames(frames, stats, metric);
  $('filmstrip').innerHTML = selected.map((item) => (
    '<button class="thumb" type="button" data-film-frame="' + item.frame + '">'
    + '<img alt="' + item.label + ' source frame ' + item.frame + '" src="' + frameUrl(item.frame, true) + '" />'
    + '<span><strong>' + item.label + '</strong><br />source frame ' + item.frame + '</span>'
    + '</button>'
  )).join('');
  [...document.querySelectorAll('[data-film-frame]')].forEach((button) => {
    button.onclick = () => setFrame(Number(button.dataset.filmFrame));
  });
}

function filmstripFrames(frames, stats, metric) {
  if (!frames.length) return [];
  const lastSupportedLabel = Number(stats.endFrame) === Number(frameBounds().end) ? 'END' : 'LAST SUPPORTED';
  const candidates = [
    {label: 'START', frame: stats.startFrame ?? frames[0]},
    {label: 'MINIMUM', frame: stats.minFrame},
    {label: 'MAXIMUM', frame: stats.maxFrame},
    {label: 'SALIENT CHANGE', frame: largestChangeFrame(metric)},
    {label: lastSupportedLabel, frame: stats.endFrame ?? frames[frames.length - 1]},
  ];
  const used = new Set();
  return candidates.filter((item) => {
    if (item.frame === null || item.frame === undefined) return false;
    const frame = Number(item.frame);
    if (used.has(frame)) return false;
    if (!frames.includes(frame)) return false;
    item.frame = frame;
    used.add(frame);
    return true;
  });
}

function graphClickToFrame(event) {
  if (!result) return;
  const canvas = $('featureGraph');
  if (canvas.hidden) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const plot = plotBox(rect.width, rect.height || 360);
  const ratio = Math.max(0, Math.min(1, (x - plot.left) / Math.max(1, plot.width)));
  const bounds = frameBounds();
  const frame = Math.round(bounds.start + ratio * (bounds.end - bounds.start));
  setFrame(frame);
}

function setFrame(frame, options = {}) {
  const bounds = frameBounds();
  currentFrame = Math.max(bounds.start, Math.min(bounds.end, Number(frame)));
  $('videoFrame').src = frameUrl(currentFrame, true);
  const frameRecord = frameRecordFor(currentFrame);
  const seconds = Number(frameRecord?.timestamp_ms ?? 0) / 1000;
  $('frameReadout').textContent = 'Frame ' + currentFrame + ' | ' + seconds.toFixed(2) + ' s';
  if (options.redrawFeature !== false) renderSelectedFeature();
}

function stepFrame(delta) {
  setFrame(currentFrame + delta);
}

function togglePlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    $('playPauseButton').textContent = 'Play';
    return;
  }
  $('playPauseButton').textContent = 'Pause';
  playTimer = setInterval(() => {
    const bounds = frameBounds();
    if (currentFrame >= bounds.end) {
      togglePlayback();
      return;
    }
    setFrame(currentFrame + 1);
  }, 120);
}

function frameUrl(frame, pose) {
  return '/api/results/frame?case=' + encodeURIComponent(caseSlug)
    + '&frame=' + encodeURIComponent(frame)
    + '&roi=1&pose=' + (pose ? '1' : '0');
}

function frameBounds() {
  const frames = result?.frames || [];
  if (!frames.length) return {start: 0, end: 0};
  return {
    start: Number(frames[0].source_frame_index),
    end: Number(frames[frames.length - 1].source_frame_index)
  };
}

function frameRecordFor(frame) {
  return (result.frames || []).find((item) => Number(item.source_frame_index) === Number(frame));
}

function evidenceCoverage() {
  return result?.observable_movement_descriptions?.clip_evidence_coverage || {};
}

function relevantFrameCount() {
  const ranges = evidenceCoverage().supported_source_ranges || [];
  if (!ranges.length) return result?.frames?.length || 0;
  return ranges.reduce((total, range) => total + Number(range.frame_count || 0), 0);
}

function metricSpec(metric) {
  return result?.metric_explorer?.metrics?.[metric] || {};
}

function displayUnit(unit, metric = '') {
  const normalizedUnit = String(unit || '').trim();
  if (normalizedUnit === 'deg' || String(metric).endsWith('_deg')) return 'degrees';
  if (
    String(metric).includes('_normalized')
    || normalizedUnit === 'normalized'
    || normalizedUnit === 'normalised'
    || normalizedUnit.includes('body')
  ) return 'body scale';
  if (normalizedUnit === 'px') return 'pixels';
  return normalizedUnit;
}

function axisLabelForFeature(feature, stats) {
  const primaryMetric = feature?.metrics?.[0] || stats.metric || '';
  const unit = displayUnit(stats.unit, primaryMetric);
  if (unit === 'degrees') return 'Degrees';
  if (unit === 'body scale') return 'Body scale';
  if (unit === 'pixels') return 'Pixels';
  return unit || 'Measurement value';
}

function frameTicks(start, end, count = 5) {
  if (!Number.isFinite(Number(start)) || !Number.isFinite(Number(end))) return [];
  if (Number(start) === Number(end)) return [Number(start)];
  const ticks = [];
  for (let index = 0; index < count; index += 1) {
    ticks.push(Math.round(Number(start) + (index / Math.max(1, count - 1)) * (Number(end) - Number(start))));
  }
  return [...new Set(ticks)];
}

function shortMetricLabel(metric) {
  return {
    injured_hka_angle_2d_deg: 'Injured',
    contralateral_hka_angle_2d_deg: 'Contralateral',
    left_hka_angle_2d_deg: 'Left',
    right_hka_angle_2d_deg: 'Right',
    left_knee_line_deviation_normalized: 'Left',
    right_knee_line_deviation_normalized: 'Right',
    left_elbow_angle_2d_deg: 'Left',
    right_elbow_angle_2d_deg: 'Right',
    left_upper_arm_orientation_2d_deg: 'Left',
    right_upper_arm_orientation_2d_deg: 'Right'
  }[metric] || (metricSpec(metric).display_label || metric).replaceAll('_', ' ');
}

function quantile(sorted, q) {
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function formatValue(value, unit) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
  return Number(value).toFixed(1) + (unit ? ' ' + unit : '');
}

function formatSigned(value, unit) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
  const sign = Number(value) > 0 ? '+' : '';
  return sign + Number(value).toFixed(1) + (unit ? ' ' + unit : '');
}

function percent(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
  return Math.round(Number(value) * 100) + '%';
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
</script>
</body>
</html>
"""


RESULTS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Explorer - Results</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1d2630;
      --muted: #5c6775;
      --line: #d9dee5;
      --accent: #215f9a;
      --accent-soft: #dceaf7;
      --good: #176d4d;
      --limited: #9a6400;
      --bad: #9d2735;
      --subtle: #eef2f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1, h2, h3 { margin: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 16px; margin-bottom: 10px; }
    h3 { font-size: 14px; margin-bottom: 8px; }
    a, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      text-decoration: none;
    }
    button.active, a.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    main {
      display: block;
      width: min(1500px, 100%);
      margin: 0 auto;
      padding: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .lede { color: var(--muted); max-width: 980px; line-height: 1.4; margin: 8px 0 0; }
    .metric-grid, .evidence-grid, .region-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metadata-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .metadata-row strong { color: var(--ink); }
    .compact-panel { padding: 10px 14px; }
    .selection-flow {
      display: grid;
      grid-template-columns: minmax(220px, 0.85fr) minmax(300px, 1.15fr) minmax(220px, 0.85fr);
      gap: 12px;
      align-items: end;
      margin: 12px 0;
    }
    .selection-flow label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .selection-sentence {
      border: 1px solid var(--line);
      background: #fbfcfd;
      border-radius: 8px;
      padding: 10px;
      font-weight: 750;
    }
    .technical-subtitle {
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 500;
      margin-top: 3px;
    }
    .segmented-control {
      display: inline-flex;
      flex-wrap: wrap;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }
    .segmented-control button {
      border: 0;
      border-right: 1px solid var(--line);
      border-radius: 0;
      padding: 8px 10px;
    }
    .segmented-control button:last-child { border-right: 0; }
    .segmented-control button.active {
      background: var(--accent);
      color: white;
    }
    .feature-inspector {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      margin: 12px 0;
    }
    .feature-toolbar {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin: 10px 0;
    }
    .control-label {
      color: var(--muted);
      display: block;
      font-size: 12px;
      font-weight: 750;
      margin-bottom: 5px;
    }
    .key-values {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .key-values span {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--muted);
      display: block;
      font-size: 12px;
      padding: 8px;
    }
    .key-values strong {
      color: var(--ink);
      display: block;
      font-size: 18px;
      margin-top: 3px;
    }
    .research-details {
      border-top: 1px solid var(--line);
      margin-top: 14px;
      padding-top: 10px;
    }
    .metric, .evidence-card, .feature-card, .region-card, .limitation-row, .observation-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 10px;
    }
    .metric span, .evidence-card span, .feature-card span, .observation-card span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong { font-size: 18px; }
    .workspace {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    .story-workspace { grid-template-columns: 1fr; }
    .scope-indicator {
      border: 1px solid var(--accent);
      background: var(--accent-soft);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
      font-weight: 800;
    }
    canvas#videoCanvas {
      width: 100%;
      background: #111820;
      border-radius: 8px;
      display: block;
    }
    canvas#graphCanvas {
      width: 100%;
      min-height: 420px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
    }
    input[type="range"] { width: 100%; }
    .timeline {
      position: relative;
      height: 26px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #e8edf3;
      margin: 8px 0;
      overflow: hidden;
    }
    .timeline-band {
      position: absolute;
      top: 5px;
      height: 14px;
      background: rgba(33, 95, 154, 0.28);
      border-radius: 12px;
    }
    .timeline-cursor {
      position: absolute;
      top: 0;
      width: 2px;
      height: 100%;
      background: var(--bad);
    }
    .phase-timeline {
      position: relative;
      display: flex;
      gap: 4px;
      min-height: 82px;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #eef2f6;
      padding: 8px;
    }
    .phase-segment {
      min-width: 96px;
      border-radius: 6px;
      border: 2px solid transparent;
      background: white;
      padding: 8px;
      text-align: left;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 6px;
    }
    .phase-segment.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .phase-segment strong { font-size: 12px; line-height: 1.15; }
    .phase-segment span { color: var(--muted); font-size: 11px; font-weight: 650; }
    .phase-story-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }
    .snapshot-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .snapshot {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
      color: inherit;
      cursor: pointer;
      display: block;
      font: inherit;
      padding: 0;
      text-align: left;
      width: 100%;
    }
    .snapshot.change-lower { border-color: #b8d9b4; box-shadow: inset 4px 0 0 #b8d9b4; }
    .snapshot.change-larger { border-color: #e7d98d; box-shadow: inset 4px 0 0 #e7d98d; }
    .snapshot.change-largest { border-color: #e2a2a0; box-shadow: inset 4px 0 0 #e2a2a0; }
    .snapshot img {
      width: 100%;
      display: block;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #111820;
    }
    .snapshot figcaption, .snapshot-caption {
      padding: 8px;
      font-size: 12px;
      color: var(--muted);
      display: block;
    }
    .change-badge {
      border-radius: 999px;
      display: inline-flex;
      font-size: 11px;
      font-weight: 800;
      margin-top: 5px;
      padding: 2px 7px;
    }
    .change-badge.lower { background: #e9f5e6; color: #246b36; }
    .change-badge.larger { background: #fff7d8; color: #7b5c00; }
    .change-badge.largest { background: #fff0f0; color: #8a3131; }
    .change-legend {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 12px;
    }
    .change-swatch {
      border-radius: 999px;
      display: inline-block;
      height: 10px;
      width: 28px;
    }
    .visual-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      align-items: stretch;
      gap: 12px;
      margin-top: 12px;
    }
    .visual-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .visual-card span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      line-height: 1.25;
    }
    .visual-card h3 {
      line-height: 1.2;
      margin-bottom: 0;
    }
    .visual-summary {
      margin: 0;
    }
    .visual-card canvas {
      width: 100%;
      height: 190px;
      border-radius: 6px;
      background: #f8fafc;
      display: block;
    }
    .visual-numbers {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .visual-numbers strong {
      color: var(--ink);
      display: block;
    }
    .evidence-note {
      border-left: 4px solid var(--warn);
      background: #fffaf0;
      padding: 12px 14px;
      margin-top: 12px;
      border-radius: 6px;
    }
    .evidence-note strong {
      display: block;
      margin-bottom: 5px;
    }
    .evidence-note ul {
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }
    .evidence-note li { margin-bottom: 4px; }
    .sequence-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .sequence-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
    }
    .phase-category {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    details.panel > summary {
      cursor: pointer;
      font-weight: 750;
      list-style: none;
    }
    details.panel > summary::-webkit-details-marker { display: none; }
    details.panel > summary::after {
      content: " +";
      color: var(--accent);
    }
    details.panel[open] > summary::after { content: " -"; }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }
    .feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .observation-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 750;
      border: 1px solid var(--line);
    }
    .status.supported, .status.high, .status.good { color: var(--good); background: #e9f5ef; }
    .status.moderate, .status.limited, .status.low { color: var(--limited); background: #fff5dd; }
    .status.unavailable { color: var(--bad); background: #fff0f2; }
    .detail {
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      max-height: 420px;
      overflow: auto;
      font-size: 13px;
    }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.4; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0; }
    .mini-button { padding: 6px 8px; font-size: 12px; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted); }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .screen { display: none; }
    .screen.active { display: block; }
    .split-list { display: grid; gap: 8px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .metric-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0;
    }
    .metric-list button {
      font-size: 12px;
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .stats-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 10px;
    }
    .stats-card span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .frame-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
    .frame-table th, .frame-table td { border-bottom: 1px solid var(--line); padding: 4px; text-align: left; }
    .phase-bars { display: grid; gap: 5px; margin-top: 10px; }
    .phase-bar-row { display: grid; grid-template-columns: 72px 1fr 64px; gap: 8px; align-items: center; font-size: 12px; }
    .phase-bar-track { height: 9px; background: #e5ebf1; border-radius: 999px; overflow: hidden; }
    .phase-bar-fill { height: 100%; background: var(--accent); }
    .analytics-grid {
      display: grid;
      grid-template-columns: minmax(420px, 1.25fr) minmax(320px, 0.85fr);
      gap: 12px;
      margin-top: 12px;
    }
    .analytics-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }
    .analytics-chart {
      width: 100%;
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
    }
    .small-chart {
      width: 100%;
      height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
      margin-top: 10px;
    }
    .stats-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .stats-table th, .stats-table td {
      border-bottom: 1px solid var(--line);
      padding: 6px 4px;
      text-align: left;
    }
    .stats-table th { color: var(--muted); font-size: 12px; }
    .scope-chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 750;
      color: var(--accent);
      background: var(--accent-soft);
    }
    select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
      background: white;
    }
    @media (max-width: 760px) {
      .snapshot-strip { grid-template-columns: 1fr; }
      .analytics-grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="caseTitle">Christen Press</h1>
      <p id="caseSubtitle" class="lede">Human-annotated movement analysis</p>
    </div>
    <div class="controls">
      <a href="/" class="button">Annotate</a>
      <a id="resultLink" class="primary" href="/results?case=christen_press">View Analysis</a>
    </div>
  </header>
  <main>
    <section class="panel compact-panel">
      <div class="metadata-row" id="headerMetrics"></div>
      <p id="summaryStatement" class="lede"></p>
    </section>
    <section class="panel compact-panel" style="margin-top: 10px;">
      <div class="metadata-row" id="evidenceCompactRow"></div>
    </section>
    <section class="workspace story-workspace">
      <div class="panel">
        <div id="scopeIndicator" class="scope-indicator">Viewing: Phase</div>
        <canvas id="videoCanvas"></canvas>
        <div id="phaseTimeline" class="phase-timeline" aria-label="Movement phase timeline"></div>
        <div class="controls" aria-label="Phase subclip controls">
          <button id="playFullButton" class="mini-button">Play Full Movement</button>
          <button id="phasePlaybackButton" class="mini-button">▶ Play phase</button>
          <button id="backFiveButton" class="mini-button">-5 frames</button>
          <button id="backOneButton" class="mini-button">-1 frame</button>
          <button id="forwardOneButton" class="mini-button">+1 frame</button>
          <button id="forwardFiveButton" class="mini-button">+5 frames</button>
        </div>
        <p id="selectionReadout" class="hint"></p>
        <details style="margin-top: 10px;">
          <summary>Video overlays & target-region marks</summary>
          <div class="controls">
            <label><input id="toggleRoi" type="checkbox" checked /> target ROI</label>
            <label><input id="togglePose" type="checkbox" checked /> pose skeleton</label>
            <label><input id="toggleMask" type="checkbox" /> generated target segmentation mask</label>
            <span id="frameReadout" class="hint"></span>
          </div>
          <div class="controls">
            <button id="refineMaskButton" class="mini-button">Mark visible target regions</button>
            <button id="maskPositiveButton" class="mini-button active">Target region</button>
            <button id="maskNegativeButton" class="mini-button">Non-target region</button>
            <button id="undoMaskPromptButton" class="mini-button">Undo region mark</button>
            <button id="clearMaskPromptsButton" class="mini-button">Clear frame regions</button>
            <span id="maskPromptReadout" class="hint">Visible-region marks are saved with human provenance.</span>
          </div>
          <p class="hint">Legend: rectangular target ROI, pose skeleton, generated target segmentation mask, green target-region marks, and red non-target marks are independent overlays. Region marks guide the mask; they are not pose/keypoints and not proof that hidden joints were observed.</p>
          <input id="frameScrub" type="range" min="0" max="0" value="0" />
          <div id="timeline" class="timeline">
            <div id="timelineBand" class="timeline-band"></div>
            <div id="timelineCursor" class="timeline-cursor"></div>
          </div>
          <p class="hint">Movement End = 0 ms. Earlier movement is shown as negative time.</p>
        </details>
      </div>
      <div class="panel">
        <h2>Movement Story</h2>
        <p id="sequenceOverview" class="lede"></p>
        <div id="selectedPhaseStory" class="phase-story-card" style="margin-top: 12px;"></div>
      </div>
    </section>
    <details class="panel" id="researchMeasurements" style="margin-top: 14px;">
      <summary>Research measurements</summary>
      <div style="margin-top: 12px;">
        <p class="hint">
          Inspect one movement feature at a time. Sides, comparisons, and angle/change views
          are controls on the feature, not separate first-level measurements.
        </p>
        <div class="feature-inspector">
          <div class="selection-flow">
            <label>Category
              <select id="featureCategorySelect"></select>
            </label>
            <label>Feature
              <select id="canonicalFeatureSelect"></select>
            </label>
            <label>Scope
              <select id="researchScopeSelect"></select>
            </label>
          </div>
          <div class="feature-toolbar">
            <div>
              <span class="control-label">Side / comparison</span>
              <div id="featureSideControl" class="segmented-control"></div>
            </div>
            <div>
              <span class="control-label">View</span>
              <div id="featureViewControl" class="segmented-control">
                <button class="mini-button active" data-feature-view="angle">Angle</button>
                <button class="mini-button" data-feature-view="change">Change</button>
              </div>
            </div>
          </div>
          <div id="researchSelectionSentence" class="selection-sentence"></div>
        </div>
        <section id="angularAnalytics" class="analytics-panel" style="display:none;">
          <div class="metadata-row">
            <span class="scope-chip" id="angularScopeLabel">WHOLE MOVEMENT</span>
            <span class="hint">Unsupported intervals remain visible; no missing values are fabricated.</span>
          </div>
          <h3 id="angularChartTitle" style="margin-top: 12px;"></h3>
          <p id="angularChartSubtitle" class="hint"></p>
          <div class="analytics-grid">
            <div>
              <canvas id="angularChartCanvas" class="analytics-chart" width="900" height="420"></canvas>
              <div class="legend" id="angularLegend"></div>
            </div>
            <div id="angularStatsPanel"></div>
          </div>
        </section>
        <h3 id="metricStatsTitle" style="margin-top: 12px;">More statistics</h3>
        <div id="metricStatsPanel"></div>
        <canvas id="graphCanvas" width="720" height="520"></canvas>
        <div class="legend" id="graphLegend"></div>
        <details class="research-details">
          <summary>Research Details</summary>
          <p class="hint">
            Advanced/internal metrics, alternative transformations, bilateral difference series,
            provenance, and evidence details remain available here.
          </p>
          <div class="selection-flow">
            <label>Internal category
              <select id="metricCategorySelect"></select>
            </label>
            <label>Internal metric
              <select id="metricSelect"></select>
            </label>
            <label>Advanced transformation
              <select id="advancedAngleModeSelect">
                <option value="absolute">Angle</option>
                <option value="whole_delta">Δ° from movement start</option>
                <option value="phase_delta">Δ° from phase start</option>
                <option value="frame_delta">Frame-to-frame Δ°</option>
              </select>
            </label>
          </div>
          <div id="advancedMetricDetails" class="feature-grid"></div>
        </details>
        <h2 id="tabTitle" style="margin-top: 12px;">Feature details</h2>
        <div id="featureCards" class="feature-grid"></div>
      </div>
    </details>
    <details class="panel" id="evidencePanel" style="margin-top: 14px;">
      <summary><span id="phaseEvidenceSummary">Evidence details</span> <span class="hint">Why?</span></summary>
      <div id="evidenceDimensions" class="split-list"></div>
    </details>
    <details class="panel" style="margin-top: 14px;">
      <summary>Selected-frame evidence</summary>
      <p class="hint">Click a graph point or open a research measurement to inspect exact feature/frame evidence.</p>
      <div id="evidenceDetail" class="detail">No graph point selected yet.</div>
    </details>
    <details class="panel" style="margin-top: 14px;">
      <summary>Technical provenance</summary>
      <div class="tabs">
        <button id="summaryTab" class="active">Body Regions</button>
        <button id="limitsTab">Why limited</button>
        <button id="crossCaseTab">Cross-case</button>
      </div>
      <div id="bodyRegionScreen" class="screen active"></div>
      <div id="limitsScreen" class="screen"></div>
      <div id="crossCaseScreen" class="screen"></div>
    </details>
  </main>
<script>
const params = new URLSearchParams(location.search);
const caseSlug = params.get("case") || "christen_press";
const SelectionMode = Object.freeze({
  WHOLE_MOVEMENT: "WHOLE_MOVEMENT",
  PHASE: "PHASE",
  FIVE_FRAME_WINDOW: "FIVE_FRAME_WINDOW",
  SINGLE_FRAME: "SINGLE_FRAME"
});
let result = null;
let activeSemanticCategory = "movement_path";
let activeGroup = "whole_body";
let activeFeature = null;
let activePhaseId = null;
let activeFeatureCategory = "lower_limb";
let activeConceptId = "hka_angle";
let featureSideMode = "injured";
let featureViewMode = "angle";
let inspectionMode = "feature";
let activeMetricCategory = "movement_path";
let activeMetricName = "path:compensated_x";
let angularMode = "absolute";
let angularScope = "WHOLE_MOVEMENT";
let selectionMode = SelectionMode.PHASE;
let fiveFrameAnchor = null;
let playbackTimer = null;
let playbackMode = null;
let phasePlaybackState = "idle";
let currentFrame = 0;
let maskRefineMode = false;
let maskPromptLabel = "target";
let maskBrushActive = false;
let maskBrushPoints = [];
let lastMaskBrushPoint = null;
let frameBySource = new Map();
let phaseByFrame = new Map();
let graphLayout = [];
const colors = ["#215f9a", "#176d4d", "#9a6400", "#8236a7", "#b14528", "#52616f", "#0e7490"];
const MASK_BRUSH_STEP_PX = 14;
const FEATURE_CATEGORY_LABELS = {
  movement_path: "Movement Path",
  lower_limb: "Lower Limb",
  trunk_pelvis: "Trunk & Pelvis",
  upper_body: "Upper Body",
};
const FRIENDLY_METRIC_LABELS = {
  injured_hka_angle_2d_deg: "Injured projected HKA",
  contralateral_hka_angle_2d_deg: "Contralateral projected HKA",
  hka_projected_bilateral_difference_deg: "Projected bilateral HKA difference",
  hka_projected_bilateral_absolute_difference_deg: "Projected bilateral HKA absolute difference",
  projected_trunk_axis_angle_deg: "Projected trunk-axis orientation",
  projected_hip_line_angle_deg: "Projected hip-line orientation",
  projected_shoulder_line_angle_deg: "Projected shoulder-line orientation",
  projected_shoulder_pelvis_orientation_difference_deg: "Shoulder-pelvis orientation difference",
  right_elbow_angle_2d_deg: "Right elbow angle",
  left_elbow_angle_2d_deg: "Left elbow angle",
  right_upper_arm_orientation_2d_deg: "Right upper-arm orientation",
  left_upper_arm_orientation_2d_deg: "Left upper-arm orientation",
};
const FEATURE_CATALOG = [
  {
    id: "projected_path",
    category: "movement_path",
    label: "Projected Path",
    description: "Camera-compensated body-center movement path through the selected sequence.",
    angular: false,
    defaultSide: "feature",
    sideOptions: [
      {id: "feature", label: "Feature", metric: "path:compensated_x", metrics: ["path:compensated_x", "path:compensated_y"]},
    ],
  },
  {
    id: "hka_angle",
    category: "lower_limb",
    label: "HKA Angle",
    description: "Generic projected hip-knee-ankle angle. This is not labelled as knee flexion or valgus.",
    angular: true,
    defaultSide: "injured",
    sideOptions: [
      {id: "injured", label: "Injured", metric: "injured_hka_angle_2d_deg"},
      {id: "contralateral", label: "Contralateral", metric: "contralateral_hka_angle_2d_deg"},
      {id: "compare", label: "Compare", metric: "injured_hka_angle_2d_deg", metrics: ["injured_hka_angle_2d_deg", "contralateral_hka_angle_2d_deg"], summaryMetric: "hka_projected_bilateral_absolute_difference_deg"},
    ],
  },
  {
    id: "knee_ankle_distance",
    category: "lower_limb",
    label: "Segment-Length Diagnostic",
    description: "Projected knee-to-ankle segment-length / foreshortening diagnostic. It is not a substantive movement-story or clustering feature.",
    angular: false,
    defaultSide: "left",
    sideOptions: [
      {id: "left", label: "Left", metric: "left_knee_ankle_distance_normalized"},
      {id: "right", label: "Right", metric: "right_knee_ankle_distance_normalized"},
      {id: "compare", label: "Compare", metric: "left_knee_ankle_distance_normalized", metrics: ["left_knee_ankle_distance_normalized", "right_knee_ankle_distance_normalized"]},
    ],
  },
  {
    id: "trunk_axis",
    category: "trunk_pelvis",
    label: "Trunk Orientation",
    description: "Projected pelvis-to-shoulder midpoint axis orientation.",
    angular: true,
    defaultSide: "feature",
    sideOptions: [
      {id: "feature", label: "Feature", metric: "projected_trunk_axis_angle_deg"},
    ],
  },
  {
    id: "pelvis_orientation",
    category: "trunk_pelvis",
    label: "Pelvic Orientation",
    description: "Projected left-hip to right-hip line orientation.",
    angular: true,
    defaultSide: "feature",
    sideOptions: [
      {id: "feature", label: "Feature", metric: "projected_hip_line_angle_deg"},
    ],
  },
  {
    id: "shoulder_orientation",
    category: "trunk_pelvis",
    label: "Shoulder Orientation",
    description: "Projected left-shoulder to right-shoulder line orientation.",
    angular: true,
    defaultSide: "feature",
    sideOptions: [
      {id: "feature", label: "Feature", metric: "projected_shoulder_line_angle_deg"},
    ],
  },
  {
    id: "elbow_angle",
    category: "upper_body",
    label: "Elbow Angle",
    description: "Projected shoulder-elbow-wrist angle.",
    angular: true,
    defaultSide: "left",
    sideOptions: [
      {id: "left", label: "Left", metric: "left_elbow_angle_2d_deg"},
      {id: "right", label: "Right", metric: "right_elbow_angle_2d_deg"},
      {id: "compare", label: "Compare", metric: "left_elbow_angle_2d_deg", metrics: ["left_elbow_angle_2d_deg", "right_elbow_angle_2d_deg"], summaryMetric: "elbow_projected_bilateral_absolute_difference_deg"},
    ],
  },
  {
    id: "upper_arm_orientation",
    category: "upper_body",
    label: "Upper-Arm Orientation",
    description: "Projected shoulder-to-elbow orientation.",
    angular: true,
    defaultSide: "left",
    sideOptions: [
      {id: "left", label: "Left", metric: "left_upper_arm_orientation_2d_deg"},
      {id: "right", label: "Right", metric: "right_upper_arm_orientation_2d_deg"},
      {id: "compare", label: "Compare", metric: "left_upper_arm_orientation_2d_deg", metrics: ["left_upper_arm_orientation_2d_deg", "right_upper_arm_orientation_2d_deg"]},
    ],
  },
];

function $(id) { return document.getElementById(id); }

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function statusClass(label) {
  return String(label || "").toLowerCase().replaceAll(" ", "-");
}

function pct(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Unavailable";
  return Number(value).toFixed(digits);
}

function metricUnitDisplay(spec) {
  if (!spec) return "";
  const metricName = spec.metric_name || activeMetricName || "";
  const unit = String(spec.unit || "").trim();
  if (unit === "deg" || metricName.endsWith("_deg")) return "degrees";
  if (
    metricName.includes("_normalized")
    || unit === "normalized"
    || unit === "normalised"
    || unit.includes("body")
  ) return "body scale";
  if (unit === "units/s") return "body scale/s";
  if (unit === "px") return "pixels";
  return unit;
}

function metricYAxisLabel(spec) {
  const unit = metricUnitDisplay(spec);
  if (unit === "degrees") return "Degrees";
  if (unit === "body scale") return "Body scale";
  if (unit === "body scale/s") return "Body scale per second";
  if (unit === "pixels") return "Pixels";
  return unit || "Measurement value";
}

function valueFmtForMetric(value, digits, spec) {
  return valueFmt(value, digits, metricUnitDisplay(spec));
}

function frameTicks(minFrame, maxFrame, count = 5) {
  if (!Number.isFinite(minFrame) || !Number.isFinite(maxFrame)) return [];
  if (minFrame === maxFrame) return [Math.round(minFrame)];
  const ticks = [];
  for (let index = 0; index < count; index += 1) {
    const fraction = index / Math.max(count - 1, 1);
    ticks.push(Math.round(minFrame + fraction * (maxFrame - minFrame)));
  }
  return [...new Set(ticks)];
}

async function init() {
  result = await api(`/api/results?case=${encodeURIComponent(caseSlug)}`);
  frameBySource = new Map(result.frames.map(frame => [Number(frame.source_frame_index), frame]));
  phaseByFrame = new Map(result.phase_frame_map.map(frame => [Number(frame.source_frame_index), frame]));
  activePhaseId = result.movement_story.phases[0]?.phase_id || null;
  selectionMode = activePhaseId ? SelectionMode.PHASE : SelectionMode.WHOLE_MOVEMENT;
  angularScope = activePhaseId || "WHOLE_MOVEMENT";
  const firstConcept = availableConcepts().find(concept => concept.id === "hka_angle") || availableConcepts()[0];
  if (firstConcept) {
    activeFeatureCategory = firstConcept.category;
    activeConceptId = firstConcept.id;
    featureSideMode = availableSideOptions(firstConcept)[0]?.id || firstConcept.defaultSide;
  }
  activeMetricCategory = result.metric_explorer.categories[activeMetricCategory]
    ? activeMetricCategory
    : Object.keys(result.metric_explorer.categories)[0];
  activeMetricName = result.metric_explorer.metrics[activeMetricName]
    ? activeMetricName
    : result.metric_explorer.categories[activeMetricCategory][0]?.metric_name;
  syncActiveMetricFromFeature();
  activeFeature = activeMetricName;
  currentFrame = result.movement_story.phases[0]?.start_frame || result.movement_window.movement_start_frame;
  fiveFrameAnchor = currentFrame;
  renderHeader();
  renderEvidenceCompactRow();
  renderMovementStory();
  renderEvidenceSummary();
  renderMetricExplorer();
  renderBodyRegionEvidence();
  renderLimitations();
  renderCrossCase();
  bindControls();
  renderAngularScopeOptions();
  setFrame(currentFrame);
  renderActiveMetric();
}

function renderHeader() {
  $("caseTitle").textContent = result.case.player_name;
  $("caseSubtitle").textContent = result.case.subtitle;
  $("summaryStatement").textContent = result.summary_statement;
  $("headerMetrics").innerHTML = [
    ["Target", result.case.player_name],
    ["Source", result.case.source_id],
    ["Movement", `frames ${result.header_metrics.movement_start_frame}-${result.header_metrics.movement_end_frame}`],
    ["Duration", `${result.header_metrics.movement_duration_seconds.toFixed(2)} s`],
    ["ROI", result.header_metrics.target],
    ["Keyframes", result.header_metrics.roi_keyframes],
  ].map(([label, value]) => `<span>${label}: <strong>${value}</strong></span>`).join("");
}

function renderEvidenceCompactRow() {
  const profile = result.evidence_dimensions.find(item => item.name === "Movement Profile evidence");
  const geometry = result.evidence_dimensions.find(item => item.name === "Geometry evidence");
  const dynamics = result.evidence_dimensions.find(item => item.name === "Dynamic feature coverage");
  $("evidenceCompactRow").innerHTML = [
    `<span>Evidence: <strong>${profile?.label || "Limited"}</strong></span>`,
    `<span>Geometry: <strong>${geometry?.label || "Limited"}</strong></span>`,
    `<span>Dynamics: <strong>${dynamics?.label || "Limited"}</strong></span>`,
    `<span>View: <strong>Single broadcast view</strong></span>`,
    `<span>Details below: <strong>Evidence details + Technical provenance</strong></span>`,
  ].join("");
}

function renderMovementStory() {
  $("sequenceOverview").textContent = result.movement_story.sequence_summary;
  renderPhaseTimeline();
  renderSelectedPhaseStory();
}

function renderPhaseTimeline() {
  const phases = result.movement_story.phases || [];
  if (!phases.length) {
    $("phaseTimeline").innerHTML = `
      <div class="phase-segment active">
        <strong>Phase segmentation withheld</strong>
        <span>Target/path QA removed the contaminated interval; whole-movement measurements remain available.</span>
      </div>
    `;
    return;
  }
  const start = result.movement_window.movement_start_frame;
  const end = result.movement_window.movement_end_frame;
  const total = Math.max(end - start + 1, 1);
  $("phaseTimeline").innerHTML = phases.map(phase => {
    const width = Math.max((phase.end_frame - phase.start_frame + 1) / total * 100, 10);
    const evidence = phase.evidence_summary?.evidence_status || "LIMITED";
    return `
      <button class="phase-segment ${phase.phase_id === activePhaseId ? "active" : ""}"
        style="flex-basis:${width}%" data-phase-id="${phase.phase_id}">
        <strong>Phase ${phase.phase_index}</strong>
        <span>${phase.title}</span>
        <span>${(phase.duration_ms / 1000).toFixed(2)} s</span>
        <span class="status ${statusClass(evidence)}">${evidence}</span>
      </button>
    `;
  }).join("");
  [...$("phaseTimeline").querySelectorAll("[data-phase-id]")].forEach(button => {
    button.onclick = () => selectPhase(button.dataset.phaseId);
  });
}

function selectPhase(phaseId) {
  pausePlayback();
  phasePlaybackState = "idle";
  activePhaseId = phaseId;
  angularScope = phaseId;
  selectionMode = SelectionMode.PHASE;
  renderMovementStory();
  updatePhasePlaybackButton();
  const phase = activePhase();
  if (phase) {
    fiveFrameAnchor = phase.start_frame;
    setFrame(phase.start_frame);
  }
  renderActiveMetric();
}

function activePhase() {
  const phases = result?.movement_story?.phases || [];
  if (!activePhaseId) return null;
  return phases.find(phase => phase.phase_id === activePhaseId) || null;
}

function firstPhase() {
  return result?.movement_story?.phases?.[0] || null;
}

function phaseForFrame(frame) {
  const mapped = phaseByFrame.get(Number(frame));
  if (mapped?.phase_id) {
    return (result.movement_story.phases || []).find(phase => phase.phase_id === mapped.phase_id);
  }
  return null;
}

function renderSelectedPhaseStory() {
  if (selectionMode === SelectionMode.WHOLE_MOVEMENT) {
    renderWholeMovementStory();
    return;
  }
  const phase = activePhase();
  if (!phase) {
    renderWholeMovementStory();
    return;
  }
  const story = visualStoryForPhase(phase.phase_id);
  const evidence = phase.evidence_summary || {};
  $("phaseEvidenceSummary").textContent = `Evidence: ${evidence.evidence_status || "LIMITED"}`;
  const inspectionNote = selectionMode === SelectionMode.SINGLE_FRAME || selectionMode === SelectionMode.FIVE_FRAME_WINDOW
    ? `<p class="hint"><strong>Frame inspection is active.</strong> The phase story below is labelled as phase context; exact frame/window values are in Research measurements.</p>`
    : "";
  const observations = story?.observations || [];
  const categoryHtml = observations.map(observation => `
      <div class="visual-card" data-story-category="${observation.category}">
        <span>${observation.display_label}</span>
        <h3>${observation.display_label}</h3>
        <span>${observation.technical_label}</span>
        ${supportBadgeHtml(observation)}
        <p class="hint visual-summary">${observation.plain_language}</p>
        <canvas data-primary-visual="${observation.category}" width="420" height="230"></canvas>
        ${visualNumbersHtml(observation)}
        ${storyFeatureButton(observation.category)}
      </div>
    `).join("");
  $("selectedPhaseStory").innerHTML = `
    <span>Movement Phase</span>
    <h3>Phase ${phase.phase_index} - ${phase.title}</h3>
    <p>${story?.comparison_sentence || ""}</p>
    ${inspectionNote}
    <p class="hint">This phase: <strong>${(phase.duration_ms / 1000).toFixed(2)} s</strong>. Evidence: <strong class="status ${statusClass(evidence.evidence_status)}">${evidence.evidence_status}</strong></p>
    ${snapshotStripHtml(story)}
    <h3>Main movement changes</h3>
    <div class="visual-grid">${categoryHtml || `<p class="hint">No salient supported movement family is available for this scope.</p>`}</div>
    ${otherMeasuredFeaturesHtml(story)}
    <p class="hint">${evidence.major_limitation || ""}</p>
  `;
  [...$("selectedPhaseStory").querySelectorAll("[data-story-feature]")].forEach(button => {
    button.onclick = () => openConceptInspection(button.dataset.storyFeature);
  });
  [...$("selectedPhaseStory").querySelectorAll("[data-keyframe-frame]")].forEach(button => {
    button.onclick = () => {
      selectionMode = SelectionMode.SINGLE_FRAME;
      setFrame(Number(button.dataset.keyframeFrame), {syncPhase: true});
      renderSelectedPhaseStory();
      renderActiveMetric();
    };
  });
  requestAnimationFrame(() => drawPrimaryVisuals(story));
  renderSelectionReadout();
}

function renderWholeMovementStory() {
  const story = result.movement_visual_story.whole_movement;
  const observable = result.observable_movement_descriptions || {};
  const intervals = observable.supported_intervals || [];
  const selectedInterval = intervals[0] || null;
  const coverage = observable.clip_evidence_coverage || {};
  const descriptions = observable.default_story_descriptions || [];
  const withheld = observable.withheld_descriptions || [];
  const injuryIntervalNeedsReview = hasPostSupportedInjuryInterval(coverage);
  const showWholePath = shouldShowWholePath(story);
  $("phaseEvidenceSummary").textContent = "Evidence details";
  $("selectedPhaseStory").innerHTML = `
    <span>${injuryIntervalNeedsReview ? "Injury/fall interval" : "Evidence-bounded story"}</span>
    <h3>${injuryIntervalNeedsReview ? "Needs human review before description" : selectedInterval ? "Supported evidence interval" : "Complete observable sequence"}</h3>
    <p>${injuryIntervalNeedsReview
      ? "The visible injury/fall interval occurs after the last supported measurement frame. No movement description is claimed for this interval yet."
      : selectedInterval ? selectedInterval.label : story.sequence_summary || result.movement_story.sequence_summary || ""}</p>
    ${clipEvidenceCoverageHtml(coverage)}
    ${injuryIntervalNeedsReview ? injuryIntervalSnapshotStripHtml(coverage) : selectedInterval ? semanticSnapshotStripHtml(selectedInterval) : ""}
    ${injuryIntervalNeedsReview && selectedInterval ? earlierEvidenceHtml(selectedInterval, descriptions) : `
    <div class="visual-grid">${descriptions.length ? descriptions.map(observableDescriptionCard).join("") : `
      <p class="hint">No controlled observable movement descriptions are supported for this scope.</p>
    `}</div>
    `}
    ${withheldEvidenceHtml(withheld, coverage)}
    ${showWholePath ? `<canvas data-whole-path width="980" height="260" style="width:100%;height:260px;border-radius:8px;background:#f8fafc;margin-top:10px;"></canvas>` : ""}
    <h3 style="margin-top: 14px;">Phase sequence</h3>
    <div class="sequence-grid">${(story.phase_sequence || []).length ? story.phase_sequence.map(item => `
      <button class="sequence-item" data-phase-id="${item.phase_id}">
        <strong>Phase ${item.phase_index}</strong>
        <p class="hint">${item.title}</p>
        <p class="hint">${(Number(item.duration_ms || 0) / 1000).toFixed(2)} s</p>
      </button>
    `).join("") : `
      <div class="sequence-item">
        <strong>No defensible phase sequence</strong>
        <p class="hint">The overlap/occlusion interval was withheld instead of being turned into phase boundaries.</p>
      </div>
    `}</div>
  `;
  [...$("selectedPhaseStory").querySelectorAll("[data-phase-id]")].forEach(button => {
    button.onclick = () => selectPhase(button.dataset.phaseId);
  });
  [...$("selectedPhaseStory").querySelectorAll("[data-observable-feature]")].forEach(button => {
    button.onclick = () => openResearchMeasurement(button.dataset.observableFeature);
  });
  [...$("selectedPhaseStory").querySelectorAll("[data-keyframe-frame]")].forEach(button => {
    button.onclick = () => {
      selectionMode = SelectionMode.SINGLE_FRAME;
      setFrame(Number(button.dataset.keyframeFrame), {syncPhase: true});
      renderSelectedPhaseStory();
      renderActiveMetric();
    };
  });
  if (showWholePath) requestAnimationFrame(() => drawWholePath(story.path_points || []));
  renderSelectionReadout();
}

function shouldShowWholePath(story) {
  return result.path_quality_summary?.overall_status === "SUPPORTED"
    && Array.isArray(story.path_points)
    && story.path_points.length > 1;
}

function hasPostSupportedInjuryInterval(coverage) {
  return Boolean(
    coverage
    && coverage.has_frames_after_supported_interval
    && coverage.supported_interval_reaches_annotated_movement_end === false
    && coverage.post_supported_frame_range
  );
}

function injuryIntervalSnapshotStripHtml(coverage) {
  const range = coverage?.post_supported_frame_range;
  if (!range) return "";
  const snapshots = snapshotsFromRange(range.start_frame, range.end_frame);
  return `
    <div class="evidence-note">
      <strong>Visible injury/fall interval awaiting review</strong>
      <p class="hint">Showing raw source frames ${range.start_frame}-${range.end_frame}. These frames are after the last supported measurement frame and include the selected movement end, so they should be reviewed before any movement story is written.</p>
      <div class="snapshot-strip">
        ${snapshots.map(snapshot => `
          <button class="snapshot" data-keyframe-frame="${snapshot.frame}">
            <img alt="${snapshot.label} source frame ${snapshot.frame}" src="/api/results/frame?case=${encodeURIComponent(caseSlug)}&frame=${snapshot.frame}&roi=0&pose=0" />
            <span class="snapshot-caption">
              <strong>${snapshot.label}</strong><br />
              source frame ${snapshot.frame}
            </span>
          </button>
        `).join("")}
      </div>
    </div>
  `;
}

function snapshotsFromRange(startFrame, endFrame) {
  const start = Number(startFrame);
  const end = Number(endFrame);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return [];
  const labels = end - start >= 4 ? ["Start", "25%", "50%", "75%", "End"] : ["Start", "End"];
  const fractions = labels.length === 5 ? [0, 0.25, 0.5, 0.75, 1] : [0, 1];
  const seen = new Set();
  return fractions.map((fraction, index) => {
    const frame = Math.round(start + (end - start) * fraction);
    return {label: labels[index], frame};
  }).filter(snapshot => {
    if (seen.has(snapshot.frame)) return false;
    seen.add(snapshot.frame);
    return true;
  });
}

function earlierEvidenceHtml(interval, descriptions) {
  return `
    <details class="panel" style="margin-top: 12px;">
      <summary>Earlier supported measurements, not the injury/fall interval</summary>
      <p class="hint">${interval.label}. These observations are kept for provenance, but they are not presented as the injury sequence.</p>
      <div class="visual-grid">${descriptions.length ? descriptions.map(observableDescriptionCard).join("") : `
        <p class="hint">No controlled observable movement descriptions are supported before the injury/fall interval.</p>
      `}</div>
    </details>
  `;
}

function withheldEvidenceHtml(withheld, coverage) {
  if (!withheld.length) return "";
  const pathItems = withheld.filter(item => item.family === "MOVEMENT PATH");
  const phaseItems = withheld.filter(item => item.descriptor_id === "PHASE_SEGMENTATION_WITHHELD");
  const otherItems = withheld.filter(item => item.family !== "MOVEMENT PATH" && item.descriptor_id !== "PHASE_SEGMENTATION_WITHHELD");
  const unresolvedRange = coverage?.post_supported_frame_range;
  const pathReason = uniqueStrings(pathItems.map(item => item.evidence_reason)).join(" ");
  const phaseReason = uniqueStrings(phaseItems.map(item => item.evidence_reason)).join(" ");
  const technicalReasons = [
    pathItems.length ? `Path, direction, speed, slowdown, and stop/near-stop language withheld: ${pathReason}` : "",
    phaseItems.length ? `Phase labels withheld: ${phaseReason}` : "",
    otherItems.length ? `${otherItems.length} other descriptor${otherItems.length === 1 ? "" : "s"} withheld by evidence rules.` : "",
  ].filter(Boolean);
  const unresolvedText = unresolvedRange
    ? ` The visible fall/movement-end region, source frames ${unresolvedRange.start_frame}-${unresolvedRange.end_frame}, still needs review before it can support movement claims.`
    : "";
  return `
    <div class="evidence-note">
      <strong>Not claimed yet</strong>
      <p class="hint">The current story is limited to supported body-geometry observations. It is not claiming a movement path, a direction change, a slowdown/stop, or a phase sequence.${unresolvedText}</p>
      <details>
        <summary>Why these claims are held back</summary>
        <ul>${technicalReasons.map(reason => `<li>${reason}</li>`).join("")}</ul>
      </details>
    </div>
  `;
}

function uniqueStrings(values) {
  return [...new Set(values.filter(value => value && String(value).trim()).map(value => String(value).trim()))];
}

function clipEvidenceCoverageHtml(coverage) {
  if (!coverage || coverage.status !== "AVAILABLE") return "";
  const clipStart = coverage.clip_start_frame;
  const clipEnd = coverage.clip_end_frame;
  const movementEnd = coverage.annotated_movement_end_frame ?? clipEnd;
  const lastSupported = coverage.last_supported_source_frame;
  const postRange = coverage.post_supported_frame_range;
  const counts = coverage.post_supported_status_counts || {};
  const countText = Object.entries(counts)
    .map(([status, count]) => `${status.replaceAll("_", " ")}: ${count}`)
    .join("; ");
  const postText = coverage.has_frames_after_supported_interval && postRange ? `
    <p class="hint"><strong>Important:</strong> the selected movement continues to source frame ${movementEnd}. Source frames ${postRange.start_frame}-${postRange.end_frame} are visible in the clip but withheld from movement descriptions by quality checks${countText ? ` (${countText})` : ""}. The fall/movement-end frames are therefore unresolved evidence, not missing video.</p>
  ` : "";
  return `
    <div class="sequence-item" style="margin: 10px 0;">
      <strong>Clip evidence coverage</strong>
      <p class="hint">Analysed clip frames ${clipStart}-${clipEnd}; last supported measurement frame ${lastSupported ?? "none"}.</p>
      ${postText}
    </div>
  `;
}

function semanticSnapshotStripHtml(interval) {
  const pose = $("togglePose")?.checked === false ? "0" : "1";
  const snapshots = interval.snapshot_frames || [];
  if (!snapshots.length) return "";
  return `
    <div class="snapshot-strip">
      ${snapshots.map(snapshot => `
        <button class="snapshot" data-keyframe-frame="${snapshot.source_frame_index}">
          <img alt="${snapshot.label} source frame ${snapshot.source_frame_index}" src="/api/results/frame?case=${encodeURIComponent(caseSlug)}&frame=${snapshot.source_frame_index}&roi=0&pose=${pose}" />
          <span class="snapshot-caption">
            <strong>${snapshot.label}</strong><br />
            source frame ${snapshot.source_frame_index}
          </span>
        </button>
      `).join("")}
    </div>
  `;
}

function observableDescriptionCard(description) {
  const feature = firstExplorableFeature(description.supporting_features || []);
  return `
    <div class="visual-card">
      <span>${description.family}</span>
      <h3>${description.user_label}</h3>
      <p><strong class="status ${statusClass(description.evidence_status)}">${description.evidence_status}</strong></p>
      <p class="hint visual-summary">${description.summary}</p>
      ${observableMiniVisual(description)}
      <details>
        <summary>Show why</summary>
        ${observableEvidenceTable(description)}
      </details>
      ${feature ? `<button data-observable-feature="${feature}">Open Research Measurement</button>` : ""}
    </div>
  `;
}

function observableMiniVisual(description) {
  const hint = description.visualisation_hint || "";
  if (hint === "ghosted_pose_sequence") {
    return `<p class="hint">Visual: supported start/25/50/75/end pose snapshots above.</p>`;
  }
  if (hint.includes("bilateral")) {
    return `<p class="hint">Visual: compact injured-vs-contralateral relationship; open the HKA comparison graph below for the full trajectory.</p>`;
  }
  if (hint.includes("hka")) {
    return `<p class="hint">Visual: supported pose snapshots plus projected HKA measurement in Research Measurements.</p>`;
  }
  if (hint.includes("trunk") || hint.includes("hip") || hint.includes("shoulder")) {
    return `<p class="hint">Visual: supported pose snapshots plus projected orientation axes in Research Measurements.</p>`;
  }
  if (hint.includes("upper")) {
    return `<p class="hint">Visual: supported pose snapshots plus projected upper-body measurement in Research Measurements.</p>`;
  }
  return "";
}

function observableEvidenceTable(description) {
  const values = description.supporting_values || {};
  const rows = Object.entries(values).map(([feature, record]) => `
    <tr>
      <td>${FRIENDLY_METRIC_LABELS[feature] || feature}</td>
      <td>${valueFmt(record.start_value, 1, record.unit || "")}</td>
      <td>${valueFmt(record.end_value, 1, record.unit || "")}</td>
      <td>${signedFmt(record.change, 1, record.unit || "")}</td>
      <td>${record.supported_samples ?? "-"} / ${record.relevant_frames ?? "-"}</td>
    </tr>
  `).join("");
  return `
    <table class="stats-table">
      <thead><tr><th>Measurement</th><th>Start</th><th>End</th><th>Change</th><th>Supported</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hint">Scope: source frames ${description.scope_start}-${description.scope_end}. ${description.evidence_reason}</p>
    <p class="hint">Rule: ${description.provenance}. Thresholds are descriptive engineering rules, not clinical or population-normal cutoffs.</p>
  `;
}

function visualStoryForPhase(phaseId) {
  return (result.movement_visual_story.phases || []).find(item => item.phase_id === phaseId) || null;
}

function storyFeatureButton(category) {
  const conceptId = conceptIdForStoryCategory(category);
  return conceptId ? `<button data-story-feature="${conceptId}">Open Research Measurement</button>` : "";
}

function conceptIdForStoryCategory(category) {
  return {
    movement_path: "projected_path",
    hip_knee_ankle_chain: "hka_angle",
    bilateral_limb_relationship: "hka_angle",
    trunk_pelvis: "trunk_axis",
    upper_body: "elbow_angle",
  }[category] || null;
}

function otherMeasuredFeaturesHtml(story) {
  const others = story?.other_observations || [];
  if (!others.length) return "";
  return `
    <details style="margin-top: 12px;">
      <summary>Other measured features (${others.length})</summary>
      <div class="sequence-grid">${others.map(observation => `
        <div class="sequence-item">
          <strong>${observation.display_label}</strong>
          <p class="hint">${observation.plain_language}</p>
          ${supportBadgeHtml(observation)}
          ${storyFeatureButton(observation.category)}
        </div>
      `).join("")}</div>
    </details>
  `;
}

function openConceptInspection(conceptId) {
  const concept = availableConcepts().find(item => item.id === conceptId);
  if (!concept) return;
  const research = $("researchMeasurements");
  if (research) research.open = true;
  inspectionMode = "feature";
  activeFeatureCategory = concept.category;
  activeConceptId = concept.id;
  featureSideMode = availableSideOptions(concept)[0]?.id || concept.defaultSide;
  if (activePhaseId) angularScope = activePhaseId;
  syncActiveMetricFromFeature();
  renderMetricExplorer();
  renderActiveMetric();
  $("researchMeasurements").scrollIntoView({behavior: "smooth", block: "start"});
}

function snapshotStripHtml(story) {
  if (!story?.snapshot_frames?.length) return "";
  const pose = $("togglePose")?.checked === false ? "0" : "1";
  const hasSalient = story.snapshot_frames.some(snapshot => snapshot.change_intensity);
  return `
    <div class="snapshot-strip">
      ${story.snapshot_frames.map(snapshot => `
        <button class="snapshot ${snapshot.change_intensity ? `change-${snapshot.change_intensity}` : ""}" data-keyframe-frame="${snapshot.source_frame_index}">
          <img alt="${snapshot.label} source frame ${snapshot.source_frame_index}" src="/api/results/frame?case=${encodeURIComponent(caseSlug)}&frame=${snapshot.source_frame_index}&roi=0&pose=${pose}" />
          <span class="snapshot-caption">
            <strong>${snapshot.label}</strong><br />
            source frame ${snapshot.source_frame_index}${snapshot.selected_nearest_supported ? " (nearest supported)" : ""}
            ${snapshot.change_intensity ? `<br /><span class="change-badge ${snapshot.change_intensity}">${snapshot.change_intensity} movement change</span>` : ""}
            ${snapshot.change_reason ? `<br />${snapshot.change_reason}` : ""}
          </span>
        </button>
      `).join("")}
    </div>
    ${hasSalient ? `
      <div class="change-legend hint">
        <strong>Movement-change magnitude</strong>
        <span class="change-swatch" style="background:#e9f5e6;"></span> lower change
        <span class="change-swatch" style="background:#fff7d8;"></span> larger change
        <span class="change-swatch" style="background:#fff0f0;"></span> largest change
        <span>This scale describes movement change only, not evidence quality, severity, or risk.</span>
      </div>
    ` : ""}
  `;
}

function supportBadgeHtml(observation) {
  const support = observation.support || {};
  const fraction = support.supported_fraction;
  const percent = fraction === null || fraction === undefined ? "" : ` · ${Math.round(Number(fraction) * 100)}% supported`;
  const title = support.reason || "Measurement support is based on supported pose/feature samples.";
  const count = support.supported_samples !== null && support.supported_samples !== undefined && support.relevant_frames
    ? `<span class="hint">${support.supported_samples} / ${support.relevant_frames} phase frames supported.</span>`
    : "";
  return `
    <p title="${title}">
      <strong class="status ${statusClass(observation.evidence_status)}">${observation.evidence_status}${percent}</strong>
      ${count}
    </p>
  `;
}

function visualNumbersHtml(observation) {
  const metrics = observation.primary_metrics || {};
  let cells = [];
  if (observation.category === "movement_path") {
    cells = [
      ["Heading change", signedFmt(metrics.heading_change_deg, 1, "deg")],
      ["Speed change", signedFmt(metrics.speed_change_normalized_per_s, 2, "units/s")],
      ["Mean speed", valueFmt(metrics.mean_normalized_projected_speed_per_s, 2, "units/s")],
    ];
  } else if (observation.category === "bilateral_limb_relationship") {
    cells = [
      ["Start diff", valueFmt(metrics.signed_difference_start_deg, 1, "degrees")],
      ["End diff", valueFmt(metrics.signed_difference_end_deg, 1, "degrees")],
      ["Change", signedFmt(metrics.signed_difference_change_deg, 1, "deg")],
    ];
  } else {
    cells = [
      ["Metric", metrics.largest_metric_label || "Selected angle"],
      ["Change", signedFmt(metrics.largest_change, 1, "deg")],
      ["Evidence", observation.evidence_status || "-"],
    ];
  }
  return `<div class="visual-numbers">${cells.map(([label, value]) => `<span>${label}<strong>${value}</strong></span>`).join("")}</div>`;
}

function valueFmt(value, digits = 1, unit = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Unavailable";
  return `${Number(value).toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function signedFmt(value, digits = 1, unit = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Unavailable";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function drawPrimaryVisuals(story) {
  if (!story) return;
  [...document.querySelectorAll("[data-primary-visual]")].forEach(canvas => {
    const category = canvas.dataset.primaryVisual;
    const visual = (story.visuals || []).find(item => item.category === category);
    drawStoryVisual(canvas, story, visual, category);
  });
}

function drawStoryVisual(canvas, story, visual, category) {
  const ctx = prepareCanvas(canvas);
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  if (!visual) {
    ctx.fillText("Visual evidence unavailable for this movement family.", 16, 26);
    return;
  }
  if (category === "movement_path") drawMovementPathStory(ctx, canvas, visual, story);
  else if (category === "hip_knee_ankle_chain") drawHkaStory(ctx, canvas, story, visual);
  else if (category === "bilateral_limb_relationship") drawBilateralStory(ctx, canvas, visual);
  else if (category === "trunk_pelvis") drawTrunkPelvisStory(ctx, canvas, story);
  else if (category === "upper_body") drawUpperBodyStory(ctx, canvas, story);
  else ctx.fillText(visual.title || "Movement visual", 16, 26);
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(Math.round(canvas.clientWidth || canvas.width || 360), 260);
  const height = Math.max(Math.round(canvas.clientHeight || canvas.height || 190), 170);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);
  ctx.canvas.cssWidth = width;
  ctx.canvas.cssHeight = height;
  return ctx;
}

function drawWholePath(points) {
  const canvas = document.querySelector("[data-whole-path]");
  if (!canvas) return;
  const ctx = prepareCanvas(canvas);
  drawMiniPath(ctx, canvas, points, {title: "Whole movement projected path", color: "#215f9a"});
}

function drawMovementPathStory(ctx, canvas, visual, story) {
  drawMiniPath(ctx, canvas, visual.points || [], {title: "This phase - projected path", color: "#215f9a"});
  const width = canvas.clientWidth || 360;
  const height = canvas.clientHeight || 220;
  const speedVisual = (story.visuals || []).find(item => item.visual_type === "projected_speed_sparkline");
  if (speedVisual) drawSparkline(ctx, speedVisual.points || [], 18, height - 32, Math.max(width - 36, 120), 20);
}

function drawMiniPath(ctx, canvas, points, options = {}) {
  const width = canvas.clientWidth || canvas.width || 360;
  const height = canvas.clientHeight || canvas.height || 190;
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText(options.title || "Projected path", 16, 22);
  const clean = (points || []).filter(point => point.x !== null && point.y !== null);
  if (clean.length < 2) {
    ctx.fillStyle = "#5c6775";
    ctx.fillText("Path evidence unavailable in this scope.", 16, 54);
    return;
  }
  const rect = {x: 32, y: 46, width: width - 64, height: height - 96};
  const fit = fitCoordinatePoints(clean, rect);
  ctx.strokeStyle = options.color || "#215f9a";
  ctx.lineWidth = 2.5;
  groupedPathSegments(fit).forEach(segment => {
    ctx.beginPath();
    segment.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  });
  const start = fit[0];
  const end = fit[fit.length - 1];
  ctx.fillStyle = "#176d4d";
  ctx.beginPath();
  ctx.arc(start.x, start.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#9d2735";
  ctx.beginPath();
  ctx.arc(end.x, end.y, 5, 0, Math.PI * 2);
  ctx.fill();
  drawArrow(ctx, fit[Math.max(0, fit.length - 2)], end, "#9d2735");
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.fillText("START", 18, height - 14);
  ctx.fillText("END", width - 44, height - 14);
}

function groupedPathSegments(points) {
  const groups = [];
  let current = [];
  let currentId = null;
  points.forEach(point => {
    const segmentId = point.path_segment_id || "single_path_segment";
    if (current.length && segmentId !== currentId) {
      groups.push(current);
      current = [];
    }
    currentId = segmentId;
    current.push(point);
  });
  if (current.length) groups.push(current);
  return groups;
}

function drawSparkline(ctx, points, x, y, width, height) {
  const values = points.filter(point => point.value !== null).map(point => Number(point.value));
  if (values.length < 2) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  ctx.strokeStyle = "#9a6400";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  values.forEach((value, index) => {
    const px = x + (index / Math.max(values.length - 1, 1)) * width;
    const py = y + height - ((value - min) / Math.max(max - min, 1e-9)) * height;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawHkaStory(ctx, canvas, story, visual) {
  const width = canvas.clientWidth || 360;
  const snapshots = story.snapshot_frames || [];
  const start = snapshots[0]?.landmarks || {};
  const end = snapshots[snapshots.length - 1]?.landmarks || {};
  const injured = (visual.laterality_mapping?.injured || result.movement_visual_story.laterality_mapping?.injured || "right").toLowerCase();
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText("Injured projected HKA chain", 16, 22);
  drawLegSnapshot(ctx, start, injured, {x: 20, y: 40, width: width / 2 - 32, height: 110}, "#215f9a", "Start");
  drawLegSnapshot(ctx, end, injured, {x: width / 2 + 12, y: 40, width: width / 2 - 32, height: 110}, "#176d4d", "End");
}

function drawBilateralStory(ctx, canvas, visual) {
  const width = canvas.clientWidth || 360;
  const metrics = visual.metrics || {};
  const start = Number(metrics.signed_difference_start_deg || 0);
  const end = Number(metrics.signed_difference_end_deg || 0);
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText("Injured minus contralateral projected HKA", 16, 22);
  drawDifferenceBar(ctx, "Phase start", start, 48, width);
  drawDifferenceBar(ctx, "Phase end", end, 100, width);
}

function drawDifferenceBar(ctx, label, value, y, width) {
  const center = width / 2;
  const scale = Math.min((width - 80) / 90, 4);
  ctx.strokeStyle = "#d9dee5";
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(40, y);
  ctx.lineTo(width - 40, y);
  ctx.stroke();
  ctx.strokeStyle = value >= 0 ? "#176d4d" : "#9a6400";
  ctx.beginPath();
  ctx.moveTo(center, y);
  ctx.lineTo(center + value * scale, y);
  ctx.stroke();
  ctx.fillStyle = "#1d2630";
  ctx.beginPath();
  ctx.arc(center + value * scale, y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, 16, y - 13);
  ctx.fillText(`${fmt(value, 1)} deg`, Math.min(width - 72, center + value * scale + 8), y + 4);
}

function drawTrunkPelvisStory(ctx, canvas, story) {
  const width = canvas.clientWidth || 360;
  const snapshots = story.snapshot_frames || [];
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText("Projected trunk, shoulder, and hip axes", 16, 22);
  drawAxisSnapshot(ctx, snapshots[0]?.landmarks || {}, {x: 20, y: 42, width: width / 2 - 32, height: 110}, "Start");
  drawAxisSnapshot(ctx, snapshots[snapshots.length - 1]?.landmarks || {}, {x: width / 2 + 12, y: 42, width: width / 2 - 32, height: 110}, "End");
}

function drawUpperBodyStory(ctx, canvas, story) {
  const width = canvas.clientWidth || 360;
  const snapshots = story.snapshot_frames || [];
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText("Projected upper body", 16, 22);
  drawUpperSnapshot(ctx, snapshots[0]?.landmarks || {}, {x: 20, y: 42, width: width / 2 - 32, height: 110}, "Start");
  drawUpperSnapshot(ctx, snapshots[snapshots.length - 1]?.landmarks || {}, {x: width / 2 + 12, y: 42, width: width / 2 - 32, height: 110}, "End");
}

function drawLegSnapshot(ctx, landmarks, side, rect, color, label) {
  const names = [`${side}_hip`, `${side}_knee`, `${side}_ankle`];
  const points = projectLandmarks(landmarks, names, rect);
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, rect.x, rect.y - 8);
  if (!points.every(Boolean)) {
    ctx.fillText("Leg points unavailable", rect.x, rect.y + 48);
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  ctx.lineTo(points[1].x, points[1].y);
  ctx.lineTo(points[2].x, points[2].y);
  ctx.stroke();
  points.forEach(point => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawAxisSnapshot(ctx, landmarks, rect, label) {
  const names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"];
  const points = projectLandmarks(landmarks, names, rect);
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, rect.x, rect.y - 8);
  if (!points.every(Boolean)) {
    ctx.fillText("Axis points unavailable", rect.x, rect.y + 48);
    return;
  }
  line(ctx, points[0], points[1], "#215f9a", 4);
  line(ctx, points[2], points[3], "#176d4d", 4);
  const shoulderMid = midpointPoint(points[0], points[1]);
  const pelvisMid = midpointPoint(points[2], points[3]);
  line(ctx, pelvisMid, shoulderMid, "#9a6400", 3);
}

function drawUpperSnapshot(ctx, landmarks, rect, label) {
  const names = ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"];
  const points = projectLandmarks(landmarks, names, rect);
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, rect.x, rect.y - 8);
  if (!points.some(Boolean)) {
    ctx.fillText("Arm points unavailable", rect.x, rect.y + 48);
    return;
  }
  if (points[0] && points[1] && points[2]) {
    line(ctx, points[0], points[1], "#215f9a", 3);
    line(ctx, points[1], points[2], "#215f9a", 3);
  }
  if (points[3] && points[4] && points[5]) {
    line(ctx, points[3], points[4], "#176d4d", 3);
    line(ctx, points[4], points[5], "#176d4d", 3);
  }
}

function projectLandmarks(landmarks, names, rect) {
  const raw = names.map(name => landmarks?.[name]).map(point => point && point.x !== null && point.y !== null ? {x: Number(point.x), y: Number(point.y)} : null);
  const finite = raw.filter(Boolean);
  if (!finite.length) return raw;
  const minX = Math.min(...finite.map(point => point.x));
  const maxX = Math.max(...finite.map(point => point.x));
  const minY = Math.min(...finite.map(point => point.y));
  const maxY = Math.max(...finite.map(point => point.y));
  const scale = Math.min(rect.width / Math.max(maxX - minX, 1e-9), rect.height / Math.max(maxY - minY, 1e-9)) * 0.82;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  return raw.map(point => point ? {
    x: rect.x + rect.width / 2 + (point.x - centerX) * scale,
    y: rect.y + rect.height / 2 + (point.y - centerY) * scale
  } : null);
}

function fitCoordinatePoints(points, rect) {
  const minX = Math.min(...points.map(point => point.x));
  const maxX = Math.max(...points.map(point => point.x));
  const minY = Math.min(...points.map(point => point.y));
  const maxY = Math.max(...points.map(point => point.y));
  return points.map(point => ({
    ...point,
    x: rect.x + ((point.x - minX) / Math.max(maxX - minX, 1e-9)) * rect.width,
    y: rect.y + rect.height - ((point.y - minY) / Math.max(maxY - minY, 1e-9)) * rect.height,
  }));
}

function line(ctx, a, b, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function midpointPoint(a, b) {
  return {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2};
}

function drawArrow(ctx, from, to, color) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(to.x, to.y);
  ctx.lineTo(to.x - 12 * Math.cos(angle - Math.PI / 6), to.y - 12 * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(to.x - 12 * Math.cos(angle + Math.PI / 6), to.y - 12 * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function semanticLabel(key) {
  return result.semantic_category_labels[key] || key.replaceAll("_", " ");
}

function readableFamily(key) {
  return {
    movement_path: "movement path",
    hip_knee_ankle_chain: "Hip-Knee-Ankle chain",
    bilateral_limb_relationship: "bilateral limb relationship",
    trunk_pelvis: "trunk and pelvis",
    upper_body: "upper body"
  }[key] || key.replaceAll("_", " ");
}

function selectionBounds() {
  const phase = activePhase();
  const movementStart = result.movement_window.movement_start_frame;
  const movementEnd = result.movement_window.movement_end_frame;
  if (selectionMode === SelectionMode.WHOLE_MOVEMENT) return {start: movementStart, end: movementEnd};
  if (selectionMode === SelectionMode.SINGLE_FRAME) return {start: currentFrame, end: currentFrame};
  if (selectionMode === SelectionMode.FIVE_FRAME_WINDOW) {
    const bounds = phase || {start_frame: movementStart, end_frame: movementEnd};
    const anchor = clampFrame(fiveFrameAnchor ?? currentFrame, bounds.start_frame, bounds.end_frame);
    const start = Math.min(anchor, bounds.end_frame);
    const end = Math.min(start + 4, bounds.end_frame);
    return {start, end};
  }
  if (phase) return {start: phase.start_frame, end: phase.end_frame};
  return {start: movementStart, end: movementEnd};
}

function clampFrame(frame, start, end) {
  return Math.min(Math.max(Math.round(frame), start), end);
}

function renderSelectionReadout() {
  if (!result) return;
  const bounds = selectionBounds();
  const phase = activePhase();
  const phaseSuffix = phase ? ` - Phase ${phase.phase_index}` : "";
  const scopeLabel = {
    WHOLE_MOVEMENT: `Viewing: Whole Movement - ${result.header_metrics.movement_duration_seconds.toFixed(2)} s`,
    PHASE: phase ? `Viewing: Phase ${phase.phase_index} of ${(result.movement_story.phases || []).length} - ${(phase.duration_ms / 1000).toFixed(2)} s` : "Viewing: Phase",
    FIVE_FRAME_WINDOW: `Viewing: Frames ${bounds.start}-${bounds.end}${phaseSuffix}`,
    SINGLE_FRAME: `Viewing: Frame ${currentFrame}${phaseSuffix}`
  }[selectionMode];
  $("scopeIndicator").textContent = scopeLabel;
  $("selectionReadout").textContent = `${scopeLabel}. Active scope source frames ${bounds.start}-${bounds.end}. Navigation stays within the selected phase unless you scrub or click another phase/frame.`;
}

function renderEvidenceSummary() {
  $("evidenceDimensions").innerHTML = result.evidence_dimensions.map(item => `
    <div class="evidence-card" title="${item.explanation}">
      <span>${item.name}</span>
      <strong class="status ${statusClass(item.label)}">${item.label}</strong>
      <p class="hint">${item.value === null ? item.explanation : `${pct(item.value)}. ${item.explanation}`}</p>
    </div>
  `).join("");
}

function observationCard(observation) {
  const feature = firstExplorableFeature(observation.technical_feature_names || []);
  return `
    <div class="observation-card">
      <span>${result.semantic_category_labels[observation.category] || observation.category}</span>
      <h3>${observation.title}</h3>
      <p><strong class="status ${statusClass(observation.evidence_status)}">${observation.evidence_status}</strong></p>
      <p class="hint">${observation.plain_language_summary}</p>
      ${feature ? `<button data-explore-feature="${feature}">Open Research Measurement</button>` : ""}
    </div>
  `;
}

function renderTabs() {
  if (!$("tabs")) return;
  const labels = {
    whole_body: "WHOLE BODY",
    lower_limb: "LOWER LIMB",
    trunk_pelvis: "TRUNK & PELVIS",
    upper_body: "UPPER BODY",
    bilateral: "BILATERAL",
    evidence: "EVIDENCE"
  };
  $("tabs").innerHTML = Object.keys(labels).map(key => (
    `<button data-group="${key}" class="${key === activeGroup ? "active" : ""}">${labels[key]}</button>`
  )).join("");
  [...$("tabs").querySelectorAll("button")].forEach(button => {
    button.onclick = () => {
      activeGroup = button.dataset.group;
      if (activeGroup !== "evidence") {
        activeFeature = result.feature_groups[activeGroup][0]?.feature_name || activeFeature;
      }
      renderTabs();
      renderActiveGroup();
    };
  });
}

function firstExplorableFeature(features) {
  return features.find(feature => result.metric_explorer.metrics[feature] || result.feature_cards[feature]);
}

function groupForFeature(feature) {
  for (const [group, items] of Object.entries(result.feature_groups)) {
    if (items.some(item => item.feature_name === feature)) return group;
  }
  return null;
}

function openResearchMeasurement(feature) {
  if (!feature || !result.metric_explorer.metrics[feature]) return;
  const research = $("researchMeasurements");
  if (research) research.open = true;
  const conceptMatch = conceptForMetric(feature);
  if (conceptMatch) {
    inspectionMode = "feature";
    activeFeatureCategory = conceptMatch.concept.category;
    activeConceptId = conceptMatch.concept.id;
    featureSideMode = conceptMatch.side.id;
    syncActiveMetricFromFeature();
  } else {
    inspectionMode = "advanced";
    activeMetricName = feature;
    activeFeature = feature;
    activeMetricCategory = result.metric_explorer.metrics[feature]?.metric_family || activeMetricCategory;
  }
  renderMetricExplorer();
  renderActiveMetric();
  loadEvidenceForMetric(feature, currentFrame);
  $("researchMeasurements").scrollIntoView({behavior: "smooth", block: "start"});
}

function renderMetricExplorer() {
  renderCanonicalFeatureControls();
  renderAdvancedMetricControls();
  renderResearchScopeOptions();
  renderResearchSelectionSentence();
}

function renderCanonicalFeatureControls() {
  const categories = availableFeatureCategories();
  const categorySelect = $("featureCategorySelect");
  const featureSelect = $("canonicalFeatureSelect");
  categorySelect.innerHTML = categories.map(key => (
    `<option value="${key}">${FEATURE_CATEGORY_LABELS[key] || key}</option>`
  )).join("");
  if (!categories.includes(activeFeatureCategory)) activeFeatureCategory = categories[0] || "lower_limb";
  categorySelect.value = activeFeatureCategory;
  categorySelect.onchange = event => {
    inspectionMode = "feature";
    activeFeatureCategory = event.target.value;
    const first = availableConcepts(activeFeatureCategory)[0];
    if (first) {
      activeConceptId = first.id;
      featureSideMode = availableSideOptions(first)[0]?.id || first.defaultSide;
    }
    syncActiveMetricFromFeature();
    renderMetricExplorer();
    renderActiveMetric();
  };
  const concepts = availableConcepts(activeFeatureCategory);
  if (!concepts.some(concept => concept.id === activeConceptId) && concepts.length) {
    activeConceptId = concepts[0].id;
  }
  const concept = activeConcept();
  featureSelect.innerHTML = concepts.map(conceptItem => (
    `<option value="${conceptItem.id}">${conceptItem.label}</option>`
  )).join("");
  featureSelect.value = activeConceptId;
  featureSelect.onchange = event => {
    inspectionMode = "feature";
    activeConceptId = event.target.value;
    const next = activeConcept();
    featureSideMode = availableSideOptions(next)[0]?.id || next?.defaultSide || "feature";
    syncActiveMetricFromFeature();
    renderMetricExplorer();
    renderActiveMetric();
  };
  renderFeatureSideControl(concept);
  renderFeatureViewControl(concept);
}

function renderFeatureSideControl(concept) {
  const control = $("featureSideControl");
  const options = availableSideOptions(concept);
  if (!options.length) {
    control.innerHTML = `<span class="hint">No supported side view available.</span>`;
    return;
  }
  if (!options.some(option => option.id === featureSideMode)) featureSideMode = options[0].id;
  control.innerHTML = options.map(option => (
    `<button class="mini-button ${option.id === featureSideMode ? "active" : ""}" data-feature-side="${option.id}">${option.label}</button>`
  )).join("");
  [...control.querySelectorAll("[data-feature-side]")].forEach(button => {
    button.onclick = () => {
      inspectionMode = "feature";
      featureSideMode = button.dataset.featureSide;
      syncActiveMetricFromFeature();
      renderMetricExplorer();
      renderActiveMetric();
    };
  });
}

function renderFeatureViewControl(concept) {
  const control = $("featureViewControl");
  const buttons = [...control.querySelectorAll("[data-feature-view]")];
  const angular = Boolean(concept?.angular);
  if (!angular) featureViewMode = "angle";
  buttons.forEach(button => {
    button.disabled = !angular && button.dataset.featureView === "change";
    button.classList.toggle("active", button.dataset.featureView === featureViewMode);
    button.onclick = () => {
      if (button.disabled) return;
      inspectionMode = "feature";
      featureViewMode = button.dataset.featureView;
      syncActiveMetricFromFeature();
      renderMetricExplorer();
      renderActiveMetric();
    };
  });
}

function renderAdvancedMetricControls() {
  const labels = result.metric_explorer.metric_category_labels;
  const categories = result.metric_explorer.categories;
  const categorySelect = $("metricCategorySelect");
  const metricSelect = $("metricSelect");
  if (!categorySelect || !metricSelect) return;
  categorySelect.innerHTML = Object.keys(categories).map(key => (
    `<option value="${key}">${labels[key] || key}</option>`
  )).join("");
  categorySelect.value = activeMetricCategory;
  categorySelect.onchange = event => {
    inspectionMode = "advanced";
    activeMetricCategory = event.target.value;
    activeMetricName = result.metric_explorer.categories[activeMetricCategory][0]?.metric_name || activeMetricName;
    activeFeature = activeMetricName;
    renderMetricExplorer();
    renderActiveMetric();
  };
  const metrics = categories[activeMetricCategory] || [];
  metricSelect.innerHTML = metrics.map(item => (
    `<option value="${item.metric_name}">${item.display_label} (${item.metric_name})</option>`
  )).join("");
  metricSelect.value = activeMetricName;
  metricSelect.onchange = event => {
    inspectionMode = "advanced";
    activeMetricName = event.target.value;
    activeFeature = activeMetricName;
    renderMetricExplorer();
    renderActiveMetric();
  };
  const advancedMode = $("advancedAngleModeSelect");
  if (advancedMode) {
    advancedMode.value = angularMode;
    advancedMode.onchange = event => {
      inspectionMode = "advanced";
      angularMode = event.target.value;
      renderActiveMetric();
    };
  }
  renderAdvancedMetricDetails();
}

function renderAdvancedMetricDetails() {
  const container = $("advancedMetricDetails");
  const spec = activeMetricSpec();
  if (!container || !spec) return;
  container.innerHTML = `
    <div class="feature-card">
      <span>Internal metric</span>
      <h3>${spec.display_label}</h3>
      <p class="hint"><span class="mono">${spec.metric_name}</span></p>
      <p class="hint">Family: ${result.metric_explorer.metric_category_labels[spec.metric_family] || spec.metric_family}</p>
      <p class="hint">${spec.evidence_note || "Unsupported/rejected samples remain unavailable."}</p>
      <button data-evidence-feature="${spec.metric_name}">Inspect selected frame evidence</button>
    </div>
    <div class="feature-card">
      <span>Advanced transformations</span>
      <h3>${angleModeLabel(angularMode, angularScopeRecord()).short}</h3>
      <p class="hint">Use this only when you need the technical series. The normal feature view uses Angle or Change.</p>
      <p class="hint">Current scope: ${angularScopeRecord().label}</p>
    </div>
  `;
  [...container.querySelectorAll("[data-evidence-feature]")].forEach(button => {
    button.onclick = () => loadEvidenceForMetric(button.dataset.evidenceFeature, currentFrame);
  });
}

function availableFeatureCategories() {
  return Object.keys(FEATURE_CATEGORY_LABELS).filter(category => availableConcepts(category).length);
}

function availableConcepts(category = null) {
  return FEATURE_CATALOG.filter(concept => {
    if (category && concept.category !== category) return false;
    return availableSideOptions(concept).length > 0;
  });
}

function activeConcept() {
  return availableConcepts().find(concept => concept.id === activeConceptId) || availableConcepts()[0] || null;
}

function availableSideOptions(concept) {
  if (!concept) return [];
  return concept.sideOptions.filter(option => {
    const names = option.metrics || [option.metric];
    return names.every(name => result.metric_explorer.metrics[name] || result.metric_explorer.series[name]);
  });
}

function activeSideOption() {
  const concept = activeConcept();
  const options = availableSideOptions(concept);
  return options.find(option => option.id === featureSideMode) || options[0] || null;
}

function syncActiveMetricFromFeature() {
  const side = activeSideOption();
  if (!side) return;
  activeMetricName = side.metric || side.metrics?.[0] || activeMetricName;
  activeMetricCategory = result.metric_explorer.metrics[activeMetricName]?.metric_family || activeMetricCategory;
  activeFeature = activeMetricName;
  angularMode = featureViewMode === "change"
    ? (angularScope === "WHOLE_MOVEMENT" ? "whole_delta" : "phase_delta")
    : "absolute";
}

function conceptForMetric(metricName) {
  for (const concept of availableConcepts()) {
    for (const side of availableSideOptions(concept)) {
      const names = side.metrics || [side.metric];
      if (names.includes(metricName) || side.summaryMetric === metricName) {
        return {concept, side};
      }
    }
  }
  return null;
}

function renderActiveMetric() {
  if (inspectionMode === "feature") syncActiveMetricFromFeature();
  const spec = activeMetricSpec();
  if (!spec) return;
  const concept = activeConcept();
  $("tabTitle").textContent = inspectionMode === "feature" && concept ? `${concept.label} details` : `${spec.display_label} details`;
  $("featureCards").innerHTML = metricDetailCards(spec);
  renderAdvancedMetricDetails();
  [...document.querySelectorAll("[data-evidence-feature]")].forEach(button => {
    button.onclick = () => loadEvidenceForMetric(button.dataset.evidenceFeature, currentFrame);
  });
  drawGraph();
  renderMetricStats();
  renderAngularAnalytics();
}

function activeMetricSpec() {
  return result?.metric_explorer?.metrics?.[activeMetricName] || null;
}

function metricSeries(metricName) {
  return result.metric_explorer.series[metricName] || [];
}

function renderAngularScopeOptions() {
  renderResearchScopeOptions();
}

function renderResearchScopeOptions() {
  if (!$("researchScopeSelect") || !result) return;
  $("researchScopeSelect").innerHTML = [
    `<option value="WHOLE_MOVEMENT">Whole movement</option>`,
    ...(result.movement_story.phases || []).map(phase => (
      `<option value="${phase.phase_id}">Phase ${phase.phase_index} - ${phase.title}</option>`
    ))
  ].join("");
  $("researchScopeSelect").value = angularScope;
  $("researchScopeSelect").onchange = event => {
    angularScope = event.target.value;
    if (angularScope === "WHOLE_MOVEMENT") {
      selectionMode = SelectionMode.WHOLE_MOVEMENT;
      activePhaseId = null;
      renderMovementStory();
    } else {
      activePhaseId = angularScope;
      selectionMode = SelectionMode.PHASE;
      renderMovementStory();
      const phase = activePhase();
      if (phase) setFrame(phase.start_frame, {syncPhase: false});
    }
    if (inspectionMode === "feature") syncActiveMetricFromFeature();
    renderActiveMetric();
  };
}

function renderResearchSelectionSentence() {
  const spec = activeMetricSpec();
  if (!spec || !$("researchSelectionSentence")) return;
  const scope = angularScopeRecord();
  if (inspectionMode === "feature") {
    const concept = activeConcept();
    const side = activeSideOption();
    const view = concept?.angular ? (featureViewMode === "change" ? changeReferenceLabel(scope) : "Angle") : "Trajectory";
    $("researchSelectionSentence").innerHTML = `
      ${FEATURE_CATEGORY_LABELS[concept?.category] || "Movement Feature"} → ${concept?.label || spec.display_label} → ${side?.label || "Feature"} → ${scope.label} → ${view}
      <span class="technical-subtitle">Underlying metric: ${technicalMetricSubtitle(spec)}</span>
    `;
    return;
  }
  const category = result.metric_explorer.metric_category_labels[spec.metric_family] || spec.metric_family;
  const mode = spec.angular ? angleModeLabel(angularMode, scope).short : "Metric value";
  $("researchSelectionSentence").innerHTML = `
    Research Details → ${category} → ${spec.display_label} → ${scope.label} → ${mode}
    <span class="technical-subtitle">${technicalMetricSubtitle(spec)}</span>
  `;
}

function renderAngularAnalytics() {
  const spec = activeMetricSpec();
  const panel = $("angularAnalytics");
  if (!panel) return;
  const isAngular = Boolean(spec?.angular);
  panel.style.display = isAngular ? "block" : "none";
  $("graphCanvas").style.display = isAngular ? "none" : "block";
  $("graphLegend").style.display = isAngular ? "none" : "flex";
  $("metricStatsPanel").style.display = isAngular ? "none" : "block";
  if ($("metricStatsTitle")) $("metricStatsTitle").style.display = isAngular ? "none" : "block";
  if (!isAngular) return;
  renderAngularScopeOptions();
  const scope = angularScopeRecord();
  $("angularScopeLabel").textContent = scope.label.toUpperCase();
  const context = chartContextLabel(spec, scope);
  $("angularChartTitle").textContent = context.title;
  $("angularChartSubtitle").textContent = context.subtitle;
  renderResearchSelectionSentence();
  drawAngularTrajectory(spec, scope);
  renderAngularStats(spec, scope);
}

function angularScopeRecord() {
  if (angularScope === "WHOLE_MOVEMENT") {
    return {
      id: "WHOLE_MOVEMENT",
      label: "Whole Movement",
      start: result.movement_window.movement_start_frame,
      end: result.movement_window.movement_end_frame,
      stats: result.metric_explorer.whole_movement_statistics[activeMetricName] || {},
    };
  }
  const phase = (result.movement_story.phases || []).find(item => item.phase_id === angularScope) || activePhase() || firstPhase();
  return {
    id: phase?.phase_id || "WHOLE_MOVEMENT",
    label: phase ? `Phase ${phase.phase_index}: ${phase.title}` : "Whole Movement",
    start: phase?.start_frame ?? result.movement_window.movement_start_frame,
    end: phase?.end_frame ?? result.movement_window.movement_end_frame,
    phase,
    stats: phase ? (result.metric_explorer.phase_statistics[activeMetricName] || []).find(item => item.phase_id === phase.phase_id) || {} : {},
  };
}

function angleModeLabel(mode, scope) {
  if (mode === "absolute") return {short: "Absolute angle", full: "Angle (°)"};
  if (mode === "whole_delta") {
    return {short: "Δ° from movement start", full: "Δ° relative to movement start"};
  }
  if (mode === "phase_delta") {
    return {
      short: `Δ° from ${scope.label} start`,
      full: `Δ° relative to ${scope.label} start`,
    };
  }
  if (mode === "frame_delta") return {short: "Frame-to-frame Δ°", full: "Frame-to-frame Δ°"};
  return {short: "Metric value", full: "Metric value"};
}

function changeReferenceLabel(scope) {
  return scope.id === "WHOLE_MOVEMENT"
    ? "Change from movement start"
    : `Change from ${scope.label} start`;
}

function chartContextLabel(spec, scope) {
  if (inspectionMode === "feature") {
    const concept = activeConcept();
    const side = activeSideOption();
    const view = featureViewMode === "change" ? changeReferenceLabel(scope) : "Angle";
    return {
      title: `${concept?.label || spec.display_label} — ${scope.label}`,
      subtitle: `${side?.label || "Feature"} · ${view}`,
    };
  }
  const modeLabel = angleModeLabel(angularMode, scope);
  return {
    title: `${spec.display_label} — ${scope.label}`,
    subtitle: `Advanced internal metric · ${modeLabel.full}`,
  };
}

function technicalMetricSubtitle(spec) {
  const unit = spec.unit ? ` (${spec.unit})` : "";
  const technical = spec.metric_name.startsWith("path:")
    ? spec.metric_name.replace("path:", "path / ")
    : spec.metric_name;
  return `${technical}${unit}`;
}

function angularSeries(metricName, mode, scope) {
  const rows = metricSeries(metricName).map(point => ({...point, rawValue: point.value}));
  const supported = rows.filter(point => point.value !== null);
  if (!supported.length) return rows.map(point => ({...point, value: null}));
  if (mode === "absolute") return rows;
  if (mode === "whole_delta") {
    const startValue = supported[0].value;
    return rows.map(point => ({...point, value: point.value === null ? null : angularDifference(point.value, startValue)}));
  }
  if (mode === "phase_delta") {
    const byPhaseStart = new Map();
    (result.movement_story.phases || []).forEach(phase => {
      const first = supported.find(point => point.source_frame_index >= phase.start_frame && point.source_frame_index <= phase.end_frame);
      if (first) byPhaseStart.set(phase.phase_id, first.value);
    });
    return rows.map(point => {
      if (point.value === null) return {...point, value: null};
      const phase = phaseForFrame(point.source_frame_index);
      const startValue = byPhaseStart.get(phase?.phase_id);
      return {...point, value: startValue === undefined ? null : angularDifference(point.value, startValue)};
    });
  }
  if (mode === "frame_delta") {
    let previous = null;
    return rows.map(point => {
      if (point.value === null) {
        previous = null;
        return {...point, value: null};
      }
      const value = previous === null ? null : angularDifference(point.value, previous);
      previous = point.value;
      return {...point, value};
    });
  }
  return rows;
}

function angularDifference(current, reference) {
  const spec = activeMetricSpec();
  if (spec?.signed && /orientation|heading|line|upper-arm/i.test(spec.display_label)) {
    let diff = Number(current) - Number(reference);
    while (diff > 180) diff -= 360;
    while (diff < -180) diff += 360;
    return diff;
  }
  return Number(current) - Number(reference);
}

function drawAngularTrajectory(spec, scope) {
  const canvas = $("angularChartCanvas");
  const ctx = canvas.getContext("2d");
  graphLayout = [];
  clearChart(ctx, canvas);
  const metricNames = angularPlottedMetricNames(spec);
  const scopeIsWhole = scope.id === "WHOLE_MOVEMENT";
  const scopeFrames = result.frames.filter(frame => frame.source_frame_index >= scope.start && frame.source_frame_index <= scope.end);
  if (!scopeFrames.length) {
    ctx.fillStyle = "#1d2630";
    ctx.fillText("No frames available for this scope.", 24, 44);
    return;
  }
  const plotted = metricNames.map((metricName, index) => ({
    metricName,
    label: result.metric_explorer.metrics[metricName]?.display_label || metricName,
    color: colors[index % colors.length],
    points: angularSeries(metricName, angularMode, scope).filter(point => (
      scopeIsWhole || (point.source_frame_index >= scope.start && point.source_frame_index <= scope.end)
    )),
  }));
  const allFrames = scopeIsWhole
    ? result.frames.map(frame => Number(frame.source_frame_index))
    : scopeFrames.map(frame => Number(frame.source_frame_index));
  const minFrame = Math.min(...allFrames);
  const maxFrame = Math.max(...allFrames);
  const allValues = plotted.flatMap(item => item.points.map(point => point.value).filter(value => value !== null));
  if (!allValues.length) {
    ctx.fillStyle = "#1d2630";
    ctx.fillText("Angular evidence unavailable for this metric.", 24, 44);
    return;
  }
  const margin = {left: 72, right: 24, top: 42, bottom: 48};
  const plotW = canvas.width - margin.left - margin.right;
  const plotH = canvas.height - margin.top - margin.bottom;
  const minV = Math.min(...allValues);
  const maxV = Math.max(...allValues);
  const scaleX = frame => margin.left + ((frame - minFrame) / Math.max(maxFrame - minFrame, 1e-9)) * plotW;
  const scaleY = value => margin.top + plotH - ((value - minV) / Math.max(maxV - minV, 1e-9)) * plotH;
  drawAxes(ctx, canvas, margin, plotW, plotH, yAxisLabel(), "Frame number", frameTicks(minFrame, maxFrame), scaleX);
  drawPhaseBands(ctx, scaleX, margin, plotH, scope, scopeIsWhole);
  drawUnsupportedIntervals(ctx, scaleX, margin, plotH, plotted[0]?.points || []);
  plotted.forEach(item => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    let drawing = false;
    item.points.forEach(point => {
      if (point.value === null) {
        drawing = false;
        return;
      }
      const x = scaleX(point.source_frame_index);
      const y = scaleY(point.value);
      if (!drawing) {
        ctx.moveTo(x, y);
        drawing = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
  drawGapMarkers(ctx, scaleX, margin, plotH, plotted[0]?.points || []);
  const frameInfo = frameBySource.get(currentFrame);
  if (frameInfo) {
    const x = scaleX(currentFrame);
    ctx.strokeStyle = "#9d2735";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, margin.top - 8);
    ctx.lineTo(x, margin.top + plotH + 8);
    ctx.stroke();
    ctx.fillStyle = "#9d2735";
    ctx.font = "12px sans-serif";
    ctx.fillText(`Frame ${currentFrame} · ${fmt(frameInfo.timestamp_ms / 1000, 2)} s`, Math.min(x + 6, canvas.width - 150), margin.top + 16);
  }
  $("angularLegend").innerHTML = plotted.map(item => `<span><i class="dot" style="background:${item.color}"></i> ${item.label}</span>`).join("")
    + `<span><i class="dot" style="background:#9d2735"></i> selected frame</span>`
    + `<span><i class="dot" style="background:#9a6400"></i> unsupported interval</span>`;
  graphLayout.push({feature: spec.metric_name, minFrame, maxFrame, marginLeft: margin.left, marginRight: margin.right});
}

function angularPlottedMetricNames(spec) {
  if (inspectionMode === "feature") {
    const side = activeSideOption();
    const names = side?.metrics || [side?.metric || spec.metric_name];
    return names.filter(name => result.metric_explorer.metrics[name]);
  }
  if (spec.metric_name.includes("injured_hka") || spec.metric_name.includes("contralateral_hka")) {
    return ["injured_hka_angle_2d_deg", "contralateral_hka_angle_2d_deg"].filter(name => result.metric_explorer.metrics[name]);
  }
  const names = [spec.metric_name];
  if (spec.paired_metric_name && result.metric_explorer.metrics[spec.paired_metric_name]) names.push(spec.paired_metric_name);
  return names;
}

function yAxisLabel() {
  return {
    absolute: "Degrees",
    whole_delta: "Degrees from movement start",
    phase_delta: "Degrees from phase start",
    frame_delta: "Degrees per frame",
  }[angularMode] || "Degrees";
}

function renderAngularStats(spec, scope) {
  const wholeScope = {
    id: "WHOLE_MOVEMENT",
    label: "Whole Movement",
    start: result.movement_window.movement_start_frame,
    end: result.movement_window.movement_end_frame,
  };
  const wholeSeries = angularSeries(spec.metric_name, angularMode, wholeScope);
  const whole = statsForSeries(wholeSeries, wholeScope.start, wholeScope.end);
  const selectedSeries = angularSeries(spec.metric_name, angularMode, scope).filter(point => (
    point.source_frame_index >= scope.start && point.source_frame_index <= scope.end
  ));
  const selected = statsForSeries(selectedSeries, scope.start, scope.end);
  const single = metricSeries(spec.metric_name).find(point => point.source_frame_index === currentFrame);
  const unit = metricUnitDisplay(spec) || "degrees";
  const wholeHtml = scope.id === "WHOLE_MOVEMENT"
    ? ""
    : `<details style="margin-top: 10px;"><summary>Whole movement statistics</summary>${statsTable("Whole movement", whole, unit)}</details>`;
  $("angularStatsPanel").innerHTML = `
    <div class="analytics-panel">
      <h3>${inspectionMode === "feature" ? activeConcept()?.label || spec.display_label : spec.display_label}</h3>
      <p class="hint"><strong>${scope.label.toUpperCase()}</strong> statistics use only supported angular samples. Unsupported samples remain visible gaps.</p>
      ${keyValuesHtml(selected, unit)}
      ${comparisonSummaryHtml()}
      ${statsTable(`${scope.label} statistics`, selected, unit)}
      ${wholeHtml}
      <h3 style="margin-top: 12px;">Selected frame</h3>
      <p class="hint">Source frame <strong>${currentFrame}</strong>: <strong>${valueFmt(single?.value, 2, unit)}</strong> (${single?.evidence_status || "UNAVAILABLE"})</p>
      <div class="controls">
        ${keyFrameButton("Start", selected.start_frame)}
        ${keyFrameButton("End", selected.end_frame)}
        ${keyFrameButton("Minimum", selected.minimum_frame)}
        ${keyFrameButton("Maximum", selected.maximum_frame)}
        ${keyFrameButton("Peak Δ", selected.peak_frame_to_frame_change_frame)}
      </div>
    </div>
  `;
  [...$("angularStatsPanel").querySelectorAll("[data-key-frame]")].forEach(button => {
    button.onclick = () => {
      selectionMode = SelectionMode.SINGLE_FRAME;
      setFrame(Number(button.dataset.keyFrame), {syncPhase: true});
      renderSelectedPhaseStory();
      renderActiveMetric();
    };
  });
}

function keyValuesHtml(stats, unit) {
  return `
    <div class="key-values">
      <span>Start<strong>${valueFmt(stats.start_value, 1, unit)}</strong></span>
      <span>End<strong>${valueFmt(stats.end_value, 1, unit)}</strong></span>
      <span>Change<strong>${signedFmt(stats.change, 1, unit)}</strong></span>
      <span>Maximum<strong>${valueFmt(stats.maximum, 1, unit)}</strong></span>
    </div>
  `;
}

function comparisonSummaryHtml() {
  if (inspectionMode !== "feature" || activeSideOption()?.id !== "compare") return "";
  const summaryMetric = activeSideOption()?.summaryMetric;
  if (!summaryMetric || !result.metric_explorer.metrics[summaryMetric]) return "";
  const scope = angularScopeRecordForMetric(summaryMetric);
  const stats = scope.stats || {};
  return `
    <div class="stats-card">
      <span>Bilateral comparison summary</span>
      <p class="hint">Mean absolute injured-contralateral difference: <strong>${valueFmt(stats.mean, 1, "degrees")}</strong></p>
      <p class="hint">Peak absolute difference: <strong>${valueFmt(stats.maximum, 1, "degrees")}</strong></p>
    </div>
  `;
}

function angularScopeRecordForMetric(metricName) {
  if (angularScope === "WHOLE_MOVEMENT") {
    return {
      id: "WHOLE_MOVEMENT",
      label: "Whole Movement",
      stats: result.metric_explorer.whole_movement_statistics[metricName] || {},
    };
  }
  const phase = (result.movement_story.phases || []).find(item => item.phase_id === angularScope) || activePhase() || firstPhase();
  return {
    id: phase?.phase_id || "WHOLE_MOVEMENT",
    label: phase ? `Phase ${phase.phase_index}: ${phase.title}` : "Whole Movement",
    stats: phase ? (result.metric_explorer.phase_statistics[metricName] || []).find(item => item.phase_id === phase.phase_id) || {} : {},
  };
}

function statsTable(label, stats, unit = "") {
  const rows = [
    ["Mean", stats.mean],
    ["Median", stats.median],
    ["SD", stats.standard_deviation],
    ["Minimum", stats.minimum],
    ["Maximum", stats.maximum],
    ["Q1", stats.q1],
    ["Q3", stats.q3],
    ["IQR", stats.iqr],
    ["Range", stats.range],
    ["Start", stats.start_value],
    ["End", stats.end_value],
    ["Signed change", stats.change],
    ["Absolute start-to-end change", stats.absolute_change],
    ["Cumulative absolute angular change", stats.total_absolute_change],
  ];
  return `
    <table class="stats-table">
      <thead><tr><th colspan="2">${label.toUpperCase()}</th></tr></thead>
      <tbody>${rows.map(([name, value]) => `<tr><td>${name}</td><td>${valueFmt(value, 2, unit)}</td></tr>`).join("")}</tbody>
    </table>
  `;
}

function keyFrameButton(label, frame) {
  return frame === null || frame === undefined
    ? ""
    : `<button class="mini-button" data-key-frame="${frame}">${label}: frame ${frame}</button>`;
}

function drawAngularBarChart(spec) {
  const canvas = $("angularBarCanvas");
  const ctx = canvas.getContext("2d");
  clearChart(ctx, canvas);
  const stats = result.metric_explorer.phase_statistics[spec.metric_name] || [];
  const aggregate = angularMode === "absolute" ? "mean" : angularMode === "frame_delta" ? "total_absolute_change" : "change";
  const values = stats.map(item => item[aggregate]).filter(value => value !== null && value !== undefined);
  if (!values.length) {
    ctx.fillStyle = "#1d2630";
    ctx.fillText("Phase summary unavailable.", 20, 36);
    return;
  }
  const maxAbs = Math.max(...values.map(value => Math.abs(value)), 1e-9);
  const width = canvas.width;
  const rowH = Math.max((canvas.height - 48) / Math.max(stats.length, 1), 24);
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  const aggregateLabel = aggregate === "total_absolute_change"
    ? "cumulative absolute angular change"
    : aggregate.replaceAll("_", " ");
  ctx.fillText(`Phase ${aggregateLabel} (${spec.unit})`, 18, 22);
  stats.forEach((item, index) => {
    const value = item[aggregate];
    const y = 42 + index * rowH;
    const center = width * 0.46;
    const barW = value === null || value === undefined ? 0 : Math.abs(value) / maxAbs * (width * 0.40);
    ctx.fillStyle = "#5c6775";
    ctx.fillText(`P${item.phase_index}`, 18, y + 12);
    ctx.strokeStyle = "#d9dee5";
    ctx.beginPath();
    ctx.moveTo(center, y + 6);
    ctx.lineTo(center, y + 18);
    ctx.stroke();
    ctx.fillStyle = Number(value) >= 0 ? "#215f9a" : "#9a6400";
    const x = Number(value) >= 0 ? center : center - barW;
    ctx.fillRect(x, y + 5, barW, 12);
    ctx.fillStyle = "#1d2630";
    ctx.fillText(valueFmt(value, 1, spec.unit), width - 94, y + 14);
  });
}

function drawAngularHeatmap(aggregate) {
  const canvas = $("angularHeatmapCanvas");
  const ctx = canvas.getContext("2d");
  clearChart(ctx, canvas);
  const rows = (result.metric_explorer.angular_heatmap?.[aggregate] || [])
    .filter(item => item.value !== null && item.value !== undefined);
  const phaseIds = [...new Set((result.movement_story.phases || []).map(phase => phase.phase_id))];
  const metricNames = result.metric_explorer.angular_metric_names.slice(0, 10);
  if (!rows.length || !metricNames.length) {
    ctx.fillStyle = "#1d2630";
    ctx.fillText("Angular heatmap unavailable.", 20, 36);
    return;
  }
  const byKey = new Map(rows.map(item => [`${item.metric_name}:${item.phase_id}`, Number(item.value)]));
  const values = [...byKey.values()].map(value => Math.abs(value));
  const maxAbs = Math.max(...values, 1e-9);
  const left = 150;
  const top = 34;
  const cellW = Math.max((canvas.width - left - 16) / Math.max(phaseIds.length, 1), 28);
  const cellH = Math.max((canvas.height - top - 12) / Math.max(metricNames.length, 1), 18);
  ctx.fillStyle = "#1d2630";
  ctx.font = "12px sans-serif";
  ctx.fillText(`Feature x phase ${aggregate.replaceAll("_", " ")}`, 14, 18);
  phaseIds.forEach((phaseId, index) => {
    const phase = (result.movement_story.phases || []).find(item => item.phase_id === phaseId);
    ctx.fillText(`P${phase?.phase_index || index + 1}`, left + index * cellW + 4, top - 8);
  });
  metricNames.forEach((metricName, rowIndex) => {
    const label = result.metric_explorer.metrics[metricName]?.display_label || metricName;
    ctx.fillStyle = "#5c6775";
    ctx.fillText(label.slice(0, 24), 8, top + rowIndex * cellH + cellH * 0.68);
    phaseIds.forEach((phaseId, colIndex) => {
      const value = byKey.get(`${metricName}:${phaseId}`);
      const alpha = value === undefined ? 0.06 : Math.min(Math.abs(value) / maxAbs, 1) * 0.78 + 0.12;
      ctx.fillStyle = value === undefined ? "#eef2f6" : `rgba(33, 95, 154, ${alpha})`;
      ctx.fillRect(left + colIndex * cellW, top + rowIndex * cellH, cellW - 2, cellH - 2);
    });
  });
}

function clearChart(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawAxes(ctx, canvas, margin, plotW, plotH, yLabel, xLabel = "Frame number", xTicks = [], scaleX = null) {
  ctx.strokeStyle = "#d9dee5";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();
  ctx.fillStyle = "#5c6775";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(yLabel, 12, margin.top - 14);
  ctx.textAlign = "right";
  ctx.fillText(xLabel, canvas.width - margin.right, canvas.height - 14);
  if (scaleX) {
    ctx.textAlign = "center";
    xTicks.forEach(frame => {
      const x = scaleX(frame);
      ctx.strokeStyle = "#d9dee5";
      ctx.beginPath();
      ctx.moveTo(x, margin.top + plotH);
      ctx.lineTo(x, margin.top + plotH + 5);
      ctx.stroke();
      ctx.fillStyle = "#5c6775";
      ctx.fillText(String(frame), x, margin.top + plotH + 20);
    });
  }
  ctx.textAlign = "left";
}

function drawPhaseBands(ctx, scaleX, margin, plotH, scope, scopeIsWhole) {
  const phases = scopeIsWhole
    ? (result.movement_story.phases || [])
    : (result.movement_story.phases || []).filter(phase => phase.phase_id === scope.id);
  phases.forEach((phase, index) => {
    const x1 = scaleX(phase.start_frame);
    const x2 = scaleX(phase.end_frame);
    const selected = phase.phase_id === activePhaseId || phase.phase_id === scope.id;
    ctx.fillStyle = selected ? "rgba(33, 95, 154, 0.14)" : (index % 2 ? "rgba(33, 95, 154, 0.05)" : "rgba(23, 109, 77, 0.04)");
    ctx.fillRect(Math.min(x1, x2), margin.top, Math.max(Math.abs(x2 - x1), 3), plotH);
    ctx.fillStyle = "#5c6775";
    ctx.font = "11px sans-serif";
    ctx.fillText(`P${phase.phase_index}`, Math.min(x1, x2) + 4, margin.top + 13);
  });
  (result.movement_story.phases || []).forEach(phase => {
    if (!scopeIsWhole && phase.phase_id !== scope.id) return;
    const x = scaleX(phase.start_frame);
    ctx.strokeStyle = "#d9dee5";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
    ctx.stroke();
  });
}

function drawGapMarkers(ctx, scaleX, margin, plotH, points) {
  const gaps = points.filter(point => point.value === null);
  if (!gaps.length) return;
  ctx.fillStyle = "#9a6400";
  gaps.forEach(point => {
    const x = scaleX(point.source_frame_index);
    ctx.fillRect(x - 2, margin.top + plotH + 5, 4, 8);
  });
}

function drawUnsupportedIntervals(ctx, scaleX, margin, plotH, points) {
  const intervals = unsupportedIntervals(points);
  if (!intervals.length) return;
  intervals.forEach(interval => {
    const x1 = scaleX(interval.start.source_frame_index);
    const x2 = scaleX(interval.end.source_frame_index);
    ctx.fillStyle = "rgba(154, 100, 0, 0.08)";
    ctx.fillRect(Math.min(x1, x2), margin.top, Math.max(Math.abs(x2 - x1), 4), plotH);
  });
}

function unsupportedIntervals(points) {
  const intervals = [];
  let current = null;
  points.forEach(point => {
    if (point.value === null) {
      if (!current) current = {start: point, end: point};
      current.end = point;
      return;
    }
    if (current) {
      intervals.push(current);
      current = null;
    }
  });
  if (current) intervals.push(current);
  return intervals;
}

function metricDetailCards(spec) {
  const whole = result.metric_explorer.whole_movement_statistics[spec.metric_name] || {};
  const phaseStats = phaseStatsForMetric(spec.metric_name);
  const concept = activeConcept();
  if (inspectionMode === "feature" && concept) {
    const side = activeSideOption();
    return `
      <div class="feature-card">
        <span>${FEATURE_CATEGORY_LABELS[concept.category] || "Movement Feature"}</span>
        <h3>${concept.label}</h3>
        <p class="hint">${concept.description}</p>
        <p class="hint">Current view: <strong>${side?.label || "Feature"}</strong> · <strong>${concept.angular ? (featureViewMode === "change" ? "Change" : "Angle") : "Trajectory"}</strong></p>
        <button data-evidence-feature="${spec.metric_name}">Why?</button>
      </div>
      <div class="feature-card">
        <span>Underlying evidence</span>
        <h3>${spec.display_label}</h3>
        <p class="hint">Completeness: <strong>${pct(whole.completeness)}</strong> | supported N ${whole.supported_n ?? 0}/${whole.relevant_n ?? 0}</p>
        <p class="hint">Technical metric: <span class="mono">${spec.metric_name}</span></p>
      </div>
    `;
  }
  return `
    <div class="feature-card">
      <span>${result.metric_explorer.metric_category_labels[spec.metric_family] || spec.metric_family}</span>
      <h3>${spec.display_label}</h3>
      <p class="hint">Preferred visualization: <strong>${spec.preferred_visualisation.replaceAll("_", " ")}</strong></p>
      <p class="hint">Whole movement completeness: <strong>${pct(whole.completeness)}</strong> | supported N ${whole.supported_n ?? 0}/${whole.relevant_n ?? 0}</p>
      <p class="hint">${spec.evidence_note}</p>
      <button data-evidence-feature="${spec.metric_name}">Why?</button>
    </div>
    <div class="feature-card">
      <span>Selected phase context</span>
      <h3>${activePhase() ? `Phase ${activePhase().phase_index}` : "Whole movement"}</h3>
      <p class="hint">Phase mean: <strong>${fmt(phaseStats?.mean, 2)} ${spec.unit || ""}</strong></p>
      <p class="hint">Whole mean: <strong>${fmt(whole.mean, 2)} ${spec.unit || ""}</strong></p>
      <p class="hint">Phase range: <strong>${fmt(phaseStats?.range, 2)}</strong> | Whole range: <strong>${fmt(whole.range, 2)}</strong></p>
    </div>
  `;
}

function phaseStatsForMetric(metricName) {
  const phase = activePhase();
  if (!phase) return null;
  return (result.metric_explorer.phase_statistics[metricName] || []).find(item => item.phase_id === phase.phase_id);
}

function bindControls() {
  $("frameScrub").min = result.movement_window.movement_start_frame;
  $("frameScrub").max = result.movement_window.movement_end_frame;
  $("frameScrub").oninput = event => {
    selectionMode = SelectionMode.SINGLE_FRAME;
    setFrame(Number(event.target.value), {syncPhase: true});
    renderSelectedPhaseStory();
    renderActiveMetric();
  };
  $("toggleRoi").onchange = () => loadFrameImage();
  $("toggleMask").onchange = () => loadFrameImage();
  $("togglePose").onchange = () => {
    loadFrameImage();
    renderSelectedPhaseStory();
  };
  $("videoCanvas").onmousedown = event => startMaskBrush(event);
  $("videoCanvas").onmousemove = event => continueMaskBrush(event);
  window.addEventListener("mouseup", () => stopMaskBrush());
  $("refineMaskButton").onclick = () => {
    maskRefineMode = !maskRefineMode;
    $("refineMaskButton").classList.toggle("active", maskRefineMode);
    $("maskPromptReadout").textContent = maskRefineMode
      ? "Region marking active: drag on visible target pixels or non-target pixels in the video."
      : "Visible-region marks are saved with human provenance.";
    $("toggleMask").checked = true;
    loadFrameImage();
  };
  $("maskPositiveButton").onclick = () => setMaskPromptLabel("target");
  $("maskNegativeButton").onclick = () => setMaskPromptLabel("opponent");
  $("undoMaskPromptButton").onclick = () => mutateMaskPrompts("undo");
  $("clearMaskPromptsButton").onclick = () => mutateMaskPrompts("clear");
  $("graphCanvas").onclick = event => graphClick(event);
  $("angularChartCanvas").onclick = event => graphClick(event);
  $("playFullButton").onclick = () => playFullMovement();
  $("phasePlaybackButton").onclick = () => togglePhasePlayback();
  $("backOneButton").onclick = () => shiftSingleFrame(-1);
  $("forwardOneButton").onclick = () => shiftSingleFrame(1);
  $("backFiveButton").onclick = () => shiftFiveFrameWindow(-5);
  $("forwardFiveButton").onclick = () => shiftFiveFrameWindow(5);
  $("summaryTab").onclick = () => showEvidenceScreen("bodyRegion");
  $("limitsTab").onclick = () => showEvidenceScreen("limits");
  $("crossCaseTab").onclick = () => showEvidenceScreen("crossCase");
}

function setFrame(frame, options = {}) {
  const start = result.movement_window.movement_start_frame;
  const end = result.movement_window.movement_end_frame;
  currentFrame = Math.min(Math.max(Math.round(frame), start), end);
  const framePhase = phaseForFrame(currentFrame);
  if (options.syncPhase !== false && framePhase && framePhase.phase_id !== activePhaseId) {
    activePhaseId = framePhase.phase_id;
    if (angularScope !== "WHOLE_MOVEMENT") angularScope = framePhase.phase_id;
    renderMovementStory();
  }
  $("frameScrub").value = currentFrame;
  const frameInfo = frameBySource.get(currentFrame);
  const timeText = frameInfo ? `${fmt(frameInfo.movement_end_relative_ms, 1)} ms to Movement End` : "";
  $("frameReadout").textContent = `source frame ${currentFrame} | ${timeText}`;
  const span = Math.max(end - start, 1);
  $("timelineBand").style.left = "0%";
  $("timelineBand").style.width = "100%";
  $("timelineCursor").style.left = `${(currentFrame - start) / span * 100}%`;
  loadFrameImage();
  drawGraph();
  renderSelectionReadout();
  renderMetricStats();
  if (activeMetricName) loadEvidenceForMetric(activeMetricName, currentFrame);
}

function playFullMovement() {
  selectionMode = SelectionMode.WHOLE_MOVEMENT;
  activePhaseId = null;
  angularScope = "WHOLE_MOVEMENT";
  phasePlaybackState = "idle";
  renderMovementStory();
  renderActiveMetric();
  playRange(result.movement_window.movement_start_frame, result.movement_window.movement_end_frame, "whole");
  updatePhasePlaybackButton();
}

function playPhase(replay) {
  if (!activePhaseId) activePhaseId = firstPhase()?.phase_id || null;
  const phase = activePhase();
  if (!phase) return;
  selectionMode = SelectionMode.PHASE;
  angularScope = phase.phase_id;
  if (replay || currentFrame < phase.start_frame || currentFrame > phase.end_frame) {
    setFrame(phase.start_frame, {syncPhase: false});
  }
  renderMovementStory();
  renderActiveMetric();
  phasePlaybackState = "running";
  playRange(currentFrame, phase.end_frame, "phase");
  updatePhasePlaybackButton();
}

function playRange(start, end, mode) {
  pausePlayback();
  playbackMode = mode;
  setFrame(start, {syncPhase: selectionMode !== SelectionMode.WHOLE_MOVEMENT});
  playbackTimer = setInterval(() => {
    if (currentFrame >= end) {
      finishPlayback();
      return;
    }
    setFrame(currentFrame + 1, {syncPhase: selectionMode !== SelectionMode.WHOLE_MOVEMENT});
  }, 110);
}

function pausePlayback() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (playbackMode === "phase" && phasePlaybackState === "running") {
    phasePlaybackState = "idle";
  }
  playbackMode = null;
  updatePhasePlaybackButton();
}

function finishPlayback() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (playbackMode === "phase") {
    phasePlaybackState = "ended";
  }
  playbackMode = null;
  updatePhasePlaybackButton();
}

function togglePhasePlayback() {
  if (playbackMode === "phase" && playbackTimer) {
    pausePlayback();
    return;
  }
  playPhase(phasePlaybackState === "ended");
}

function updatePhasePlaybackButton() {
  const button = $("phasePlaybackButton");
  if (!button) return;
  if (playbackMode === "phase" && playbackTimer) {
    button.textContent = "⏸ Pause phase";
    button.classList.add("active");
  } else if (phasePlaybackState === "ended") {
    button.textContent = "↻ Replay phase";
    button.classList.remove("active");
  } else {
    button.textContent = "▶ Play phase";
    button.classList.remove("active");
  }
}

function shiftSingleFrame(delta) {
  pausePlayback();
  phasePlaybackState = "idle";
  selectionMode = SelectionMode.SINGLE_FRAME;
  const phase = activePhase();
  const start = phase ? phase.start_frame : result.movement_window.movement_start_frame;
  const end = phase ? phase.end_frame : result.movement_window.movement_end_frame;
  setFrame(clampFrame(currentFrame + delta, start, end), {syncPhase: false});
  renderSelectedPhaseStory();
  renderActiveMetric();
}

function shiftFiveFrameWindow(delta) {
  pausePlayback();
  phasePlaybackState = "idle";
  selectionMode = SelectionMode.FIVE_FRAME_WINDOW;
  const phase = activePhase();
  const start = phase ? phase.start_frame : result.movement_window.movement_start_frame;
  const end = phase ? phase.end_frame : result.movement_window.movement_end_frame;
  const maxAnchor = Math.max(start, end - 4);
  const base = fiveFrameAnchor === null ? currentFrame : fiveFrameAnchor;
  fiveFrameAnchor = clampFrame(base + delta, start, maxAnchor);
  setFrame(fiveFrameAnchor, {syncPhase: false});
  renderSelectedPhaseStory();
  renderActiveMetric();
}

function loadFrameImage() {
  const roi = $("toggleRoi").checked ? "1" : "0";
  const pose = $("togglePose").checked ? "1" : "0";
  const mask = $("toggleMask").checked ? "1" : "0";
  const img = new Image();
  img.onload = () => {
    const canvas = $("videoCanvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);
  };
  img.src = `/api/results/frame?case=${encodeURIComponent(caseSlug)}&frame=${currentFrame}&roi=${roi}&pose=${pose}&mask=${mask}&t=${Date.now()}`;
}

function setMaskPromptLabel(label) {
  maskPromptLabel = label;
  $("maskPositiveButton").classList.toggle("active", label === "target");
  $("maskNegativeButton").classList.toggle("active", label === "opponent");
}

function maskPromptLabelText(label = maskPromptLabel) {
  return label === "target" ? "target region" : "non-target region";
}

function maskCanvasPoint(event) {
  const canvas = $("videoCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  };
}

function addMaskBrushPoint(event) {
  const point = maskCanvasPoint(event);
  if (lastMaskBrushPoint) {
    const dx = point.x - lastMaskBrushPoint.x;
    const dy = point.y - lastMaskBrushPoint.y;
    if (Math.hypot(dx, dy) < MASK_BRUSH_STEP_PX) return;
  }
  maskBrushPoints.push(point);
  lastMaskBrushPoint = point;
  $("maskPromptReadout").textContent = `Marking ${maskPromptLabelText()} on frame ${currentFrame}: ${maskBrushPoints.length} sampled point${maskBrushPoints.length === 1 ? "" : "s"}.`;
}

function startMaskBrush(event) {
  if (!maskRefineMode) return;
  event.preventDefault();
  maskBrushActive = true;
  maskBrushPoints = [];
  lastMaskBrushPoint = null;
  addMaskBrushPoint(event);
}

function continueMaskBrush(event) {
  if (!maskRefineMode || !maskBrushActive) return;
  event.preventDefault();
  addMaskBrushPoint(event);
}

async function stopMaskBrush() {
  if (!maskBrushActive) return;
  maskBrushActive = false;
  const points = maskBrushPoints;
  maskBrushPoints = [];
  lastMaskBrushPoint = null;
  if (!points.length) return;
  await saveMaskBrushPrompts(points);
}

async function saveMaskBrushPrompts(points) {
  let lastData = null;
  for (const point of points) {
    lastData = await saveMaskPromptAt(point.x, point.y);
  }
  $("maskPromptReadout").textContent = `Saved ${points.length} ${maskPromptLabelText()} sample${points.length === 1 ? "" : "s"} at frame ${currentFrame}. Total prompts: ${lastData?.prompts?.length ?? 0}.`;
  $("toggleMask").checked = true;
  loadFrameImage();
}

async function saveMaskPromptAt(x, y) {
  const response = await fetch("/api/results/mask-prompt", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      case: caseSlug,
      frame: currentFrame,
      x,
      y,
      label: maskPromptLabel
    })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function mutateMaskPrompts(action) {
  const response = await fetch(`/api/results/mask-prompts/${action}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      case: caseSlug,
      frame: currentFrame
    })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  const verb = action === "undo" ? "Undid latest visible-region mark" : "Cleared visible-region marks";
  $("maskPromptReadout").textContent = `${verb} for frame ${currentFrame}. Total prompts: ${data.prompts?.length ?? 0}.`;
  $("toggleMask").checked = true;
  loadFrameImage();
}

function renderActiveGroup() {
  if (activeGroup === "evidence") {
    $("tabTitle").textContent = "Evidence";
    $("featureCards").innerHTML = `<div class="feature-card"><strong>Evidence is shown in the right panel.</strong><p class="hint">Use Body Regions, Why limited, and Cross-case tabs to inspect data support without opening raw files.</p></div>`;
    drawEvidenceGraph();
    return;
  }
  const title = activeGroup.replaceAll("_", " ").replace(/\b\w/g, m => m.toUpperCase()).replace("Trunk Pelvis", "Trunk & Pelvis");
  $("tabTitle").textContent = title;
  const groupFeatures = result.feature_groups[activeGroup] || [];
  $("featureCards").innerHTML = groupFeatures.map(item => featureCard(result.feature_cards[item.feature_name])).join("");
  [...document.querySelectorAll("[data-evidence-feature]")].forEach(button => {
    button.onclick = () => {
      activeFeature = button.dataset.evidenceFeature;
      loadEvidence(activeFeature, currentFrame);
      drawGraph();
    };
  });
  drawGraph();
}

function featureCard(card) {
  if (!card) return "";
  const endValue = card.at_movement_end === null ? "Unavailable" : `${fmt(card.at_movement_end, 2)} ${card.unit}`;
  return `
    <div class="feature-card">
      <span>Projected 2D descriptor</span>
      <h3>${card.display_label}</h3>
      <p><strong class="status ${statusClass(card.sequence_evidence)}">${card.sequence_evidence}</strong></p>
      <p class="hint">Geometry completeness: <strong>${pct(card.geometry_completeness)}</strong></p>
      <p class="hint">Dynamic evidence: <strong>${card.dynamic_evidence}</strong> (${pct(card.dynamic_completeness)})</p>
      <p class="hint">At Movement End: <strong>${endValue}</strong></p>
      <p class="hint">Why limited: ${card.why_limited || "No primary limitation for this displayed feature."}</p>
      <button data-evidence-feature="${card.feature_name}">View evidence</button>
    </div>
  `;
}

function drawGraph() {
  if (!result || !activeMetricName) return;
  const canvas = $("graphCanvas");
  const ctx = canvas.getContext("2d");
  const spec = activeMetricSpec();
  graphLayout = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!spec) return;
  if (spec.preferred_visualisation === "path_2d") {
    drawPathGraph(ctx, canvas);
    return;
  }
  const margin = {left: 72, right: 24, top: 30, bottom: 42};
  const frames = result.frames.map(frame => Number(frame.source_frame_index));
  const minFrame = Math.min(...frames);
  const maxFrame = Math.max(...frames);
  const plotW = canvas.width - margin.left - margin.right;
  const plotH = canvas.height - margin.top - margin.bottom;
  const globalScaleX = frame => margin.left + ((frame - minFrame) / Math.max(maxFrame - minFrame, 1e-9)) * plotW;
  drawSelectionBackground(ctx, margin, globalScaleX, plotH);
  const plottedMetrics = [activeMetricName];
  if (spec.paired_metric_name && result.metric_explorer.series[spec.paired_metric_name]) {
    plottedMetrics.push(spec.paired_metric_name);
  }
  const allValues = plottedMetrics.flatMap(name => metricSeries(name).map(p => p.value).filter(v => v !== null));
  const minV = allValues.length ? Math.min(...allValues) : 0;
  const maxV = allValues.length ? Math.max(...allValues) : 1;
  const scaleY = value => margin.top + plotH - ((value - minV) / Math.max(maxV - minV, 1e-9)) * plotH;
  drawAxes(ctx, canvas, margin, plotW, plotH, metricYAxisLabel(spec), "Frame number", frameTicks(minFrame, maxFrame), globalScaleX);
  plottedMetrics.forEach((metricName, index) => {
    const points = metricSeries(metricName);
    ctx.strokeStyle = colors[index % colors.length];
    ctx.lineWidth = index === 0 ? 2.4 : 1.7;
    ctx.beginPath();
    let drawing = false;
    points.forEach(point => {
      if (point.value === null) {
        drawing = false;
        return;
      }
      const x = globalScaleX(point.source_frame_index);
      const y = scaleY(point.value);
      if (!drawing) {
        ctx.moveTo(x, y);
        drawing = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
  drawSelectedSamples(ctx, globalScaleX, scaleY, metricSeries(activeMetricName));
  const frameInfo = frameBySource.get(currentFrame);
  if (frameInfo) {
    const x = globalScaleX(currentFrame);
    ctx.strokeStyle = "#9d2735";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, margin.top - 8);
    ctx.lineTo(x, canvas.height - margin.bottom + 8);
    ctx.stroke();
    ctx.fillStyle = "#9d2735";
    ctx.fillText(`Frame ${currentFrame}`, Math.min(x + 6, canvas.width - 96), 18);
  }
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText(spec.display_label, margin.left, 18);
  graphLayout.push({feature: activeMetricName, minFrame, maxFrame, scaleX: globalScaleX, marginLeft: margin.left, marginRight: margin.right});
  $("graphLegend").innerHTML = plottedMetrics.map((metricName, index) => `
    <span><i class="dot" style="background:${colors[index % colors.length]}"></i> ${result.metric_explorer.metrics[metricName]?.display_label || metricName}</span>
  `).join("");
}

function drawSelectionBackground(ctx, margin, scaleX, plotH) {
  const phase = activePhase();
  if (phase) {
    const x1 = scaleX(phase.start_frame);
    const x2 = scaleX(phase.end_frame);
    ctx.fillStyle = "rgba(33, 95, 154, 0.08)";
    ctx.fillRect(Math.min(x1, x2), margin.top - 12, Math.abs(x2 - x1), plotH + 24);
  }
  if (selectionMode === SelectionMode.FIVE_FRAME_WINDOW) {
    const bounds = selectionBounds();
    const startInfo = frameBySource.get(bounds.start);
    const endInfo = frameBySource.get(bounds.end);
    if (startInfo && endInfo) {
      const x1 = scaleX(bounds.start);
      const x2 = scaleX(bounds.end);
      ctx.fillStyle = "rgba(154, 100, 0, 0.18)";
      ctx.fillRect(Math.min(x1, x2), margin.top - 12, Math.max(Math.abs(x2 - x1), 4), plotH + 24);
    }
  }
}

function drawSelectedSamples(ctx, scaleX, scaleY, points) {
  const bounds = selectionBounds();
  const selected = points.filter(point => point.source_frame_index >= bounds.start && point.source_frame_index <= bounds.end && point.value !== null);
  ctx.fillStyle = selectionMode === SelectionMode.FIVE_FRAME_WINDOW ? "#9a6400" : "#9d2735";
  selected.forEach(point => {
    ctx.beginPath();
    ctx.arc(scaleX(point.source_frame_index), scaleY(point.value), 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPathGraph(ctx, canvas) {
  const xSeries = metricSeries("path:compensated_x");
  const ySeries = metricSeries("path:compensated_y");
  const points = xSeries.map((xPoint, index) => ({
    source_frame_index: xPoint.source_frame_index,
    x: xPoint.value,
    y: ySeries[index]?.value,
    path_segment_id: xPoint.path_segment_id || ySeries[index]?.path_segment_id || "",
  })).filter(point => point.x !== null && point.y !== null);
  const margin = {left: 54, right: 28, top: 30, bottom: 38};
  if (!points.length) {
    ctx.fillStyle = "#1d2630";
    ctx.fillText("Camera-compensated projected path unavailable.", 32, 52);
    return;
  }
  const xs = points.map(point => point.x);
  const ys = points.map(point => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scaleX = value => margin.left + ((value - minX) / Math.max(maxX - minX, 1e-9)) * (canvas.width - margin.left - margin.right);
  const scaleY = value => margin.top + (canvas.height - margin.top - margin.bottom) - ((value - minY) / Math.max(maxY - minY, 1e-9)) * (canvas.height - margin.top - margin.bottom);
  ctx.strokeStyle = "#215f9a";
  ctx.lineWidth = 2;
  groupedPathSegments(points).forEach(segment => {
    ctx.beginPath();
    segment.forEach((point, index) => {
      const x = scaleX(point.x);
      const y = scaleY(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  const bounds = selectionBounds();
  ctx.fillStyle = "#9a6400";
  points.filter(point => point.source_frame_index >= bounds.start && point.source_frame_index <= bounds.end).forEach(point => {
    ctx.beginPath();
    ctx.arc(scaleX(point.x), scaleY(point.y), 4, 0, Math.PI * 2);
    ctx.fill();
  });
  const current = points.find(point => point.source_frame_index === currentFrame);
  if (current) {
    ctx.fillStyle = "#9d2735";
    ctx.beginPath();
    ctx.arc(scaleX(current.x), scaleY(current.y), 6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#1d2630";
  ctx.font = "13px sans-serif";
  ctx.fillText("Camera-compensated projected body-center path", margin.left, 18);
  graphLayout.push({
    feature: activeMetricName,
    path2d: true,
    points: points.map(point => ({
      source_frame_index: point.source_frame_index,
      x: scaleX(point.x),
      y: scaleY(point.y),
    })),
  });
  $("graphLegend").innerHTML = `<span><i class="dot" style="background:#215f9a"></i> full path</span><span><i class="dot" style="background:#9a6400"></i> selected interval</span><span><i class="dot" style="background:#9d2735"></i> selected frame</span>`;
}

function renderMetricStats() {
  if (!result || !activeMetricName || !$("metricStatsPanel")) return;
  const spec = activeMetricSpec();
  const series = metricSeries(activeMetricName);
  const bounds = selectionBounds();
  const selectedStats = statsForSeries(series, bounds.start, bounds.end);
  const whole = result.metric_explorer.whole_movement_statistics[activeMetricName] || {};
  const phase = activePhase();
  const phaseStats = phaseStatsForMetric(activeMetricName) || {};
  const single = series.find(point => point.source_frame_index === currentFrame);
  const unit = metricUnitDisplay(spec);
  const metricLabel = spec?.display_label || activeMetricName;
  if ($("metricStatsTitle")) $("metricStatsTitle").textContent = `More statistics: ${metricLabel}`;
  const phaseBars = phaseComparisonBars(activeMetricName, unit);
  const fiveFrameHtml = selectionMode === SelectionMode.FIVE_FRAME_WINDOW ? fiveFrameTable(series, bounds.start, bounds.end, unit) : "";
  $("metricStatsPanel").innerHTML = `
    <div class="stats-grid">
      <div class="stats-card">
        <span>Selected measurement</span>
        <strong>${metricLabel}</strong>
        <p class="hint">Technical name: ${activeMetricName}</p>
        <p class="hint">Unit: ${unit || "relative / unitless"}</p>
      </div>
      <div class="stats-card">
        <span>Selection</span>
        <strong>${selectionMode.replaceAll("_", " ")}</strong>
        <p class="hint">Frames ${bounds.start}-${bounds.end}</p>
        <p class="hint">Mean ${valueFmt(selectedStats.mean, 2, unit)} | N ${selectedStats.supported_n}/${selectedStats.relevant_n}</p>
        <p class="hint">Median ${valueFmt(selectedStats.median, 2, unit)} | SD ${valueFmt(selectedStats.standard_deviation, 2, unit)} | Range ${valueFmt(selectedStats.range, 2, unit)}</p>
      </div>
      <div class="stats-card">
        <span>Single Frame</span>
        <strong>source frame ${currentFrame}</strong>
        <p class="hint">Value ${valueFmt(single?.value, 2, unit)}</p>
        <p class="hint">Evidence ${single?.evidence_status || "UNAVAILABLE"}</p>
      </div>
      <div class="stats-card">
        <span>${phase ? `Phase ${phase.phase_index}` : "Selected Phase"}</span>
        <strong>${phase?.title || "Unavailable"}</strong>
        <p class="hint">Mean ${valueFmt(phaseStats.mean, 2, unit)} | Range ${valueFmt(phaseStats.range, 2, unit)} | Completeness ${pct(phaseStats.completeness)}</p>
        <p class="hint">Start ${valueFmt(phaseStats.start_value, 2, unit)} to End ${valueFmt(phaseStats.end_value, 2, unit)} | Change ${signedFmt(phaseStats.change, 2, unit)}</p>
      </div>
      <div class="stats-card">
        <span>Whole Movement</span>
        <strong>${valueFmt(whole.mean, 2, unit)}</strong>
        <p class="hint">Median ${valueFmt(whole.median, 2, unit)} | Range ${valueFmt(whole.range, 2, unit)}</p>
        <p class="hint">Supported N ${whole.supported_n ?? 0}/${whole.relevant_n ?? 0} | Completeness ${pct(whole.completeness)}</p>
      </div>
    </div>
    ${fiveFrameHtml}
    ${phaseBars}
  `;
}

function statsForSeries(series, start, end) {
  const rows = series.filter(point => point.source_frame_index >= start && point.source_frame_index <= end);
  const values = rows.filter(point => point.evidence_status === "SUPPORTED" && point.value !== null).map(point => Number(point.value));
  if (!values.length) {
    return {
      supported_n: 0,
      relevant_n: rows.length,
      completeness: rows.length ? 0 : null,
      mean: null,
      median: null,
      standard_deviation: null,
      minimum: null,
      maximum: null,
      q1: null,
      q3: null,
      iqr: null,
      range: null,
      start_value: null,
      end_value: null,
      change: null,
      absolute_change: null,
      total_absolute_change: null,
      start_frame: null,
      end_frame: null,
      minimum_frame: null,
      maximum_frame: null,
      peak_frame_to_frame_change_frame: null,
    };
  }
  const supportedRows = rows.filter(point => point.evidence_status === "SUPPORTED" && point.value !== null);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const median = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  const variance = values.length > 1 ? values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / (values.length - 1) : 0;
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  const first = supportedRows[0];
  const last = supportedRows[supportedRows.length - 1];
  const change = Number(last.value) - Number(first.value);
  const minRow = supportedRows.reduce((best, point) => Number(point.value) < Number(best.value) ? point : best, supportedRows[0]);
  const maxRow = supportedRows.reduce((best, point) => Number(point.value) > Number(best.value) ? point : best, supportedRows[0]);
  let totalAbsolute = 0;
  let peakFrame = null;
  let peakChange = 0;
  for (let index = 1; index < supportedRows.length; index += 1) {
    const delta = Number(supportedRows[index].value) - Number(supportedRows[index - 1].value);
    totalAbsolute += Math.abs(delta);
    if (Math.abs(delta) >= Math.abs(peakChange)) {
      peakChange = delta;
      peakFrame = supportedRows[index].source_frame_index;
    }
  }
  return {
    supported_n: values.length,
    relevant_n: rows.length,
    completeness: rows.length ? values.length / rows.length : null,
    mean,
    median,
    standard_deviation: Math.sqrt(variance),
    minimum: min,
    maximum: max,
    q1,
    q3,
    iqr: q3 - q1,
    range: max - min,
    start_value: Number(first.value),
    end_value: Number(last.value),
    change,
    absolute_change: Math.abs(change),
    total_absolute_change: totalAbsolute,
    start_frame: first.source_frame_index,
    end_frame: last.source_frame_index,
    minimum_frame: minRow.source_frame_index,
    maximum_frame: maxRow.source_frame_index,
    peak_frame_to_frame_change: peakChange,
    peak_frame_to_frame_change_frame: peakFrame,
  };
}

function quantile(sortedValues, q) {
  if (!sortedValues.length) return null;
  const position = (sortedValues.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sortedValues[lower];
  const weight = position - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function fiveFrameTable(series, start, end, unit) {
  const rows = series.filter(point => point.source_frame_index >= start && point.source_frame_index <= end);
  return `
    <div class="stats-card">
      <span>Five-frame values</span>
      <table class="frame-table">
        <thead><tr><th>Frame</th><th>Time</th><th>Value</th><th>Evidence</th></tr></thead>
        <tbody>${rows.map(point => `
          <tr>
            <td>${point.source_frame_index}</td>
            <td>${fmt(point.movement_end_relative_ms, 1)} ms</td>
            <td>${valueFmt(point.value, 2, unit)}</td>
            <td>${point.evidence_status}</td>
          </tr>
        `).join("")}</tbody>
      </table>
    </div>
  `;
}

function phaseComparisonBars(metricName, unit) {
  const stats = result.metric_explorer.phase_statistics[metricName] || [];
  const values = stats.map(item => item.mean).filter(value => value !== null && value !== undefined);
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  return `
    <div class="stats-card">
      <span>Phase mean comparison</span>
      <div class="phase-bars">${stats.map(item => {
        const value = item.mean;
        const width = value === null || value === undefined ? 0 : ((value - min) / Math.max(max - min, 1e-9) * 80 + 10);
        return `
          <div class="phase-bar-row">
            <strong>Phase ${item.phase_index}</strong>
            <div class="phase-bar-track"><div class="phase-bar-fill" style="width:${width}%"></div></div>
            <span>${valueFmt(value, 1, unit)}</span>
          </div>
        `;
      }).join("")}</div>
    </div>
  `;
}

function drawEvidenceGraph() {
  const canvas = $("graphCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#1d2630";
  ctx.font = "18px sans-serif";
  ctx.fillText("Evidence is summarized in the right panel.", 32, 52);
  ctx.font = "14px sans-serif";
  ctx.fillText("No cross-case analytics are run for the first human-validated case.", 32, 84);
  $("graphLegend").innerHTML = "";
}

function graphClick(event) {
  if (!graphLayout.length) return;
  const targetCanvas = event.currentTarget || $("graphCanvas");
  const rect = targetCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * targetCanvas.width / rect.width;
  const layout = graphLayout[0];
  if (layout.path2d) {
    const y = (event.clientY - rect.top) * targetCanvas.height / rect.height;
    const point = layout.points.reduce((best, item) => (
      Math.hypot(item.x - x, item.y - y) < Math.hypot(best.x - x, best.y - y) ? item : best
    ), layout.points[0]);
    selectionMode = SelectionMode.SINGLE_FRAME;
    setFrame(point.source_frame_index, {syncPhase: true});
    renderSelectedPhaseStory();
    renderActiveMetric();
    return;
  }
  const left = layout.marginLeft || 72;
  const right = layout.marginRight || 24;
  const fraction = (x - left) / Math.max(targetCanvas.width - left - right, 1);
  const targetFrame = layout.minFrame + Math.min(Math.max(fraction, 0), 1) * (layout.maxFrame - layout.minFrame);
  const frame = result.frames.reduce((best, item) => (
    Math.abs(Number(item.source_frame_index) - targetFrame) < Math.abs(Number(best.source_frame_index) - targetFrame) ? item : best
  ), result.frames[0]);
  selectionMode = SelectionMode.SINGLE_FRAME;
  setFrame(frame.source_frame_index, {syncPhase: true});
  renderSelectedPhaseStory();
  renderActiveMetric();
}

async function loadEvidence(feature, frame) {
  activeFeature = feature;
  const detail = await api(`/api/results/evidence?case=${encodeURIComponent(caseSlug)}&feature=${encodeURIComponent(feature)}&frame=${frame}`);
  const value = detail.feature_value === null ? "Unavailable" : `${fmt(detail.feature_value, 3)} ${detail.unit}`;
  $("evidenceDetail").innerHTML = `
Feature: ${detail.display_label}
Canonical name: ${detail.feature_name}
Value at selected point: ${value}
Source frame: ${detail.source_frame_index}
Timestamp: ${fmt(detail.timestamp_ms, 1)} ms
Movement time: ${fmt(detail.movement_end_relative_ms, 1)} ms to Movement End

Landmarks used: ${detail.landmarks_used.join(", ") || "None"}
Frame QC: ${detail.frame_status_text}
Feature status: ${detail.feature_status_text}
Robust dynamic status: ${detail.dynamic_status_text}
Reason withheld: ${detail.rejection_reason || detail.dynamic_rejection_reason || "No rejection reason for this point."}
Input provenance: ${detail.input_provenance.observed_or_interpolated}, smoothed=${detail.input_provenance.smoothed}

Advanced detail
${JSON.stringify(detail.advanced, null, 2)}

Landmark QC
${JSON.stringify(detail.landmark_qc, null, 2)}
  `;
}

async function loadEvidenceForMetric(metricName, frame) {
  activeFeature = metricName;
  const spec = result.metric_explorer.metrics[metricName];
  if (!spec) return;
  if (metricName.startsWith("path:")) {
    const point = metricSeries(metricName).find(item => item.source_frame_index === frame);
    $("evidenceDetail").innerHTML = `
Metric: ${spec.display_label}
Canonical name: ${metricName}
Source frame: ${frame}
Value: ${fmt(point?.value, 3)} ${spec.unit || ""}
Evidence: ${point?.evidence_status || "UNAVAILABLE"}
Reason withheld: ${point?.quality_reason || "No rejection reason for this point."}

Path evidence
Camera-compensated path metrics use the M5.7 sparse-background optical-flow estimate with the target ROI masked. They are projected image-plane descriptors, not true pitch speed or distance.
    `;
    return;
  }
  const featureName = metricName.startsWith("dynamic_rate:") ? metricName.replace("dynamic_rate:", "") : metricName;
  await loadEvidence(featureName, frame);
}

function renderBodyRegionEvidence() {
  $("bodyRegionScreen").innerHTML = `<div class="region-grid">${result.body_region_evidence.map(item => `
    <div class="region-card">
      <h3>${item.display_label}</h3>
      <p><strong>${item.supported}</strong> supported / <strong>${item.limited}</strong> limited / <strong>${item.unavailable}</strong> unavailable</p>
      <p class="hint">Geometry ${pct(item.geometry_completeness)} | Dynamic ${pct(item.dynamic_completeness)}</p>
    </div>
  `).join("")}</div>`;
}

function renderLimitations() {
  $("limitsScreen").innerHTML = result.quality_limitations.slice(0, 12).map(item => `
    <div class="limitation-row">
      <strong>${item.label}</strong>
      <p class="hint">${item.source} | ${item.count} affected rows/frames</p>
    </div>
  `).join("");
}

function renderCrossCase() {
  $("crossCaseScreen").innerHTML = Object.values(result.cross_case_analytics).map(item => `
    <div class="limitation-row">
      <strong>${item.label}</strong>
      <p class="hint">${item.reason}</p>
    </div>
  `).join("");
}

function showEvidenceScreen(name) {
  [["summaryTab", "bodyRegionScreen", "bodyRegion"], ["limitsTab", "limitsScreen", "limits"], ["crossCaseTab", "crossCaseScreen", "crossCase"]].forEach(([buttonId, screenId, key]) => {
    $(buttonId).classList.toggle("active", key === name);
    $(screenId).classList.toggle("active", key === name);
  });
}

init().catch(error => {
  document.body.innerHTML = `<main><section class="panel"><h1>Results unavailable</h1><p>${error.message}</p><p><a href="/">Back to annotation</a></p></section></main>`;
});
</script>
</body>
</html>
"""
