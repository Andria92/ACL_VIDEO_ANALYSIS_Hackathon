"""Pose accuracy metrics against manually annotated or laboratory reference points."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POSE_REFERENCE_VALIDATION_VERSION = "pose_reference_validation_v1"
REQUIRED_REFERENCE_COLUMNS = {
    "source_frame_index",
    "landmark_name",
    "x_px",
    "y_px",
}


def validate_pose_against_reference(
    predicted_pose: pd.DataFrame,
    reference_pose: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate pixel error and body-box-normalized PCK without filling missing joints."""

    missing = REQUIRED_REFERENCE_COLUMNS.difference(reference_pose.columns)
    if missing:
        raise ValueError(f"Reference pose is missing columns: {sorted(missing)}")
    prediction = predicted_pose.copy()
    if "source_frame_index" not in prediction and "frame_index" in prediction:
        prediction["source_frame_index"] = prediction["frame_index"]
    prediction_missing = REQUIRED_REFERENCE_COLUMNS.difference(prediction.columns)
    if prediction_missing:
        raise ValueError(f"Predicted pose is missing columns: {sorted(prediction_missing)}")

    reference = reference_pose.copy()
    if "visible" in reference:
        reference = reference.loc[reference["visible"].fillna(False).astype(bool)].copy()
    reference = reference.rename(columns={"x_px": "reference_x", "y_px": "reference_y"})
    prediction = prediction.rename(columns={"x_px": "predicted_x", "y_px": "predicted_y"})
    prediction_columns = [
        "source_frame_index",
        "landmark_name",
        "predicted_x",
        "predicted_y",
        *[
            column
            for column in (
                "confidence",
                "observed",
                "target_bbox_width",
                "target_bbox_height",
            )
            if column in prediction
        ],
    ]
    merged = reference.merge(
        prediction[prediction_columns],
        on=["source_frame_index", "landmark_name"],
        how="left",
        validate="one_to_one",
    )
    reference_valid = (
        pd.to_numeric(merged["reference_x"], errors="coerce").notna()
        & pd.to_numeric(merged["reference_y"], errors="coerce").notna()
    )
    predicted_valid = (
        pd.to_numeric(merged["predicted_x"], errors="coerce").notna()
        & pd.to_numeric(merged["predicted_y"], errors="coerce").notna()
    )
    if "observed" in merged:
        predicted_valid &= merged["observed"].fillna(False).astype(bool)
    evaluable_reference = merged.loc[reference_valid].copy()
    detected = predicted_valid.loc[evaluable_reference.index]
    evaluated = evaluable_reference.loc[detected].copy()
    evaluated["pixel_error"] = np.hypot(
        evaluated["predicted_x"].astype(float) - evaluated["reference_x"].astype(float),
        evaluated["predicted_y"].astype(float) - evaluated["reference_y"].astype(float),
    )
    evaluated["normalizer_px"] = _normalizers(evaluated)
    evaluated["normalized_error"] = evaluated["pixel_error"] / evaluated["normalizer_px"]
    normalized = evaluated["normalized_error"].replace([np.inf, -np.inf], np.nan).dropna()
    pixel_errors = evaluated["pixel_error"].dropna()

    per_landmark = []
    for landmark_name, group in evaluated.groupby("landmark_name", sort=True):
        errors = group["pixel_error"].dropna()
        normal = group["normalized_error"].replace([np.inf, -np.inf], np.nan).dropna()
        reference_count = int(
            evaluable_reference["landmark_name"].eq(landmark_name).sum()
        )
        per_landmark.append(
            {
                "landmark_name": str(landmark_name),
                "reference_count": reference_count,
                "detected_count": len(errors),
                "detection_rate": round(len(errors) / reference_count, 3)
                if reference_count
                else None,
                "median_pixel_error": _quantile(errors, 0.50),
                "p90_pixel_error": _quantile(errors, 0.90),
                "pck_0_05": _pck(normal, 0.05),
                "pck_0_10": _pck(normal, 0.10),
            }
        )
    reference_count = len(evaluable_reference)
    detected_count = len(evaluated)
    return {
        "validation_version": POSE_REFERENCE_VALIDATION_VERSION,
        "status": "REFERENCE_COMPARISON_AVAILABLE" if reference_count else "NO_REFERENCE_POINTS",
        "reference_point_count": reference_count,
        "detected_reference_point_count": detected_count,
        "detection_rate": (
            round(detected_count / reference_count, 3) if reference_count else None
        ),
        "median_pixel_error": _quantile(pixel_errors, 0.50),
        "p90_pixel_error": _quantile(pixel_errors, 0.90),
        "pck_0_05": _pck(normalized, 0.05),
        "pck_0_10": _pck(normalized, 0.10),
        "normalization": "target bounding-box diagonal unless reference normalizer_px is supplied",
        "per_landmark": per_landmark,
        "interpretation": (
            "PCK reports the fraction of visible reference joints within a normalized distance "
            "threshold. Reference provenance, annotator agreement, and camera calibration must "
            "be assessed separately."
        ),
    }


def _normalizers(frame: pd.DataFrame) -> pd.Series:
    if "normalizer_px" in frame:
        supplied = pd.to_numeric(frame["normalizer_px"], errors="coerce")
    else:
        supplied = pd.Series(np.nan, index=frame.index, dtype=float)
    if "target_bbox_width" in frame and "target_bbox_height" in frame:
        width = pd.to_numeric(frame["target_bbox_width"], errors="coerce")
        height = pd.to_numeric(frame["target_bbox_height"], errors="coerce")
        fallback = np.hypot(width, height)
        supplied = supplied.where(supplied > 0, fallback)
    return supplied.where(supplied > 0, np.nan)


def _quantile(values: pd.Series, quantile: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return round(float(clean.quantile(quantile)), 3) if not clean.empty else None


def _pck(values: pd.Series, threshold: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return round(float((clean <= threshold).mean()), 3) if not clean.empty else None
