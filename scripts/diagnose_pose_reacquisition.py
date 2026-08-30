"""Diagnose where target-pose evidence disappears after an occlusion.

This script is intentionally diagnostic-only. It does not alter pose,
quality, geometry, dynamics, semantics, or UI outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from acl_motion.pose.yolo_backend import COCO_POSE_LANDMARKS
from acl_motion.video.io import read_video_metadata
from acl_motion.video.roi import BBox, RoiTimeline

IMPORTANT_FRAMES = {
    95,
    96,
    97,
    98,
    99,
    100,
    110,
    111,
    113,
    119,
    121,
    122,
    128,
    137,
    144,
    145,
    149,
}
CONFIDENCE_THRESHOLD = 0.25


@dataclass(frozen=True, slots=True)
class Candidate:
    """One raw full-frame YOLO pose candidate."""

    frame: int
    candidate_id: int
    bbox: tuple[float, float, float, float]
    keypoints_xy: np.ndarray
    keypoint_conf: np.ndarray


def main() -> int:
    args = parse_args()
    output_frame_path = Path(args.frame_output)
    output_candidate_path = Path(args.candidate_output)
    output_contact_sheet = Path(args.contact_sheet_output)
    target_segments_output = Path(args.target_segments_output) if args.target_segments_output else None
    for path in (output_frame_path, output_candidate_path, output_contact_sheet, target_segments_output):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_pose = pd.read_parquet(args.raw_pose)
    frame_quality = pd.read_csv(args.frame_quality)
    roi_timeline = RoiTimeline.from_csv(args.roi_keyframes)
    metadata = read_video_metadata(args.video)
    frames = list(range(args.start_frame, args.end_frame + 1))

    candidates_by_frame, images_by_frame = run_yolo_full_frame(
        video_path=Path(args.video),
        model_path=Path(args.model_path),
        frames=frames,
    )
    candidate_rows = build_candidate_rows(
        candidates_by_frame=candidates_by_frame,
        frame_quality=frame_quality,
        roi_timeline=roi_timeline,
        image_width=metadata.width,
        image_height=metadata.height,
    )
    candidate_df = pd.DataFrame(candidate_rows)
    frame_df = pd.DataFrame(
        build_frame_rows(
            frames=frames,
            raw_pose=raw_pose,
            frame_quality=frame_quality,
            candidate_df=candidate_df,
            roi_timeline=roi_timeline,
            fps=metadata.fps,
        )
    )

    frame_df.to_csv(output_frame_path, index=False)
    candidate_df.to_csv(output_candidate_path, index=False)
    if target_segments_output is not None:
        target_segments_output.write_text(
            json.dumps(build_target_segments(frame_quality), indent=2),
            encoding="utf-8",
        )
    draw_contact_sheet(
        images_by_frame=images_by_frame,
        frame_df=frame_df,
        candidate_df=candidate_df,
        roi_timeline=roi_timeline,
        output_path=output_contact_sheet,
    )

    print(f"Wrote frame diagnostics to {output_frame_path}")
    print(f"Wrote candidate diagnostics to {output_candidate_path}")
    if target_segments_output is not None:
        print(f"Wrote target segments to {target_segments_output}")
    print(f"Wrote contact sheet to {output_contact_sheet}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/videos/analysis_clips/02_YvnMYc6OdT8_160s-166s.mp4")
    parser.add_argument("--model-path", default="data/models/yolov8n-pose.pt")
    parser.add_argument("--raw-pose", default="data/pose/human/christen_press_raw_pose.parquet")
    parser.add_argument("--frame-quality", default="data/quality/human/christen_press_frame_quality.csv")
    parser.add_argument("--roi-keyframes", default="data/annotations/human/christen_press_target_roi_human.csv")
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--end-frame", type=int, default=155)
    parser.add_argument(
        "--frame-output",
        default="data/quality/human/christen_press_pose_pipeline_frame_diagnostics.csv",
    )
    parser.add_argument(
        "--candidate-output",
        default="data/quality/human/christen_press_pose_candidate_diagnostics.csv",
    )
    parser.add_argument(
        "--contact-sheet-output",
        default="data/diagnostics/human/christen_press_pose_reacquisition_contact_sheet.png",
    )
    parser.add_argument(
        "--target-segments-output",
        default="data/quality/human/christen_press_target_segments.json",
    )
    return parser.parse_args()


def run_yolo_full_frame(
    *,
    video_path: Path,
    model_path: Path,
    frames: list[int],
) -> tuple[dict[int, list[Candidate]], dict[int, np.ndarray]]:
    """Run raw full-frame YOLO pose inference for selected frames."""

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    wanted = set(frames)
    candidates_by_frame: dict[int, list[Candidate]] = {}
    images_by_frame: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        for frame_index in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = capture.read()
            if not ok:
                continue
            images_by_frame[frame_index] = image
            if frame_index not in wanted:
                continue
            result = model.predict(
                source=image,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
            )[0]
            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None and result.boxes.xyxy is not None
                else np.empty((0, 4))
            )
            keypoints_xy = (
                result.keypoints.xy.cpu().numpy()
                if result.keypoints is not None and result.keypoints.xy is not None
                else np.empty((0, len(COCO_POSE_LANDMARKS), 2))
            )
            keypoint_conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints is not None and result.keypoints.conf is not None
                else np.full((len(keypoints_xy), len(COCO_POSE_LANDMARKS)), np.nan)
            )
            frame_candidates = []
            for candidate_id in range(min(len(boxes), len(keypoints_xy))):
                frame_candidates.append(
                    Candidate(
                        frame=frame_index,
                        candidate_id=candidate_id,
                        bbox=tuple(float(value) for value in boxes[candidate_id]),
                        keypoints_xy=keypoints_xy[candidate_id],
                        keypoint_conf=keypoint_conf[candidate_id],
                    )
                )
            candidates_by_frame[frame_index] = frame_candidates
    finally:
        capture.release()
    return candidates_by_frame, images_by_frame


def build_candidate_rows(
    *,
    candidates_by_frame: dict[int, list[Candidate]],
    frame_quality: pd.DataFrame,
    roi_timeline: RoiTimeline,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    previous_supported = previous_supported_targets(frame_quality)
    rows: list[dict[str, Any]] = []
    for frame, candidates in candidates_by_frame.items():
        roi = roi_timeline.bbox_for_frame(frame).clamp(image_width, image_height)
        selected_id = match_current_selected_candidate(frame_quality, frame, candidates)
        ranked_ids = rank_candidates(candidates, roi)
        for candidate in candidates:
            row = candidate_row(
                candidate,
                roi=roi,
                selected_id=selected_id,
                rank=ranked_ids.index(candidate.candidate_id) + 1,
                previous_supported=previous_supported.get(frame),
                final_status=frame_status(frame_quality, frame),
                final_reason=frame_reason(frame_quality, frame),
            )
            rows.append(row)
    return rows


def candidate_row(
    candidate: Candidate,
    *,
    roi: BBox,
    selected_id: int | None,
    rank: int,
    previous_supported: dict[str, Any] | None,
    final_status: str,
    final_reason: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = candidate.bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    roi_center = (roi.x + roi.width / 2, roi.y + roi.height / 2)
    pelvis = midpoint(candidate, "left_hip", "right_hip")
    shoulders = midpoint(candidate, "left_shoulder", "right_shoulder")
    hips = midpoint(candidate, "left_hip", "right_hip")
    body_scale = distance(pelvis, shoulders)
    previous_frame = previous_supported.get("source_frame_index") if previous_supported else None
    previous_point = (
        (previous_supported.get("pelvis_x"), previous_supported.get("pelvis_y"))
        if previous_supported
        else None
    )
    previous_scale = previous_supported.get("body_scale_px") if previous_supported else np.nan
    continuity_distance = distance(pelvis if point_available(pelvis) else center, previous_point)
    scale_ratio = body_scale / previous_scale if finite_positive(body_scale) and finite_positive(previous_scale) else np.nan
    mean_conf = finite_mean(candidate.keypoint_conf)
    median_conf = finite_median(candidate.keypoint_conf)
    supported_count = int(np.nansum(candidate.keypoint_conf >= CONFIDENCE_THRESHOLD))
    iou = bbox_iou(candidate.bbox, roi)
    inside_fraction = bbox_intersection_area(candidate.bbox, roi) / max((x2 - x1) * (y2 - y1), 1e-9)
    selected = candidate.candidate_id == selected_id
    accepted = selected and final_status == "VALID_TARGET"
    rejection_reason = ""
    if accepted:
        accepted_or_rejected = "ACCEPTED"
    elif selected:
        accepted_or_rejected = "REJECTED_SELECTED_CANDIDATE"
        rejection_reason = final_reason or final_status
    else:
        accepted_or_rejected = "REJECTED_NOT_SELECTED"
        rejection_reason = "Not selected by current target association."

    return {
        "source_frame": int(candidate.frame),
        "candidate_id": int(candidate.candidate_id),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_center_x": center[0],
        "bbox_center_y": center[1],
        "human_roi_iou": iou,
        "fraction_candidate_inside_roi": inside_fraction,
        "candidate_center_inside_roi": point_inside(center, roi),
        "pelvis_midpoint_x": pelvis[0],
        "pelvis_midpoint_y": pelvis[1],
        "pelvis_midpoint_available": point_available(pelvis),
        "distance_candidate_center_to_roi_center": distance(center, roi_center),
        "distance_pelvis_to_roi_center": distance(pelvis, roi_center),
        "mean_keypoint_confidence": mean_conf,
        "median_keypoint_confidence": median_conf,
        "supported_keypoint_count": supported_count,
        "shoulder_midpoint_available": point_available(shoulders),
        "hip_midpoint_available": point_available(hips),
        "previous_supported_target_frame": previous_frame,
        "continuity_distance_from_previous_supported_target": continuity_distance,
        "body_scale_ratio_vs_previous_supported_target": scale_ratio,
        "candidate_rank": int(rank),
        "accepted_or_rejected": accepted_or_rejected,
        "rejection_reason": rejection_reason,
    }


def build_frame_rows(
    *,
    frames: list[int],
    raw_pose: pd.DataFrame,
    frame_quality: pd.DataFrame,
    candidate_df: pd.DataFrame,
    roi_timeline: RoiTimeline,
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        roi = roi_timeline.bbox_for_frame(frame)
        candidates = candidate_df[candidate_df["source_frame"].eq(frame)]
        current_quality = frame_quality[frame_quality["source_frame_index"].astype(int).eq(frame)]
        selected_id = selected_candidate_from_diagnostics(candidates)
        final_status = frame_status(frame_quality, frame)
        final_segment = frame_segment(frame_quality, frame)
        pose_rendered = final_status == "VALID_TARGET"
        yolo_pose_count = len(candidates)
        candidate_count_in_roi = int(
            (
                (candidates["human_roi_iou"] > 0)
                | candidates["candidate_center_inside_roi"].astype(bool)
                | (candidates["fraction_candidate_inside_roi"] > 0)
            ).sum()
        ) if not candidates.empty else 0
        raw_selected_rows = raw_pose[raw_pose["source_frame_index"].astype(int).eq(frame)]
        crop_pose_count = current_backend_pose_count(raw_selected_rows)
        stage, reason = failure_stage(
            final_status=final_status,
            final_reason=frame_reason(frame_quality, frame),
            raw_yolo_pose_count=yolo_pose_count,
            candidate_count_in_roi=candidate_count_in_roi,
            pose_rendered=pose_rendered,
            current_quality=current_quality,
        )
        rows.append(
            {
                "source_frame": int(frame),
                "timestamp_ms": (frame / fps) * 1000 if fps else np.nan,
                "human_roi_present": True,
                "human_roi_x1": roi.x,
                "human_roi_y1": roi.y,
                "human_roi_x2": roi.x2,
                "human_roi_y2": roi.y2,
                "raw_yolo_person_count": yolo_pose_count,
                "raw_yolo_pose_candidate_count": yolo_pose_count,
                "crop_yolo_pose_candidate_count": crop_pose_count,
                "candidate_count_intersecting_human_roi": candidate_count_in_roi,
                "selected_candidate_id": selected_id,
                "candidate_selected_yes_no": selected_id is not None,
                "final_frame_qc": final_status,
                "final_target_segment_id": final_segment,
                "pose_rendered_yes_no": pose_rendered,
                "failure_stage": stage,
                "failure_reason": reason,
            }
        )
    return rows


def build_target_segments(frame_quality: pd.DataFrame) -> dict[str, Any]:
    """Return explicit valid target segments and non-valid status intervals."""

    frame_quality = frame_quality.sort_values("source_frame_index").reset_index(drop=True)
    valid_segments: list[dict[str, Any]] = []
    status_intervals: list[dict[str, Any]] = []
    for segment_id, group in frame_quality.dropna(subset=["valid_segment_id"]).groupby("valid_segment_id"):
        valid_segments.append(
            {
                "valid_segment_id": int(segment_id),
                "start_source_frame": int(group["source_frame_index"].min()),
                "end_source_frame": int(group["source_frame_index"].max()),
                "frame_count": len(group),
            }
        )
    interval_start = None
    previous_status = None
    previous_frame = None
    for _, row in frame_quality.iterrows():
        current_frame = int(row["source_frame_index"])
        current_status = str(row["frame_status"])
        if interval_start is None:
            interval_start = current_frame
            previous_status = current_status
            previous_frame = current_frame
            continue
        if current_status != previous_status or current_frame != int(previous_frame) + 1:
            status_intervals.append(
                {
                    "status": previous_status,
                    "start_source_frame": int(interval_start),
                    "end_source_frame": int(previous_frame),
                    "frame_count": int(previous_frame - interval_start + 1),
                }
            )
            interval_start = current_frame
            previous_status = current_status
        previous_frame = current_frame
    if interval_start is not None and previous_status is not None and previous_frame is not None:
        status_intervals.append(
            {
                "status": previous_status,
                "start_source_frame": int(interval_start),
                "end_source_frame": int(previous_frame),
                "frame_count": int(previous_frame - interval_start + 1),
            }
        )
    return {
        "case_id": str(frame_quality["case_id"].iloc[0]) if "case_id" in frame_quality.columns else None,
        "source_id": str(frame_quality["source_id"].iloc[0]) if "source_id" in frame_quality.columns else None,
        "valid_target_segments": valid_segments,
        "status_intervals": status_intervals,
    }


def failure_stage(
    *,
    final_status: str,
    final_reason: str,
    raw_yolo_pose_count: int,
    candidate_count_in_roi: int,
    pose_rendered: bool,
    current_quality: pd.DataFrame,
) -> tuple[str, str]:
    if raw_yolo_pose_count == 0:
        return "YOLO_NO_PERSON", "Full-frame YOLO returned no person pose candidates."
    if candidate_count_in_roi == 0:
        return "NO_CANDIDATE_IN_ROI", "YOLO candidates did not intersect the human ROI."
    if pose_rendered:
        return "SUPPORTED", "Frame belongs to a valid target segment."
    if final_status == "TARGET_NOT_FOUND":
        return "YOLO_NO_POSE", final_reason
    if final_status == "TARGET_IDENTITY_UNCERTAIN":
        if "TARGET_OVERLAP_AMBIGUOUS" in final_reason:
            return "MULTIPLE_AMBIGUOUS_CANDIDATES", final_reason
        return "TEMPORAL_CONTINUITY_REJECTED", final_reason
    if final_status == "INVALID_TRACK_SEGMENT":
        return "TRACK_STATE_REJECTED", final_reason
    if final_status in {"LOW_POSE_CONFIDENCE", "PARTIAL_POSE"}:
        return "POSE_QC_REJECTED", final_reason
    if not current_quality.empty and not bool(current_quality["pose_centroid_consistent_with_roi"].iloc[0]):
        return "TARGET_SELECTION_REJECTED", final_reason
    return "OTHER_EXPLICIT_REASON", final_reason or final_status


def draw_contact_sheet(
    *,
    images_by_frame: dict[int, np.ndarray],
    frame_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    roi_timeline: RoiTimeline,
    output_path: Path,
) -> None:
    tile_w, tile_h = 320, 210
    columns = 6
    frames = sorted(images_by_frame)
    rows = int(np.ceil(len(frames) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        image = images_by_frame[frame].copy()
        roi = roi_timeline.bbox_for_frame(frame).clamp(image.shape[1], image.shape[0])
        draw_frame_debug(image, frame, roi, frame_df, candidate_df)
        thumb = cv2.resize(image, (tile_w, int(tile_w * image.shape[0] / image.shape[1])))
        if thumb.shape[0] > tile_h:
            thumb = cv2.resize(image, (int(tile_h * image.shape[1] / image.shape[0]), tile_h))
        canvas = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        canvas[: thumb.shape[0], : thumb.shape[1]] = thumb
        r = index // columns
        c = index % columns
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = canvas
    cv2.imwrite(str(output_path), sheet)


def draw_frame_debug(
    image: np.ndarray,
    frame: int,
    roi: BBox,
    frame_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> None:
    x1, y1, x2, y2 = (round(v) for v in (roi.x, roi.y, roi.x2, roi.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 180, 30), 3)
    selected = selected_candidate_from_diagnostics(candidate_df[candidate_df["source_frame"].eq(frame)])
    for _, row in candidate_df[candidate_df["source_frame"].eq(frame)].iterrows():
        color = (60, 60, 240)
        thickness = 2
        if selected is not None and int(row["candidate_id"]) == int(selected):
            color = (0, 190, 255)
            thickness = 4
        bx1, by1, bx2, by2 = (round(row[col]) for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        cv2.rectangle(image, (bx1, by1), (bx2, by2), color, thickness)
        cv2.putText(
            image,
            f"id {int(row['candidate_id'])}",
            (bx1, max(18, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    status_row = frame_df[frame_df["source_frame"].eq(frame)]
    status = str(status_row["final_frame_qc"].iloc[0]) if not status_row.empty else "UNKNOWN"
    stage = str(status_row["failure_stage"].iloc[0]) if not status_row.empty else "UNKNOWN"
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (255, 255, 255), -1)
    cv2.putText(
        image,
        f"frame {frame} | {status} | {stage}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def previous_supported_targets(frame_quality: pd.DataFrame) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    for _, row in frame_quality.sort_values("source_frame_index").iterrows():
        frame = int(row["source_frame_index"])
        output[frame] = previous
        if row.get("frame_status") == "VALID_TARGET":
            previous = row.to_dict()
    return output


def match_current_selected_candidate(
    frame_quality: pd.DataFrame,
    frame: int,
    candidates: list[Candidate],
) -> int | None:
    row = frame_quality[frame_quality["source_frame_index"].astype(int).eq(frame)]
    if row.empty or not candidates:
        return None
    center = (float(row["pose_centroid_x"].iloc[0]), float(row["pose_centroid_y"].iloc[0]))
    if not point_available(center):
        return None
    return min(candidates, key=lambda candidate: distance(candidate_center(candidate), center)).candidate_id


def selected_candidate_from_diagnostics(candidates: pd.DataFrame) -> int | None:
    if candidates.empty:
        return None
    selected = candidates[candidates["accepted_or_rejected"].isin(["ACCEPTED", "REJECTED_SELECTED_CANDIDATE"])]
    if selected.empty:
        return None
    return int(selected.sort_values("candidate_rank").iloc[0]["candidate_id"])


def rank_candidates(candidates: list[Candidate], roi: BBox) -> list[int]:
    roi_center = (roi.x + roi.width / 2, roi.y + roi.height / 2)
    return [
        candidate.candidate_id
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                not point_inside(candidate_center(candidate), roi),
                -bbox_iou(candidate.bbox, roi),
                distance(candidate_center(candidate), roi_center),
            ),
        )
    ]


def current_backend_pose_count(raw_selected_rows: pd.DataFrame) -> int:
    if raw_selected_rows.empty:
        return 0
    metadata = raw_selected_rows["backend_metadata"].iloc[0]
    if isinstance(metadata, dict):
        return int(metadata.get("pose_count", 0) or 0)
    return 0


def frame_status(frame_quality: pd.DataFrame, frame: int) -> str:
    row = frame_quality[frame_quality["source_frame_index"].astype(int).eq(frame)]
    return str(row["frame_status"].iloc[0]) if not row.empty else "UNKNOWN"


def frame_reason(frame_quality: pd.DataFrame, frame: int) -> str:
    row = frame_quality[frame_quality["source_frame_index"].astype(int).eq(frame)]
    if row.empty:
        return ""
    value = row["frame_rejection_reason"].iloc[0]
    return "" if pd.isna(value) else str(value)


def frame_segment(frame_quality: pd.DataFrame, frame: int) -> int | None:
    row = frame_quality[frame_quality["source_frame_index"].astype(int).eq(frame)]
    if row.empty or pd.isna(row["valid_segment_id"].iloc[0]):
        return None
    return int(row["valid_segment_id"].iloc[0])


def midpoint(candidate: Candidate, left: str, right: str) -> tuple[float, float]:
    left_point = keypoint(candidate, left)
    right_point = keypoint(candidate, right)
    if not point_available(left_point) or not point_available(right_point):
        return np.nan, np.nan
    return (left_point[0] + right_point[0]) / 2, (left_point[1] + right_point[1]) / 2


def keypoint(candidate: Candidate, name: str) -> tuple[float, float]:
    index = COCO_POSE_LANDMARKS.index(name)
    if index >= len(candidate.keypoints_xy):
        return np.nan, np.nan
    conf = candidate.keypoint_conf[index] if index < len(candidate.keypoint_conf) else np.nan
    point = candidate.keypoints_xy[index]
    if not np.isfinite(conf) or conf < CONFIDENCE_THRESHOLD:
        return np.nan, np.nan
    return float(point[0]), float(point[1])


def candidate_center(candidate: Candidate) -> tuple[float, float]:
    x1, y1, x2, y2 = candidate.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_iou(box: tuple[float, float, float, float], roi: BBox) -> float:
    intersection = bbox_intersection_area(box, roi)
    x1, y1, x2, y2 = box
    area_box = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_roi = roi.width * roi.height
    union = area_box + area_roi - intersection
    return intersection / union if union > 0 else 0.0


def bbox_intersection_area(box: tuple[float, float, float, float], roi: BBox) -> float:
    x1, y1, x2, y2 = box
    ix1 = max(x1, roi.x)
    iy1 = max(y1, roi.y)
    ix2 = min(x2, roi.x2)
    iy2 = min(y2, roi.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def point_inside(point: tuple[float, float], roi: BBox) -> bool:
    return point_available(point) and roi.x <= point[0] <= roi.x2 and roi.y <= point[1] <= roi.y2


def point_available(point: tuple[float, float] | None) -> bool:
    return point is not None and all(np.isfinite(value) for value in point)


def distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if not point_available(a) or not point_available(b):
        return np.nan
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def finite_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0)


if __name__ == "__main__":
    raise SystemExit(main())
