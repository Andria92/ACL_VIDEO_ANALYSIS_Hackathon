"""Extract raw pose coordinates from a manually selected video ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.annotations.models import (
    HumanAnnotationSession,
    TargetUnavailableIntervalAnnotation,
)
from acl_motion.annotations.storage import load_human_annotation_session
from acl_motion.pose.mediapipe_backend import MediaPipePoseBackend
from acl_motion.pose.models import (
    Landmark,
    PoseFrame,
    PoseSequence,
    QualityFlag,
    QualityFlagCode,
)
from acl_motion.pose.yolo_backend import YoloPoseBackend
from acl_motion.video.io import read_video_metadata
from acl_motion.video.roi import BBox, RoiTimeline

DEFAULT_MEDIAPIPE_MODEL_PATH = Path("data/models/pose_landmarker_lite.task")
DEFAULT_YOLO_MODEL_PATH = Path("data/models/yolov8n-pose.pt")
DEFAULT_YOLO_IMAGE_SIZE = 640
DEFAULT_YOLO_DETECTION_CONFIDENCE = 0.25
DEFAULT_YOLO_LANDMARK_CONFIDENCE = 0.25
DEFAULT_YOLO_IOU_THRESHOLD = 0.70
DEFAULT_YOLO_TEMPORAL_MAX_GAP_FRAMES = 12


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import pandas as pd

    if args.roi and args.roi_keyframes:
        raise ValueError("Use either --roi or --roi-keyframes, not both.")
    static_roi = BBox.from_string(args.roi) if args.roi else None
    roi_timeline = RoiTimeline.from_csv(args.roi_keyframes) if args.roi_keyframes else None
    human_session = (
        load_human_annotation_session(args.annotation_session)
        if args.annotation_session
        else None
    )
    metadata = read_video_metadata(video_path)
    backend = build_backend(
        args.backend,
        model_path=args.model_path,
        yolo_selection_strategy=args.yolo_selection_strategy,
        yolo_image_size=args.yolo_image_size,
        yolo_detection_confidence=args.yolo_detection_confidence,
        yolo_landmark_confidence=args.yolo_landmark_confidence,
        yolo_iou_threshold=args.yolo_iou_threshold,
        yolo_temporal_max_gap_frames=args.yolo_temporal_max_gap_frames,
    )

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        start_frame = max(args.start_frame, 0)
        end_frame = args.end_frame if args.end_frame is not None else metadata.frame_count - 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        while current <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if (current - start_frame) % args.every_n == 0:
                timestamp_ms = (current / metadata.fps) * 1000 if metadata.fps else 0.0
                unavailable = (
                    human_session.target_unavailable_interval_at(current)
                    if human_session is not None
                    else None
                )
                if unavailable is not None:
                    reset_tracking = getattr(backend, "reset_tracking", None)
                    if callable(reset_tracking):
                        reset_tracking()
                    pose_frame = target_unavailable_pose_frame(
                        backend=backend,
                        frame=frame,
                        frame_index=current,
                        timestamp_ms=timestamp_ms,
                        source_id=args.source_id,
                        interval=unavailable,
                    )
                else:
                    roi = pose_extraction_roi(
                        frame_index=current,
                        roi_timeline=roi_timeline,
                        static_roi=static_roi,
                        padding_fraction=args.roi_pad,
                    )
                    pose_frame = backend.extract_frame(
                        frame,
                        roi=roi,
                        frame_index=current,
                        timestamp_ms=timestamp_ms,
                        source_id=args.source_id,
                    )
                frames.append(pose_frame)
            current += 1
            if args.max_frames is not None and len(frames) >= args.max_frames:
                break
    finally:
        capture.release()
        backend.close()

    sequence = PoseSequence(
        case_id=args.case_id,
        source_id=args.source_id,
        backend=backend.name,
        frames=tuple(frames),
        metadata={
            "video_path": str(video_path),
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": metadata.frame_count,
            "duration_seconds": metadata.duration_seconds,
        },
    )
    pose_df = pd.DataFrame(list(sequence.iter_landmark_rows()))
    add_traceability_and_annotation_provenance(
        pose_df,
        args,
        roi_timeline,
        human_session,
    )
    save_table(pose_df, output_path)

    metadata_path = args.metadata_output
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".metadata.json")
    write_run_metadata(
        Path(metadata_path),
        args,
        metadata,
        row_count=len(pose_df),
        frame_count=len(frames),
        roi_timeline=roi_timeline,
        human_session=human_session,
    )

    print(f"Wrote {len(pose_df)} landmark rows across {len(frames)} frames to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to local video clip.")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "mediapipe"])
    parser.add_argument("--output", required=True, help="Output .parquet or .csv path.")
    parser.add_argument("--case-id", default=None, help="Optional documented ACL case id.")
    parser.add_argument("--source-id", default=None, help="Video source id. Defaults to video stem.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--every-n", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--roi", default=None, help="Manual ROI as x,y,width,height.")
    parser.add_argument(
        "--roi-keyframes",
        default=None,
        help="CSV with frame_index,x,y,width,height manual ROI keyframes.",
    )
    parser.add_argument(
        "--annotation-session",
        default=None,
        help=(
            "Human annotation session JSON. Explicit target-unavailable intervals "
            "bypass pose extraction and remain unavailable."
        ),
    )
    parser.add_argument("--roi-pad", type=float, default=0.0, help="Symmetric ROI padding fraction.")
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Pose model path. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--yolo-selection-strategy",
        default="largest",
        choices=["largest", "center", "temporal"],
        help=(
            "When YOLO returns multiple poses, choose the largest box, the box nearest "
            "the crop center, or a temporally continuous target anchored by the human ROI."
        ),
    )
    parser.add_argument(
        "--yolo-image-size",
        type=int,
        default=DEFAULT_YOLO_IMAGE_SIZE,
        help="YOLO inference image size. Larger values may improve small-player localisation.",
    )
    parser.add_argument(
        "--yolo-detection-confidence",
        type=float,
        default=DEFAULT_YOLO_DETECTION_CONFIDENCE,
        help="Minimum confidence for a person detection candidate.",
    )
    parser.add_argument(
        "--yolo-landmark-confidence",
        type=float,
        default=DEFAULT_YOLO_LANDMARK_CONFIDENCE,
        help="Landmark confidence below which a raw joint is flagged as uncertain.",
    )
    parser.add_argument(
        "--yolo-iou-threshold",
        type=float,
        default=DEFAULT_YOLO_IOU_THRESHOLD,
        help="YOLO non-maximum-suppression IoU threshold.",
    )
    parser.add_argument(
        "--yolo-temporal-max-gap-frames",
        type=int,
        default=DEFAULT_YOLO_TEMPORAL_MAX_GAP_FRAMES,
        help="Maximum source-frame gap over which temporal target continuity is used.",
    )
    args = parser.parse_args()
    if args.every_n <= 0:
        parser.error("--every-n must be positive.")
    if args.roi_pad < 0:
        parser.error("--roi-pad cannot be negative.")
    if args.yolo_image_size <= 0:
        parser.error("--yolo-image-size must be positive.")
    for option, value in (
        ("--yolo-detection-confidence", args.yolo_detection_confidence),
        ("--yolo-landmark-confidence", args.yolo_landmark_confidence),
        ("--yolo-iou-threshold", args.yolo_iou_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{option} must be between 0 and 1.")
    if args.yolo_temporal_max_gap_frames < 0:
        parser.error("--yolo-temporal-max-gap-frames cannot be negative.")
    if args.source_id is None:
        args.source_id = Path(args.video).stem
    return args


def build_backend(
    name: str,
    *,
    model_path: str | None,
    yolo_selection_strategy: str,
    yolo_image_size: int = DEFAULT_YOLO_IMAGE_SIZE,
    yolo_detection_confidence: float = DEFAULT_YOLO_DETECTION_CONFIDENCE,
    yolo_landmark_confidence: float = DEFAULT_YOLO_LANDMARK_CONFIDENCE,
    yolo_iou_threshold: float = DEFAULT_YOLO_IOU_THRESHOLD,
    yolo_temporal_max_gap_frames: int = DEFAULT_YOLO_TEMPORAL_MAX_GAP_FRAMES,
):
    if model_path is None:
        model_path = (
            str(DEFAULT_MEDIAPIPE_MODEL_PATH)
            if name == "mediapipe"
            else str(DEFAULT_YOLO_MODEL_PATH)
        )
    if name == "mediapipe":
        return MediaPipePoseBackend(model_path=model_path)
    if name == "yolo":
        return YoloPoseBackend(
            model_path=model_path,
            selection_strategy=yolo_selection_strategy,
            image_size=yolo_image_size,
            detection_confidence_threshold=yolo_detection_confidence,
            confidence_threshold=yolo_landmark_confidence,
            iou_threshold=yolo_iou_threshold,
            temporal_max_gap_frames=yolo_temporal_max_gap_frames,
        )
    raise ValueError(f"Unsupported backend: {name}")


def save_table(pose_df, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        pose_df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        pose_df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end in .parquet or .csv.")


def pose_extraction_roi(
    *,
    frame_index: int,
    roi_timeline: RoiTimeline | None,
    static_roi: BBox | None,
    padding_fraction: float,
) -> BBox | None:
    """Return the effective pose crop while preserving the human ROI itself."""

    roi = roi_timeline.bbox_for_frame(frame_index) if roi_timeline is not None else static_roi
    return roi.pad(padding_fraction) if roi is not None else None


def target_unavailable_pose_frame(
    *,
    backend,
    frame,
    frame_index: int,
    timestamp_ms: float,
    source_id: str,
    interval: TargetUnavailableIntervalAnnotation,
) -> PoseFrame:
    """Return an explicit empty pose without invoking the pose model."""

    height, width = frame.shape[:2]
    message = (
        "Human operator marked target unavailable for source frames "
        f"{interval.start_frame}-{interval.end_frame}: {interval.reason.value}."
    )
    flag = QualityFlag(
        code=QualityFlagCode.HUMAN_TARGET_UNAVAILABLE,
        message=message,
        frame_index=frame_index,
    )
    landmarks = {
        name: Landmark(
            name=name,
            x_px=None,
            y_px=None,
            x_norm=None,
            y_norm=None,
            confidence=None,
            visibility=None,
            presence=None,
            observed=False,
            quality_flags=(flag,),
        )
        for name in backend.landmark_names
    }
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        source_id=source_id,
        backend=backend.name,
        target_bbox=None,
        landmarks=landmarks,
        quality_flags=(flag,),
        metadata={
            "image_width": width,
            "image_height": height,
            "model_name": backend.model_name,
            "pose_extraction_skipped": True,
            "human_target_unavailable": True,
            "human_target_unavailable_reason": interval.reason.value,
            "human_target_unavailable_note": interval.note,
            "human_target_unavailable_start_frame": interval.start_frame,
            "human_target_unavailable_end_frame": interval.end_frame,
        },
    )


def add_traceability_and_annotation_provenance(
    pose_df,
    args: argparse.Namespace,
    roi_timeline,
    human_session: HumanAnnotationSession | None,
) -> None:
    """Add source/analysis frame ids and manual ROI provenance to raw rows."""

    if pose_df.empty:
        return
    frame_order = {
        frame_index: analysis_index
        for analysis_index, frame_index in enumerate(sorted(pose_df["frame_index"].unique()))
    }
    pose_df["source_frame_index"] = pose_df["frame_index"].astype(int)
    pose_df["analysis_frame_index"] = pose_df["frame_index"].map(frame_order).astype(int)
    if roi_timeline is not None:
        keyframe_indices = {keyframe.frame_index for keyframe in roi_timeline.keyframes}
        pose_df["manual_roi_provenance"] = "manual_roi_keyframes"
        pose_df["manual_roi_keyframes_path"] = str(args.roi_keyframes)
        pose_df["manual_roi_keyframe_count"] = len(keyframe_indices)
        pose_df["manual_roi_is_keyframe"] = pose_df["source_frame_index"].isin(keyframe_indices)
    elif args.roi:
        pose_df["manual_roi_provenance"] = "manual_static_roi"
        pose_df["manual_roi_keyframes_path"] = None
        pose_df["manual_roi_keyframe_count"] = 0
        pose_df["manual_roi_is_keyframe"] = False
    else:
        pose_df["manual_roi_provenance"] = "none"
        pose_df["manual_roi_keyframes_path"] = None
        pose_df["manual_roi_keyframe_count"] = 0
        pose_df["manual_roi_is_keyframe"] = False
    pose_df["pose_extraction_roi_padding_fraction"] = float(args.roi_pad)
    if args.backend == "yolo":
        pose_df["yolo_selection_strategy"] = args.yolo_selection_strategy
        pose_df["yolo_image_size"] = int(args.yolo_image_size)
        pose_df["yolo_detection_confidence"] = float(
            args.yolo_detection_confidence
        )
        pose_df["yolo_landmark_confidence"] = float(
            args.yolo_landmark_confidence
        )
        pose_df["yolo_iou_threshold"] = float(args.yolo_iou_threshold)
        pose_df["yolo_temporal_max_gap_frames"] = int(
            args.yolo_temporal_max_gap_frames
        )
    pose_df["human_annotation_session_path"] = args.annotation_session
    pose_df["human_target_unavailable"] = False
    pose_df["human_target_unavailable_reason"] = None
    pose_df["human_target_unavailable_note"] = None
    pose_df["human_target_unavailable_start_frame"] = None
    pose_df["human_target_unavailable_end_frame"] = None
    pose_df["human_target_accepted"] = False
    pose_df["human_target_accepted_note"] = None
    pose_df["human_target_accepted_start_frame"] = None
    pose_df["human_target_accepted_end_frame"] = None
    if human_session is None:
        return
    for interval in human_session.target_unavailable_intervals:
        mask = pose_df["source_frame_index"].between(
            interval.start_frame,
            interval.end_frame,
        )
        pose_df.loc[mask, "manual_roi_provenance"] = "human_target_unavailable_interval"
        pose_df.loc[mask, "manual_roi_is_keyframe"] = False
        pose_df.loc[mask, "human_target_unavailable"] = True
        pose_df.loc[mask, "human_target_unavailable_reason"] = interval.reason.value
        pose_df.loc[mask, "human_target_unavailable_note"] = interval.note
        pose_df.loc[mask, "human_target_unavailable_start_frame"] = interval.start_frame
        pose_df.loc[mask, "human_target_unavailable_end_frame"] = interval.end_frame
    for interval in human_session.target_accepted_intervals:
        mask = pose_df["source_frame_index"].between(
            interval.start_frame,
            interval.end_frame,
        )
        pose_df.loc[mask, "human_target_accepted"] = True
        pose_df.loc[mask, "human_target_accepted_note"] = interval.note
        pose_df.loc[mask, "human_target_accepted_start_frame"] = interval.start_frame
        pose_df.loc[mask, "human_target_accepted_end_frame"] = interval.end_frame


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    video_metadata,
    *,
    row_count: int,
    frame_count: int,
    roi_timeline,
    human_session: HumanAnnotationSession | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = Path(
        args.model_path
        or (
            str(DEFAULT_MEDIAPIPE_MODEL_PATH)
            if args.backend == "mediapipe"
            else str(DEFAULT_YOLO_MODEL_PATH)
        )
    )
    payload = {
        "run_id": f"{args.source_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "case_id": args.case_id,
        "source_id": args.source_id,
        "pose_backend": args.backend,
        "model_name": (
            MediaPipePoseBackend.model_name if args.backend == "mediapipe" else YoloPoseBackend.model_name
        ),
        "model_path": str(resolved_model_path),
        "model_sha256": _file_sha256(resolved_model_path),
        "video_path": str(args.video),
        "fps": video_metadata.fps,
        "width": video_metadata.width,
        "height": video_metadata.height,
        "frame_count_video": video_metadata.frame_count,
        "frame_count_processed": frame_count,
        "landmark_row_count": row_count,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "every_n": args.every_n,
        "roi": args.roi,
        "roi_keyframes": args.roi_keyframes,
        "roi_keyframe_count": len(roi_timeline.keyframes) if roi_timeline is not None else 0,
        "roi_keyframe_records": roi_keyframe_records(roi_timeline),
        "human_annotation_session": args.annotation_session,
        "human_target_unavailable_interval_count": (
            len(human_session.target_unavailable_intervals)
            if human_session is not None
            else 0
        ),
        "human_target_unavailable_frame_count": (
            human_session.manual_target_unavailable_frame_count
            if human_session is not None
            else 0
        ),
        "human_target_unavailable_intervals": (
            [interval.to_dict() for interval in human_session.target_unavailable_intervals]
            if human_session is not None
            else []
        ),
        "human_target_accepted_interval_count": (
            len(human_session.target_accepted_intervals)
            if human_session is not None
            else 0
        ),
        "human_target_accepted_frame_count": (
            human_session.manual_target_accepted_frame_count
            if human_session is not None
            else 0
        ),
        "human_target_accepted_intervals": (
            [interval.to_dict() for interval in human_session.target_accepted_intervals]
            if human_session is not None
            else []
        ),
        "roi_pad": args.roi_pad,
        "yolo_selection_strategy": (
            args.yolo_selection_strategy if args.backend == "yolo" else None
        ),
        "yolo_image_size": args.yolo_image_size if args.backend == "yolo" else None,
        "yolo_detection_confidence": (
            args.yolo_detection_confidence if args.backend == "yolo" else None
        ),
        "yolo_landmark_confidence": (
            args.yolo_landmark_confidence if args.backend == "yolo" else None
        ),
        "yolo_iou_threshold": (
            args.yolo_iou_threshold if args.backend == "yolo" else None
        ),
        "yolo_temporal_max_gap_frames": (
            args.yolo_temporal_max_gap_frames if args.backend == "yolo" else None
        ),
        "notes": (
            "Raw pose extraction only. Manual ROI keyframe records are preserved unchanged; "
            "roi_pad records the symmetric context margin applied only to the pose crop. "
            "Temporal selection combines ROI-center, previous-box overlap, candidate size, "
            "and candidate pose confidence when enabled. "
            "No biomechanical interpretation."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roi_keyframe_records(roi_timeline) -> list[dict]:
    if roi_timeline is None:
        return []
    return [
        {
            "frame_index": int(keyframe.frame_index),
            "x": float(keyframe.bbox.x),
            "y": float(keyframe.bbox.y),
            "width": float(keyframe.bbox.width),
            "height": float(keyframe.bbox.height),
        }
        for keyframe in roi_timeline.keyframes
    ]


if __name__ == "__main__":
    raise SystemExit(main())
