"""Extract raw pose coordinates from a manually selected video ROI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.pose.mediapipe_backend import MediaPipePoseBackend
from acl_motion.pose.models import PoseSequence
from acl_motion.pose.yolo_backend import YoloPoseBackend
from acl_motion.video.io import read_video_metadata
from acl_motion.video.roi import BBox, RoiTimeline

DEFAULT_MEDIAPIPE_MODEL_PATH = Path("data/models/pose_landmarker_lite.task")
DEFAULT_YOLO_MODEL_PATH = Path("data/models/yolov8n-pose.pt")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import pandas as pd

    if args.roi and args.roi_keyframes:
        raise ValueError("Use either --roi or --roi-keyframes, not both.")
    static_roi = BBox.from_string(args.roi).pad(args.roi_pad) if args.roi else None
    roi_timeline = RoiTimeline.from_csv(args.roi_keyframes) if args.roi_keyframes else None
    metadata = read_video_metadata(video_path)
    backend = build_backend(
        args.backend,
        model_path=args.model_path,
        yolo_selection_strategy=args.yolo_selection_strategy,
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
                roi = roi_timeline.bbox_for_frame(current) if roi_timeline else static_roi
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
    add_traceability_and_annotation_provenance(pose_df, args, roi_timeline)
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
        choices=["largest", "center"],
        help="When YOLO returns multiple poses in the crop, choose largest box or box nearest crop center.",
    )
    args = parser.parse_args()
    if args.every_n <= 0:
        parser.error("--every-n must be positive.")
    if args.source_id is None:
        args.source_id = Path(args.video).stem
    return args


def build_backend(name: str, *, model_path: str | None, yolo_selection_strategy: str):
    if model_path is None:
        model_path = (
            str(DEFAULT_MEDIAPIPE_MODEL_PATH)
            if name == "mediapipe"
            else str(DEFAULT_YOLO_MODEL_PATH)
        )
    if name == "mediapipe":
        return MediaPipePoseBackend(model_path=model_path)
    if name == "yolo":
        return YoloPoseBackend(model_path=model_path, selection_strategy=yolo_selection_strategy)
    raise ValueError(f"Unsupported backend: {name}")


def save_table(pose_df, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        pose_df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        pose_df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end in .parquet or .csv.")


def add_traceability_and_annotation_provenance(pose_df, args: argparse.Namespace, roi_timeline) -> None:
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


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    video_metadata,
    *,
    row_count: int,
    frame_count: int,
    roi_timeline,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": f"{args.source_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "case_id": args.case_id,
        "source_id": args.source_id,
        "pose_backend": args.backend,
        "model_name": (
            MediaPipePoseBackend.model_name if args.backend == "mediapipe" else YoloPoseBackend.model_name
        ),
        "model_path": args.model_path
        or (
            str(DEFAULT_MEDIAPIPE_MODEL_PATH)
            if args.backend == "mediapipe"
            else str(DEFAULT_YOLO_MODEL_PATH)
        ),
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
        "roi_pad": args.roi_pad,
        "yolo_selection_strategy": args.yolo_selection_strategy,
        "notes": "Raw pose extraction only. No biomechanical interpretation.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
