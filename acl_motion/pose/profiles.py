"""The fixed YOLOv8n pose-analysis configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PoseAnalysisProfile:
    """The reproducible settings used for pose extraction."""

    profile_id: str
    label: str
    short_label: str
    plain_language_description: str
    scientific_description: str
    try_when: tuple[str, ...]
    advantages: tuple[str, ...]
    limitations: tuple[str, ...]
    speed_label: str
    backend: str
    model_filename: str
    selection_strategy: str
    image_size: int
    detection_confidence: float
    landmark_confidence: float
    iou_threshold: float
    temporal_max_gap_frames: int
    roi_padding_fraction: float
    recommended: bool = True
    experimental: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


YOLOV8N_PROFILE = PoseAnalysisProfile(
    profile_id="yolov8n_legacy_tight",
    label="YOLOv8n",
    short_label="YOLOv8n · tight ROI",
    plain_language_description=(
        "Uses YOLOv8n with a tight crop around the athlete selected by the reviewer."
    ),
    scientific_description=(
        "YOLOv8n-pose, 640 px inference, largest candidate inside the unpadded "
        "human-annotated ROI."
    ),
    try_when=("Use for all annotated cases in the current workflow.",),
    advantages=(
        "Keeps the established analysis consistent across cases.",
        "Runs quickly on the short reviewed clips.",
    ),
    limitations=(
        "Close player overlap can still mix joints between athletes.",
        "Hidden or blurred joints cannot be recovered reliably from one camera view.",
    ),
    speed_label="Fast",
    backend="yolo",
    model_filename="yolov8n-pose.pt",
    selection_strategy="largest",
    image_size=640,
    detection_confidence=0.25,
    landmark_confidence=0.25,
    iou_threshold=0.70,
    temporal_max_gap_frames=12,
    roi_padding_fraction=0.0,
)

POSE_ANALYSIS_PROFILES: tuple[PoseAnalysisProfile, ...] = (YOLOV8N_PROFILE,)
DEFAULT_POSE_ANALYSIS_PROFILE_ID = YOLOV8N_PROFILE.profile_id


def pose_analysis_profile(profile_id: str | None = None) -> PoseAnalysisProfile:
    """Return the fixed YOLOv8n configuration."""

    resolved_id = profile_id or DEFAULT_POSE_ANALYSIS_PROFILE_ID
    if resolved_id != DEFAULT_POSE_ANALYSIS_PROFILE_ID:
        raise ValueError("Only the YOLOv8n pose workflow is available.")
    return YOLOV8N_PROFILE


def pose_analysis_profiles_payload() -> list[dict]:
    """Return the fixed profile for compatibility with internal tooling."""

    return [YOLOV8N_PROFILE.to_dict()]
