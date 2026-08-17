"""Human-correctable target-athlete mask generation.

The manual ROI is treated as a target-identity seed, not proof that every pixel
inside the rectangle belongs to the target athlete. This lightweight layer uses
OpenCV GrabCut plus optional human positive/negative point prompts so the UI can
separate visible target pixels from nearby opponents without requiring framewise
manual tracing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from acl_motion.video.roi import BBox

MASK_VERSION = "m5_9_target_mask_grabcut_prompt_v1"


@dataclass(frozen=True, slots=True)
class MaskPrompt:
    """One human point/brush prompt for visible target-region refinement."""

    frame_index: int
    x_px: float
    y_px: float
    label: str
    provenance: str = "human_ui"
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in {"target", "opponent"}:
            raise ValueError("Mask prompt label must be 'target' or 'opponent'.")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready prompt data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskPrompt:
        """Load one prompt from JSON."""

        return cls(
            frame_index=int(data["frame_index"]),
            x_px=float(data["x_px"]),
            y_px=float(data["y_px"]),
            label=str(data["label"]),
            provenance=str(data.get("provenance", "human_ui")),
            created_at=str(data.get("created_at", "")),
        )


def mask_prompt_path(data_root: str | Path, slug: str) -> Path:
    """Return the human target-mask prompt path for one case."""

    return Path(data_root) / "segmentation" / "human" / f"{slug}_target_mask_prompts.json"


def load_mask_prompts(path: str | Path) -> tuple[MaskPrompt, ...]:
    """Load saved human positive/negative target-mask prompts."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        return ()
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    return tuple(MaskPrompt.from_dict(item) for item in payload.get("prompts", []))


def append_mask_prompt(path: str | Path, prompt: MaskPrompt) -> dict:
    """Append one human target-mask prompt and return the saved payload."""

    prompt_path = Path(path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    existing = list(load_mask_prompts(prompt_path))
    existing.append(prompt)
    payload = {
        "mask_version": MASK_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "prompts": [item.to_dict() for item in existing],
    }
    prompt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def pop_mask_prompt(path: str | Path, frame_index: int | None = None) -> dict:
    """Remove the most recent prompt, optionally restricted to one frame."""

    prompt_path = Path(path)
    existing = list(load_mask_prompts(prompt_path))
    for index in range(len(existing) - 1, -1, -1):
        if frame_index is None or existing[index].frame_index == int(frame_index):
            existing.pop(index)
            break
    return _write_prompt_payload(prompt_path, existing)


def clear_mask_prompts(path: str | Path, frame_index: int | None = None) -> dict:
    """Clear saved prompts, optionally only for one source frame."""

    prompt_path = Path(path)
    if frame_index is None:
        remaining: list[MaskPrompt] = []
    else:
        remaining = [
            prompt
            for prompt in load_mask_prompts(prompt_path)
            if prompt.frame_index != int(frame_index)
        ]
    return _write_prompt_payload(prompt_path, remaining)


def target_mask_for_frame(
    frame: Any,
    *,
    bbox: BBox,
    prompts: tuple[MaskPrompt, ...] = (),
    frame_index: int,
) -> np.ndarray:
    """Return a binary visible-target mask for one frame.

    The output is intentionally a visible-pixel mask. Fragmented components are
    allowed; occluded anatomy should remain absent unless a pose/temporal layer
    later marks it as inferred.
    """

    import cv2

    height, width = frame.shape[:2]
    rect = _safe_rect(bbox, width, height)
    if rect[2] <= 1 or rect[3] <= 1:
        return np.zeros((height, width), dtype=np.uint8)
    grabcut_mask = np.zeros((height, width), dtype=np.uint8)
    grabcut_mask[:, :] = cv2.GC_BGD
    x, y, w, h = rect
    grabcut_mask[y : y + h, x : x + w] = cv2.GC_PR_FGD
    frame_prompts = [prompt for prompt in prompts if prompt.frame_index == frame_index]
    for prompt in frame_prompts:
        px = round(prompt.x_px)
        py = round(prompt.y_px)
        if not 0 <= px < width or not 0 <= py < height:
            continue
        value = cv2.GC_FGD if prompt.label == "target" else cv2.GC_BGD
        cv2.circle(grabcut_mask, (px, py), 7, value, thickness=-1)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)
    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = _clean_mask(foreground)
    return foreground


def draw_target_mask_overlay(frame: Any, mask: np.ndarray, *, alpha: float = 0.34) -> Any:
    """Draw a semi-transparent target mask over a BGR video frame."""

    import cv2

    if mask is None or mask.size == 0 or not np.any(mask):
        return frame
    overlay = frame.copy()
    color = np.zeros_like(frame)
    color[:, :] = (185, 140, 35)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(frame[mask_bool], 1.0 - alpha, color[mask_bool], alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 210, 70), 2, lineType=cv2.LINE_AA)
    return overlay


def draw_mask_prompt_overlay(
    frame: Any,
    prompts: tuple[MaskPrompt, ...],
    *,
    frame_index: int,
) -> Any:
    """Draw saved human target/non-target prompt samples as separate evidence."""

    import cv2

    output = frame.copy()
    for prompt in prompts:
        if prompt.frame_index != int(frame_index):
            continue
        color = (65, 180, 95) if prompt.label == "target" else (45, 45, 220)
        center = (round(prompt.x_px), round(prompt.y_px))
        cv2.circle(output, center, 7, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(output, center, 9, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return output


def _write_prompt_payload(path: Path, prompts: list[MaskPrompt]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mask_version": MASK_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "prompts": [item.to_dict() for item in prompts],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _safe_rect(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    x = max(0, min(width - 1, round(bbox.x)))
    y = max(0, min(height - 1, round(bbox.y)))
    right = max(x + 1, min(width, round(bbox.x + bbox.width)))
    bottom = max(y + 1, min(height, round(bbox.y + bbox.height)))
    return x, y, right - x, bottom - y


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned
