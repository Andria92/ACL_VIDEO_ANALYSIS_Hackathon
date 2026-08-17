from __future__ import annotations

import numpy as np

from acl_motion.segmentation.target_mask import (
    MaskPrompt,
    append_mask_prompt,
    clear_mask_prompts,
    draw_mask_prompt_overlay,
    draw_target_mask_overlay,
    load_mask_prompts,
    pop_mask_prompt,
    target_mask_for_frame,
)
from acl_motion.video.roi import BBox


def test_mask_prompts_preserve_human_target_and_opponent_labels(tmp_path) -> None:
    path = tmp_path / "mask_prompts.json"

    payload = append_mask_prompt(
        path,
        MaskPrompt(frame_index=12, x_px=10.0, y_px=20.0, label="target"),
    )
    append_mask_prompt(
        path,
        MaskPrompt(frame_index=12, x_px=30.0, y_px=40.0, label="opponent"),
    )

    prompts = load_mask_prompts(path)

    assert payload["mask_version"] == "m5_9_target_mask_grabcut_prompt_v1"
    assert [prompt.label for prompt in prompts] == ["target", "opponent"]
    assert prompts[0].provenance == "human_ui"


def test_target_mask_for_frame_returns_visible_pixel_mask() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:65, 30:70] = (60, 180, 80)
    bbox = BBox(x=22, y=12, width=58, height=60)

    mask = target_mask_for_frame(
        frame,
        bbox=bbox,
        prompts=(MaskPrompt(frame_index=0, x_px=50, y_px=40, label="target"),),
        frame_index=0,
    )

    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert mask[40, 50] > 0


def test_target_mask_overlay_changes_masked_pixels() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    overlay = draw_target_mask_overlay(frame, mask)

    assert overlay.shape == frame.shape
    assert overlay[15, 15].sum() > 0
    assert overlay[0, 0].sum() == 0


def test_mask_prompt_overlay_draws_human_region_evidence() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    overlay = draw_mask_prompt_overlay(
        frame,
        (
            MaskPrompt(frame_index=4, x_px=12, y_px=14, label="target"),
            MaskPrompt(frame_index=5, x_px=28, y_px=30, label="opponent"),
        ),
        frame_index=4,
    )

    assert overlay[14, 12].sum() > 0
    assert overlay[30, 28].sum() == 0


def test_mask_prompt_undo_and_clear_can_be_frame_scoped(tmp_path) -> None:
    path = tmp_path / "mask_prompts.json"
    append_mask_prompt(path, MaskPrompt(frame_index=12, x_px=10.0, y_px=20.0, label="target"))
    append_mask_prompt(path, MaskPrompt(frame_index=12, x_px=11.0, y_px=21.0, label="opponent"))
    append_mask_prompt(path, MaskPrompt(frame_index=13, x_px=30.0, y_px=40.0, label="target"))

    undo_payload = pop_mask_prompt(path, frame_index=12)

    assert len(undo_payload["prompts"]) == 2
    assert [(prompt.frame_index, prompt.label) for prompt in load_mask_prompts(path)] == [
        (12, "target"),
        (13, "target"),
    ]

    clear_payload = clear_mask_prompts(path, frame_index=12)

    assert len(clear_payload["prompts"]) == 1
    assert [(prompt.frame_index, prompt.label) for prompt in load_mask_prompts(path)] == [
        (13, "target"),
    ]
