"""Target-athlete segmentation helpers."""

from acl_motion.segmentation.target_mask import (
    MaskPrompt,
    append_mask_prompt,
    draw_target_mask_overlay,
    load_mask_prompts,
    target_mask_for_frame,
)

__all__ = [
    "MaskPrompt",
    "append_mask_prompt",
    "draw_target_mask_overlay",
    "load_mask_prompts",
    "target_mask_for_frame",
]
