"""Human Movement Window diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acl_motion.annotations.models import MovementWindowAnnotation, RoiKeyframeAnnotation


def plot_movement_window_timeline(
    keyframes: tuple[RoiKeyframeAnnotation, ...],
    movement_window: MovementWindowAnnotation,
    output_path: str | Path,
) -> Path:
    """Plot Movement Start, Movement End, and manual ROI keyframes."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 2.8))
    start = movement_window.movement_start_frame
    end = movement_window.movement_end_frame
    axis.axvspan(start, end, color="#0969da", alpha=0.18, label="Analysis window")
    axis.axvline(start, color="#2da44e", linewidth=2, label="Movement Start")
    axis.axvline(end, color="#cf222e", linewidth=2, label="Movement End")
    frames = [keyframe.frame_index for keyframe in keyframes]
    axis.scatter(frames, [1] * len(frames), color="#24292f", s=34, label="manual ROI keyframes")
    axis.set_yticks([])
    axis.set_xlabel("source_frame_index")
    axis.set_title("Human Movement Window")
    axis.text(start, 1.06, "Movement Start", ha="left", va="bottom", fontsize=9)
    axis.text(end, 1.06, "Movement End", ha="right", va="bottom", fontsize=9)
    axis.legend(loc="lower center", ncols=4, fontsize=8)
    axis.set_ylim(0.8, 1.18)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_body_region_evidence_availability(
    dynamic_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot supported feature fraction through the human movement window by body region."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    region_by_feature = _region_by_feature(dynamic_df)
    frames = sorted(dynamic_df["source_frame_index"].unique())
    rows = []
    for frame in frames:
        frame_rows = dynamic_df[dynamic_df["source_frame_index"].eq(frame)]
        for region in ("lower_limb", "trunk_pelvis", "upper_body", "bilateral"):
            names = [name for name, mapped in region_by_feature.items() if mapped == region]
            region_rows = frame_rows[frame_rows["feature_name"].isin(names)]
            supported = region_rows["feature_status"].eq("SUPPORTED").mean() if len(region_rows) else 0.0
            rows.append({"frame": frame, "region": region, "supported_fraction": supported})
    table = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(11, 4.8))
    for region, rows_for_region in table.groupby("region", sort=False):
        axis.plot(
            rows_for_region["frame"],
            rows_for_region["supported_fraction"],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=region,
        )
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("source_frame_index")
    axis.set_ylabel("supported feature fraction")
    axis.set_title("Pose / Evidence Availability Through Human Movement Window")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _region_by_feature(dynamic_df: pd.DataFrame) -> dict[str, str]:
    from acl_motion.profiles.registry import classify_feature

    return {
        feature_name: classify_feature(str(feature_name))[0].value
        for feature_name in dynamic_df["feature_name"].unique()
    }


def _prepare_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
