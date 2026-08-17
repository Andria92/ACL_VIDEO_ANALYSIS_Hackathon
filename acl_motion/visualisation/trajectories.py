"""Raw coordinate diagnostic plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

DEFAULT_DIAGNOSTIC_LANDMARKS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def plot_joint_coordinate_diagnostics(
    pose_df,
    output_path: str | Path,
    *,
    landmark_names: Sequence[str] = DEFAULT_DIAGNOSTIC_LANDMARKS,
) -> Path:
    """Plot raw x/y pixel trajectories for selected landmarks."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    subset = pose_df[
        pose_df["landmark_name"].isin(landmark_names) & pose_df["observed"].astype(bool)
    ].copy()
    if subset.empty:
        raise ValueError("No observed landmark rows available for diagnostic plotting.")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    for landmark_name in landmark_names:
        landmark_rows = subset[subset["landmark_name"] == landmark_name]
        if landmark_rows.empty:
            continue
        axes[0].plot(
            landmark_rows["timestamp_ms"],
            landmark_rows["x_px"],
            linewidth=1.2,
            label=landmark_name,
        )
        axes[1].plot(
            landmark_rows["timestamp_ms"],
            landmark_rows["y_px"],
            linewidth=1.2,
            label=landmark_name,
        )

    axes[0].set_ylabel("x_px")
    axes[1].set_ylabel("y_px")
    axes[1].set_xlabel("timestamp_ms")
    axes[0].set_title("Raw landmark coordinate diagnostics")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
