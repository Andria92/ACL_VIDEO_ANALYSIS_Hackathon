"""Milestone 2 quality visualisations."""

from __future__ import annotations

from pathlib import Path

DIAGNOSTIC_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def plot_raw_clean_smoothed(processed_pose, output_path: str | Path) -> Path:
    """Plot raw, clean, and smoothed x/y trajectories for selected landmarks."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13, 8), sharex=True)
    for landmark in DIAGNOSTIC_LANDMARKS:
        rows = processed_pose[processed_pose["landmark_name"].eq(landmark)].sort_values("timestamp_ms")
        if rows.empty:
            continue
        axes[0].plot(rows["timestamp_ms"], rows["raw_x"], alpha=0.22, linewidth=1, linestyle=":", label=f"{landmark} raw")
        axes[0].plot(rows["timestamp_ms"], rows["clean_x"], alpha=0.8, linewidth=1.2, label=f"{landmark} clean")
        axes[0].plot(rows["timestamp_ms"], rows["smoothed_x"], alpha=0.95, linewidth=1.6, linestyle="--", label=f"{landmark} smooth")
        axes[1].plot(rows["timestamp_ms"], rows["raw_y"], alpha=0.22, linewidth=1, linestyle=":")
        axes[1].plot(rows["timestamp_ms"], rows["clean_y"], alpha=0.8, linewidth=1.2)
        axes[1].plot(rows["timestamp_ms"], rows["smoothed_y"], alpha=0.95, linewidth=1.6, linestyle="--")
    axes[0].set_ylabel("x_px")
    axes[1].set_ylabel("y_px")
    axes[1].set_xlabel("timestamp_ms")
    axes[0].set_title("Raw vs clean vs smoothed coordinate diagnostics")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="x-small", ncol=1)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_availability_timeline(processed_pose, frame_quality, output_path: str | Path) -> Path:
    """Plot target and landmark availability/status over frames."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_order = sorted(frame_quality["frame_index"].unique())
    labels = ["target", *DIAGNOSTIC_LANDMARKS]
    matrix = []
    target_codes = [
        _target_code(frame_quality.loc[frame_quality["frame_index"].eq(frame), "frame_status"].iloc[0])
        for frame in frame_order
    ]
    matrix.append(target_codes)
    for landmark in DIAGNOSTIC_LANDMARKS:
        rows = processed_pose[processed_pose["landmark_name"].eq(landmark)].set_index("frame_index")
        matrix.append([_landmark_code(rows.loc[frame, "landmark_status"]) for frame in frame_order])
    data = np.array(matrix)
    cmap = ListedColormap(["#d0d7de", "#2da44e", "#d29922", "#cf222e", "#8250df"])
    fig, axis = plt.subplots(figsize=(13, 5.4))
    axis.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=4)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("frame_index")
    axis.set_title("Confidence / availability timeline")
    tick_positions = np.linspace(0, len(frame_order) - 1, min(10, len(frame_order)), dtype=int)
    axis.set_xticks(tick_positions, [frame_order[i] for i in tick_positions])
    legend_text = "gray missing | green valid | amber low/partial | red rejected/lost | purple identity uncertain"
    fig.text(0.5, 0.035, legend_text, fontsize=9, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _target_code(status: str) -> int:
    if status == "VALID_TARGET":
        return 1
    if status in {"LOW_POSE_CONFIDENCE", "PARTIAL_POSE"}:
        return 2
    if status == "TARGET_IDENTITY_UNCERTAIN":
        return 4
    if status == "TARGET_NOT_FOUND":
        return 3
    return 0


def _landmark_code(status: str) -> int:
    if status == "OBSERVED_VALID":
        return 1
    if status == "LOW_CONFIDENCE":
        return 2
    if status == "IDENTITY_UNCERTAIN":
        return 4
    if status in {"TEMPORAL_OUTLIER", "REJECTED"}:
        return 3
    return 0
