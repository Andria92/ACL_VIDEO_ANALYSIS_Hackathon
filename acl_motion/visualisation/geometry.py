"""Milestone 3 projected-geometry diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_hka_trajectories(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot left/right projected HKA trajectories with unsupported frames as gaps."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4.8))
    _plot_feature(axis, feature_df, "left_hka_angle_2d_deg", "left HKA 2D")
    _plot_feature(axis, feature_df, "right_hka_angle_2d_deg", "right HKA 2D")
    axis.set_title("Projected HKA 2D Angle Trajectories")
    axis.set_xlabel("timestamp_ms")
    axis.set_ylabel("angle (deg)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_hka_bilateral_difference(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot injured-minus-contralateral projected HKA difference."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4.8))
    _plot_feature(
        axis,
        feature_df,
        "hka_projected_bilateral_difference_deg",
        "injured - contralateral",
    )
    axis.axhline(0, color="#6e7781", linewidth=1, alpha=0.7)
    axis.set_title("Projected Bilateral HKA Difference")
    axis.set_xlabel("timestamp_ms")
    axis.set_ylabel("difference (deg)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_trunk_pelvis_profile(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot trunk/pelvis projected orientation features as small multiples."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    features = (
        ("projected_trunk_axis_angle_deg", "trunk axis"),
        ("projected_hip_line_angle_deg", "hip line"),
        ("projected_shoulder_line_angle_deg", "shoulder line"),
        (
            "projected_shoulder_pelvis_orientation_difference_deg",
            "shoulder - pelvis orientation",
        ),
    )
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
    for axis, (feature_name, label) in zip(axes.flat, features, strict=True):
        _plot_feature(axis, feature_df, feature_name, label, break_jump_deg=120)
        axis.set_title(label)
        axis.set_ylabel("deg")
        axis.grid(True, alpha=0.25)
    axes[1][0].set_xlabel("timestamp_ms")
    axes[1][1].set_xlabel("timestamp_ms")
    fig.suptitle("Projected Trunk / Pelvis Profile")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_upper_limb_profile(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot elbow angles and upper-arm orientations."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    features = (
        ("left_elbow_angle_2d_deg", "left elbow angle"),
        ("right_elbow_angle_2d_deg", "right elbow angle"),
        ("left_upper_arm_orientation_2d_deg", "left upper-arm orientation"),
        ("right_upper_arm_orientation_2d_deg", "right upper-arm orientation"),
    )
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
    for axis, (feature_name, label) in zip(axes.flat, features, strict=True):
        _plot_feature(axis, feature_df, feature_name, label, break_jump_deg=120)
        axis.set_title(label)
        axis.set_ylabel("deg")
        axis.grid(True, alpha=0.25)
    axes[1][0].set_xlabel("timestamp_ms")
    axes[1][1].set_xlabel("timestamp_ms")
    fig.suptitle("Projected Upper-Limb Profile")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_feature_availability(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot feature support status through time."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    features = sorted(feature_df["feature_name"].unique())
    frames = sorted(feature_df["frame_index"].unique())
    code_by_status = {
        "SUPPORTED": 1,
        "LOW_CONFIDENCE": 2,
        "INSUFFICIENT_LANDMARKS": 0,
        "INVALID_TARGET_FRAME": 4,
        "INVALID_GEOMETRY": 3,
        "UNSUPPORTED_VIEW": 3,
    }
    matrix = []
    for feature_name in features:
        rows = feature_df[feature_df["feature_name"].eq(feature_name)].set_index("frame_index")
        matrix.append([code_by_status.get(rows.loc[frame, "status"], 0) for frame in frames])
    data = np.array(matrix)
    cmap = ListedColormap(["#d0d7de", "#2da44e", "#d29922", "#cf222e", "#8250df"])
    height = max(7, min(18, 0.27 * len(features)))
    fig, axis = plt.subplots(figsize=(14, height))
    axis.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=4)
    axis.set_yticks(range(len(features)), labels=features, fontsize=7)
    tick_positions = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    axis.set_xticks(tick_positions, [frames[i] for i in tick_positions])
    axis.set_xlabel("frame_index")
    axis.set_title("Feature Availability")
    legend_text = (
        "gray insufficient/missing | green supported | amber low confidence | "
        "red invalid geometry/view | purple invalid target"
    )
    fig.text(0.5, 0.025, legend_text, fontsize=9, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_feature(
    axis,
    feature_df: pd.DataFrame,
    feature_name: str,
    label: str,
    *,
    break_jump_deg: float | None = None,
) -> None:
    rows = feature_df[feature_df["feature_name"].eq(feature_name)].sort_values("timestamp_ms")
    if rows.empty:
        axis.text(0.5, 0.5, f"{feature_name} unavailable", transform=axis.transAxes, ha="center")
        return
    values = rows["feature_value"].where(rows["status"].eq("SUPPORTED"))
    if break_jump_deg is not None:
        jumps = values.diff().abs()
        values = values.mask(jumps > break_jump_deg)
    axis.plot(rows["timestamp_ms"], values, marker="o", markersize=2.5, linewidth=1.5, label=label)


def _prepare_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
