"""Milestone 4 event-relative diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ORIENTATION_FEATURES = {
    "projected_trunk_axis_angle_deg",
    "projected_hip_line_angle_deg",
    "projected_shoulder_line_angle_deg",
    "projected_shoulder_pelvis_orientation_difference_deg",
    "left_upper_arm_orientation_2d_deg",
    "right_upper_arm_orientation_2d_deg",
}


def plot_event_relative_hka(
    event_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot event-relative projected HKA trajectories."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4.8))
    _plot_feature(axis, event_df, "left_hka_angle_2d_deg", "left HKA 2D")
    _plot_feature(axis, event_df, "right_hka_angle_2d_deg", "right HKA 2D")
    _add_t0(axis)
    axis.set_title("Event-Relative Projected HKA 2D")
    axis.set_xlabel("event_relative_ms")
    axis.set_ylabel("angle (deg)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_hka_angular_velocity(event_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot HKA angular velocity inside supported feature segments only."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4.8))
    _plot_dynamic(axis, event_df, "left_hka_angle_2d_deg", "left HKA angular velocity")
    _plot_dynamic(axis, event_df, "right_hka_angle_2d_deg", "right HKA angular velocity")
    _add_t0(axis)
    axis.axhline(0, color="#6e7781", linewidth=1, alpha=0.7)
    axis.set_title("Projected HKA Angular Velocity")
    axis.set_xlabel("event_relative_ms")
    axis.set_ylabel("deg/s")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_dynamic_bilateral_hka(event_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot projected injured-minus-contralateral HKA difference and rate."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True)
    _plot_feature(
        axes[0],
        event_df,
        "hka_projected_bilateral_difference_deg",
        "injured - contralateral HKA",
    )
    _plot_dynamic(
        axes[1],
        event_df,
        "hka_projected_bilateral_difference_deg",
        "rate of projected HKA difference",
    )
    for axis in axes:
        _add_t0(axis)
        axis.axhline(0, color="#6e7781", linewidth=1, alpha=0.7)
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("difference (deg)")
    axes[1].set_ylabel("deg/s")
    axes[1].set_xlabel("event_relative_ms")
    fig.suptitle("Dynamic Projected Bilateral HKA Difference")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_whole_body_event_profile(event_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot representative event-relative whole-body geometry small multiples."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    features = (
        ("left_hka_angle_2d_deg", "left HKA"),
        ("right_hka_angle_2d_deg", "right HKA"),
        ("projected_trunk_axis_angle_deg", "trunk axis"),
        ("projected_hip_line_angle_deg", "hip line"),
        ("left_elbow_angle_2d_deg", "left elbow"),
        ("right_elbow_angle_2d_deg", "right elbow"),
        ("hka_projected_bilateral_difference_deg", "bilateral HKA difference"),
        ("projected_shoulder_pelvis_x_offset_normalized", "shoulder-pelvis x offset"),
    )
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 12), sharex=True)
    for axis, (feature_name, label) in zip(axes.flat, features, strict=True):
        _plot_feature(axis, event_df, feature_name, label)
        _add_t0(axis)
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
    axes[-1][0].set_xlabel("event_relative_ms")
    axes[-1][1].set_xlabel("event_relative_ms")
    fig.suptitle("Whole-Body Event Profile")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_event_window_summary(
    event_df: pd.DataFrame,
    output_path: str | Path,
    *,
    windows: list[dict],
) -> Path:
    """Show event windows against selected trajectories."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4.8))
    for window in windows:
        axis.axvspan(
            window["start_ms"],
            window["end_ms"],
            color="#d0d7de",
            alpha=0.25,
            linewidth=0,
        )
        axis.text(
            (window["start_ms"] + window["end_ms"]) / 2,
            0.98,
            window["name"],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    _plot_feature(axis, event_df, "left_hka_angle_2d_deg", "left HKA 2D")
    _plot_feature(axis, event_df, "right_hka_angle_2d_deg", "right HKA 2D")
    _add_t0(axis)
    axis.set_title("Event Window Summary Context")
    axis.set_xlabel("event_relative_ms")
    axis.set_ylabel("angle (deg)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_event_extrema_timing(
    event_df: pd.DataFrame,
    summary: dict,
    output_path: str | Path,
) -> Path:
    """Annotate selected extrema and timing without overcrowding the graph."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    feature_name = "hka_projected_bilateral_difference_deg"
    rows = event_df[event_df["feature_name"].eq(feature_name)]
    if rows.empty:
        feature_name = "left_hka_angle_2d_deg"
        rows = event_df[event_df["feature_name"].eq(feature_name)]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    _plot_feature(axis, event_df, feature_name, feature_name)
    _add_t0(axis)
    feature_summary = summary.get("feature_summaries", {}).get(feature_name, {})
    _annotate_time_value(axis, feature_summary.get("time_of_minimum_ms"), feature_summary.get("minimum"), "min")
    _annotate_time_value(axis, feature_summary.get("time_of_maximum_ms"), feature_summary.get("maximum"), "max")
    _annotate_time_value(
        axis,
        feature_summary.get("time_of_maximum_absolute_rate_of_change_ms"),
        _value_at_time(rows, feature_summary.get("time_of_maximum_absolute_rate_of_change_ms")),
        "max |rate|",
    )
    axis.axhline(0, color="#6e7781", linewidth=1, alpha=0.7)
    axis.set_title("Selected Extrema and Timing")
    axis.set_xlabel("event_relative_ms")
    axis.set_ylabel(str(rows.iloc[0]["unit"]) if not rows.empty else "")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_feature(axis, event_df: pd.DataFrame, feature_name: str, label: str) -> None:
    rows = event_df[event_df["feature_name"].eq(feature_name)].sort_values("event_relative_ms")
    if rows.empty:
        axis.text(0.5, 0.5, f"{feature_name} unavailable", transform=axis.transAxes, ha="center")
        return
    supported = rows["feature_status"].eq("SUPPORTED")
    values = rows["feature_value"].where(supported)
    if feature_name in ORIENTATION_FEATURES:
        values = values.mask(values.diff().abs() > 120)
    axis.plot(rows["event_relative_ms"], values, marker="o", markersize=2.5, linewidth=1.4, label=label)
    interpolated = supported & rows["input_interpolated"].astype(bool)
    if interpolated.any():
        axis.scatter(
            rows.loc[interpolated, "event_relative_ms"],
            rows.loc[interpolated, "feature_value"],
            s=28,
            facecolors="none",
            edgecolors="#d29922",
            linewidths=1.2,
            zorder=3,
        )


def _plot_dynamic(axis, event_df: pd.DataFrame, feature_name: str, label: str) -> None:
    rows = event_df[event_df["feature_name"].eq(feature_name)].sort_values("event_relative_ms")
    if rows.empty:
        axis.text(0.5, 0.5, f"{feature_name} unavailable", transform=axis.transAxes, ha="center")
        return
    values = rows["dynamic_value"].where(rows["dynamic_status"].eq("SUPPORTED"))
    axis.plot(rows["event_relative_ms"], values, marker="o", markersize=2.5, linewidth=1.4, label=label)


def _add_t0(axis) -> None:
    axis.axvline(0, color="#cf222e", linewidth=1.5, linestyle="--", alpha=0.9, label="t0")


def _annotate_time_value(axis, time_ms, value, label: str) -> None:
    if time_ms is None or value is None:
        return
    try:
        time_value = float(time_ms)
        feature_value = float(value)
    except (TypeError, ValueError):
        return
    if pd.isna(time_value) or pd.isna(feature_value):
        return
    axis.scatter([time_value], [feature_value], s=48, zorder=4)
    axis.annotate(
        f"{label}\n{time_value:.0f} ms",
        (time_value, feature_value),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )


def _value_at_time(rows: pd.DataFrame, time_ms) -> float:
    if time_ms is None or rows.empty:
        return float("nan")
    try:
        time_value = float(time_ms)
    except (TypeError, ValueError):
        return float("nan")
    supported = rows[rows["feature_status"].eq("SUPPORTED")]
    if supported.empty:
        return float("nan")
    index = (supported["event_relative_ms"] - time_value).abs().idxmin()
    return float(supported.loc[index, "feature_value"])


def _prepare_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
