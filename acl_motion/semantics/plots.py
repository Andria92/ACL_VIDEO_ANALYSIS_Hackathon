"""Diagnostic plots for semantic movement observations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acl_motion.semantics.models import MovementObservation


def plot_projected_movement_path(path_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot camera-compensated projected body-center path."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    supported = path_df[path_df["path_status"].eq("SUPPORTED")]
    fig, axis = plt.subplots(figsize=(6.6, 5.2))
    if supported.empty:
        axis.text(0.5, 0.5, "Projected path unavailable", ha="center", va="center")
    else:
        groups = (
            supported.groupby("path_segment_id", dropna=False)
            if "path_segment_id" in supported.columns
            else (("path", supported),)
        )
        for _, segment in groups:
            axis.plot(segment["compensated_x"], segment["compensated_y"], marker="o", markersize=3)
        axis.scatter(
            supported["compensated_x"].iloc[0],
            supported["compensated_y"].iloc[0],
            label="start",
            color="#176d4d",
            s=60,
        )
        axis.scatter(
            supported["compensated_x"].iloc[-1],
            supported["compensated_y"].iloc[-1],
            label="end",
            color="#9d2735",
            s=60,
        )
        axis.legend()
    axis.set_title("Camera-Compensated Projected Movement Path")
    axis.set_xlabel("projected x, body-center units not pitch-calibrated")
    axis.set_ylabel("projected y, body-center units not pitch-calibrated")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_path_validation_comparison(
    center_df: pd.DataFrame,
    translation_path: pd.DataFrame,
    affine_path: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot raw image-space and candidate compensated paths for QA comparison."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    _plot_xy_segments(
        axes[0],
        center_df.rename(columns={"center_x": "x", "center_y": "y"}),
        x_col="x",
        y_col="y",
        status_col="center_status",
        title="Raw image-space pelvis path",
        unavailable_text="Raw pelvis path unavailable",
    )
    _plot_xy_segments(
        axes[1],
        translation_path,
        x_col="compensated_x",
        y_col="compensated_y",
        status_col="path_status",
        title="Translation-compensated candidate",
        unavailable_text="Translation path unavailable",
    )
    _plot_xy_segments(
        axes[2],
        affine_path,
        x_col="compensated_x",
        y_col="compensated_y",
        status_col="path_status",
        title="Affine-compensated candidate",
        unavailable_text="Affine path unavailable",
    )
    for axis in axes:
        axis.set_xlabel("projected x (px / re-anchored px)")
        axis.set_ylabel("projected y (px / re-anchored px)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_projected_speed(path_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot body-scale-normalized projected speed."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3.6))
    supported = path_df[path_df["path_status"].eq("SUPPORTED")]
    for _, segment in _supported_path_groups(supported):
        axis.plot(
            segment["movement_end_relative_ms"],
            segment["normalized_projected_speed_per_s"],
            color="#215f9a",
        )
    axis.axvline(0, color="#9d2735", linewidth=1)
    axis.set_title("Projected Speed Pattern")
    axis.set_xlabel("ms before Movement End")
    axis.set_ylabel("body-scale units/s")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_projected_heading(path_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot projected heading through time."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3.6))
    supported = path_df[path_df["path_status"].eq("SUPPORTED")]
    for _, segment in _supported_path_groups(supported):
        axis.plot(
            segment["movement_end_relative_ms"],
            segment["projected_heading_deg"],
            color="#176d4d",
        )
    axis.axvline(0, color="#9d2735", linewidth=1)
    axis.set_title("Projected Direction / Heading")
    axis.set_xlabel("ms before Movement End")
    axis.set_ylabel("projected heading deg")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_bilateral_hka_relationship(dynamic_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot injured/contralateral HKA plus signed and absolute relationship."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    features = {
        "injured_hka_angle_2d_deg": "injured HKA",
        "contralateral_hka_angle_2d_deg": "contralateral HKA",
        "hka_projected_bilateral_difference_deg": "signed difference",
        "hka_projected_bilateral_absolute_difference_deg": "absolute difference",
    }
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for feature, label in list(features.items())[:2]:
        rows = _supported(dynamic_df, feature)
        axes[0].plot(rows["movement_end_relative_ms"], rows["feature_value"], label=label)
    for feature, label in list(features.items())[2:]:
        rows = _supported(dynamic_df, feature)
        axes[1].plot(rows["movement_end_relative_ms"], rows["feature_value"], label=label)
    axes[0].set_title("Projected HKA Relationship")
    axes[0].set_ylabel("deg")
    axes[1].set_title("Bilateral Relationship Through Time")
    axes[1].set_xlabel("ms before Movement End")
    axes[1].set_ylabel("deg")
    for axis in axes:
        axis.axvline(0, color="#9d2735", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_semantic_summary(
    observations: list[MovementObservation],
    output_path: str | Path,
) -> Path:
    """Plot semantic observation evidence states by category."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    categories = []
    statuses = []
    for observation in observations:
        categories.append(observation.category.replace("_", "\n"))
        statuses.append(observation.evidence_status.value)
    color_map = {"SUPPORTED": "#176d4d", "LIMITED": "#9a6400", "UNAVAILABLE": "#9d2735"}
    fig, axis = plt.subplots(figsize=(9, max(4, len(observations) * 0.25)))
    axis.barh(
        range(len(observations)),
        [1] * len(observations),
        color=[color_map[status] for status in statuses],
    )
    axis.set_yticks(range(len(observations)), categories, fontsize=8)
    axis.set_xticks([])
    axis.set_title("Semantic Movement Summary Evidence")
    for i, observation in enumerate(observations):
        axis.text(0.02, i, observation.title, va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_phase_change_score(
    change_df: pd.DataFrame,
    phases: list[dict],
    transitions: list[dict],
    output_path: str | Path,
) -> Path:
    """Plot the multivariate movement-change score and selected phase boundaries."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4.4))
    axis.plot(
        change_df["movement_end_relative_ms"],
        change_df["change_score"],
        color="#9aa7b5",
        linewidth=1.0,
        label="raw change score",
    )
    if "smoothed_change_score" in change_df.columns:
        axis.plot(
            change_df["movement_end_relative_ms"],
            change_df["smoothed_change_score"],
            color="#215f9a",
            linewidth=2.0,
            label="smoothed change score",
        )
    if "boundary_threshold" in change_df.columns:
        threshold = change_df["boundary_threshold"].dropna()
        if not threshold.empty:
            axis.axhline(float(threshold.iloc[0]), color="#9a6400", linestyle="--", label="boundary threshold")
    timing = change_df.set_index("source_frame_index")
    for transition in transitions:
        frame = int(transition["transition_frame"])
        if frame not in timing.index:
            continue
        time_ms = float(timing.loc[frame, "movement_end_relative_ms"])
        axis.axvline(time_ms, color="#9d2735", linewidth=1.6)
        axis.text(
            time_ms,
            axis.get_ylim()[1] * 0.95,
            f"Phase {transition.get('to_phase_id', '').split('_')[-1]}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#9d2735",
        )
    for phase in phases:
        axis.axvspan(
            phase["start_relative_ms"],
            phase["end_relative_ms"],
            color="#215f9a",
            alpha=0.04 if int(phase["phase_index"]) % 2 else 0.08,
        )
    axis.set_title("Multivariate Movement-Change Score")
    axis.set_xlabel("ms before Movement End")
    axis.set_ylabel("mean absolute standardized change")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_phase_timeline(phases: list[dict], output_path: str | Path) -> Path:
    """Plot a user-readable Movement Phase timeline."""

    output = _prepare(output_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, max(3.2, len(phases) * 0.72)))
    colors = {
        "GOOD": "#176d4d",
        "MODERATE": "#215f9a",
        "LIMITED": "#9a6400",
        "UNAVAILABLE": "#9d2735",
    }
    y_ticks = []
    y_labels = []
    for y, phase in enumerate(reversed(phases)):
        start = float(phase["start_relative_ms"])
        end = float(phase["end_relative_ms"])
        duration = max(end - start, 1.0)
        status = phase["evidence_summary"]["evidence_status"]
        axis.barh(
            y,
            duration,
            left=start,
            height=0.5,
            color=colors.get(status, "#52616f"),
            alpha=0.88,
        )
        axis.text(
            start + duration / 2,
            y,
            f"{phase['duration_ms'] / 1000:.2f}s",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        y_ticks.append(y)
        y_labels.append(
            f"Phase {phase['phase_index']} - {phase['title']}\n{status}"
        )
    axis.set_yticks(y_ticks, y_labels)
    axis.set_xlabel("ms before Movement End")
    axis.set_title("Movement Phase Timeline")
    axis.axvline(0, color="#1d2630", linewidth=1)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _supported(dynamic_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["feature_status"].eq("SUPPORTED")
        & dynamic_df["feature_value"].notna()
    ]


def _plot_xy_segments(
    axis,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    status_col: str,
    title: str,
    unavailable_text: str,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        axis.text(0.5, 0.5, unavailable_text, ha="center", va="center")
        axis.set_title(title)
        return
    rows = df[df[x_col].notna() & df[y_col].notna()].copy()
    if status_col in rows.columns:
        rows = rows[rows[status_col].eq("SUPPORTED")]
    if rows.empty:
        axis.text(0.5, 0.5, unavailable_text, ha="center", va="center")
        axis.set_title(title)
        return
    if "path_segment_id" in rows.columns:
        groups = rows.groupby("path_segment_id", dropna=False)
    else:
        groups = (("raw", rows),)
    for _, segment in groups:
        axis.plot(segment[x_col], segment[y_col], marker="o", markersize=2.5)
    axis.scatter(rows[x_col].iloc[0], rows[y_col].iloc[0], color="#176d4d", s=42, label="start")
    axis.scatter(rows[x_col].iloc[-1], rows[y_col].iloc[-1], color="#9d2735", s=42, label="end")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)


def _supported_path_groups(supported: pd.DataFrame):
    if "path_segment_id" in supported.columns:
        return supported.groupby("path_segment_id", dropna=False)
    return (("path", supported),)


def _prepare(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
