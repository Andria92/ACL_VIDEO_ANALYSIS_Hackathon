"""Milestone 4.1 dynamic reliability diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HKA_DYNAMIC_FEATURES = (
    ("left_hka_angle_2d_deg", "left HKA"),
    ("right_hka_angle_2d_deg", "right HKA"),
    ("hka_projected_bilateral_difference_deg", "bilateral HKA difference"),
)


def plot_hka_raw_vs_robust_velocity(
    dynamic_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot raw first-difference rates against robust dynamic rates."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)
    for axis, (feature_name, label) in zip(axes, HKA_DYNAMIC_FEATURES, strict=True):
        rows = dynamic_df[dynamic_df["feature_name"].eq(feature_name)].sort_values(
            "event_relative_ms"
        )
        _plot_rate(axis, rows, "raw_first_difference_rate", "raw first difference", "#0969da")
        robust_values = rows["robust_dynamic_rate"].where(rows["dynamic_status"].eq("SUPPORTED"))
        axis.plot(
            rows["event_relative_ms"],
            robust_values,
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label="robust supported rate",
            color="#2da44e",
        )
        outliers = rows[
            rows["dynamic_status"].isin(["TEMPORAL_OUTLIER", "LOW_DYNAMIC_CONFIDENCE"])
            & rows["raw_first_difference_rate"].notna()
        ]
        if not outliers.empty:
            axis.scatter(
                outliers["event_relative_ms"],
                outliers["raw_first_difference_rate"],
                s=38,
                marker="x",
                color="#cf222e",
                label="limited/outlier raw point",
                zorder=4,
            )
        axis.axvline(0, color="#cf222e", linewidth=1.2, linestyle="--", alpha=0.85)
        axis.axhline(0, color="#6e7781", linewidth=1, alpha=0.65)
        axis.set_title(label)
        axis.set_ylabel("rate")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("event_relative_ms")
    fig.suptitle("Raw vs Robust Projected HKA Dynamic Rates")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_dynamic_spike_audit(
    dynamic_df: pd.DataFrame,
    spike_audit: pd.DataFrame,
    output_path: str | Path,
    *,
    max_spikes: int = 6,
) -> Path:
    """Plot local trajectories around the largest raw-rate events."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    if spike_audit.empty:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, "No raw dynamic spikes available", ha="center", va="center")
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return output

    top = spike_audit.assign(abs_raw=spike_audit["raw_rate"].abs()).nlargest(max_spikes, "abs_raw")
    fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, max(3, 2.4 * len(top))))
    if len(top) == 1:
        axes = [axes]
    for axis, (_, spike) in zip(axes, top.iterrows(), strict=True):
        feature_name = spike["feature_name"]
        current_frame = int(spike["source_frame_current"])
        rows = dynamic_df[
            dynamic_df["feature_name"].eq(feature_name)
            & dynamic_df["source_frame_index"].between(current_frame - 5, current_frame + 5)
        ].sort_values("source_frame_index")
        supported = rows["feature_status"].eq("SUPPORTED")
        axis.plot(
            rows["source_frame_index"],
            rows["feature_value"].where(supported),
            marker="o",
            linewidth=1.4,
            color="#0969da",
        )
        axis.axvline(current_frame, color="#cf222e", linestyle="--", linewidth=1.2)
        axis.set_title(
            f"{feature_name} | raw {spike['raw_rate']:.1f} | "
            f"robust {spike['robust_rate']:.1f} | {spike['dynamic_status']}"
        )
        axis.set_ylabel(str(rows.iloc[0]["unit"]) if not rows.empty else "")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("source_frame_index")
    fig.suptitle("Dynamic Spike Audit: Local Feature Trajectories")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_rate(axis, rows: pd.DataFrame, column: str, label: str, color: str) -> None:
    values = rows[column].where(rows[column].notna())
    axis.plot(
        rows["event_relative_ms"],
        values,
        marker="o",
        markersize=2.5,
        linewidth=1.2,
        label=label,
        color=color,
        alpha=0.75,
    )


def _prepare_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
