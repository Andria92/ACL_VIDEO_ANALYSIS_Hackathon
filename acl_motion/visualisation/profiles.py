"""Milestone 5 user-centered profile and evidence plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acl_motion.profiles.models import MovementProfile, QualityCategory


def plot_body_region_coverage(profile: MovementProfile, output_path: str | Path) -> Path:
    """Plot supported/limited/unavailable feature counts by body region."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    rows = []
    for region, features in profile.body_region_profiles.items():
        counts = {"SUPPORTED": 0, "LIMITED": 0, "UNAVAILABLE": 0}
        for feature in features:
            counts[feature.quality_category] = counts.get(feature.quality_category, 0) + 1
        rows.append({"body_region": region, **counts})
    df = pd.DataFrame(rows).set_index("body_region")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bottom = None
    colors = {"SUPPORTED": "#2da44e", "LIMITED": "#d29922", "UNAVAILABLE": "#cf222e"}
    for category in ("SUPPORTED", "LIMITED", "UNAVAILABLE"):
        axis.barh(df.index, df[category], left=bottom, label=category, color=colors[category])
        bottom = df[category] if bottom is None else bottom + df[category]
    axis.set_xlabel("feature count")
    axis.set_title("Feature Evidence by Body Region")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_feature_reliability_overview(profile: MovementProfile, output_path: str | Path) -> Path:
    """Plot geometry and dynamic completeness for each feature."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt
    import numpy as np

    features = sorted(
        profile.trajectory_summaries,
        key=lambda item: (item.body_region, item.feature_name),
    )
    names = [feature.feature_name for feature in features]
    y = np.arange(len(features))
    fig, axis = plt.subplots(figsize=(12, max(8, 0.32 * len(features))))
    axis.barh(y - 0.18, [f.geometry_completeness for f in features], height=0.32, label="geometry", color="#0969da")
    axis.barh(y + 0.18, [f.dynamic_completeness for f in features], height=0.32, label="robust dynamics", color="#2da44e")
    for idx, feature in enumerate(features):
        marker = "*" if feature.analytics_eligibility == "ANALYTICS_READY" else "|"
        axis.text(1.02, idx, marker, va="center", fontsize=10)
    axis.set_yticks(y, names, fontsize=7)
    axis.set_xlim(0, 1.12)
    axis.set_xlabel("completeness")
    axis.set_title("Feature Reliability Overview (* analytics-ready)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_evidence_category_counts(profile: MovementProfile, output_path: str | Path) -> Path:
    """Plot supported vs limited vs unavailable feature totals."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    counts = profile.feature_availability["by_quality_category"]
    categories = [QualityCategory.SUPPORTED.value, QualityCategory.LIMITED.value, QualityCategory.UNAVAILABLE.value]
    values = [counts.get(category, 0) for category in categories]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(categories, values, color=["#2da44e", "#d29922", "#cf222e"])
    axis.set_ylabel("feature count")
    axis.set_title("Supported vs Limited vs Unavailable")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_rejection_reasons(profile: MovementProfile, output_path: str | Path) -> Path:
    """Plot primary feature rejection/limitation reasons."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    reasons = [
        feature.primary_rejection_reason
        for feature in profile.trajectory_summaries
        if feature.primary_rejection_reason
    ]
    counts = pd.Series(reasons).value_counts().head(8)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    if counts.empty:
        axis.text(0.5, 0.5, "No primary rejection reasons", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(range(len(counts)), counts.values, color="#8250df")
        axis.set_yticks(range(len(counts)), [str(item)[:80] for item in counts.index], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("feature count")
        axis.set_title("Primary Evidence Limitations")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_movement_profile_overview(
    dynamic_df: pd.DataFrame,
    profile: MovementProfile,
    output_path: str | Path,
) -> Path:
    """Plot a clean event-relative movement profile overview."""

    output = _prepare_output(output_path)
    import matplotlib.pyplot as plt

    selected = [
        ("left_hka_angle_2d_deg", "lower limb: left HKA"),
        ("right_hka_angle_2d_deg", "lower limb: right HKA"),
        ("projected_trunk_axis_angle_deg", "trunk axis"),
        ("projected_shoulder_line_angle_deg", "shoulder line"),
        ("left_elbow_angle_2d_deg", "upper body: left elbow"),
        ("right_elbow_angle_2d_deg", "upper body: right elbow"),
        ("hka_projected_bilateral_difference_deg", "bilateral HKA difference"),
        ("projected_shoulder_pelvis_x_offset_normalized", "shoulder-pelvis x offset"),
    ]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 11), sharex=True)
    for axis, (feature_name, label) in zip(axes.flat, selected, strict=True):
        rows = dynamic_df[dynamic_df["feature_name"].eq(feature_name)].sort_values("event_relative_ms")
        if rows.empty:
            axis.text(0.5, 0.5, "unavailable", transform=axis.transAxes, ha="center")
            continue
        values = rows["feature_value"].where(rows["feature_status"].eq("SUPPORTED"))
        axis.plot(rows["event_relative_ms"], values, marker="o", markersize=2.4, linewidth=1.3)
        axis.axvline(0, color="#cf222e", linestyle="--", linewidth=1.1)
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
    axes[-1][0].set_xlabel("event_relative_ms")
    axes[-1][1].set_xlabel("event_relative_ms")
    fig.suptitle(f"Movement Profile Overview: {profile.case['case_id']}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _prepare_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
