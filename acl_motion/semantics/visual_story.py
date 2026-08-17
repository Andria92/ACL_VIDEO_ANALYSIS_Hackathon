"""Movement-first visual story payloads for the HUMAN Results experience."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

PHASE_VISUAL_STORY_VERSION = "m5_9_movement_first_visual_story_v1"

STORY_CATEGORY_ORDER = (
    "movement_path",
    "hip_knee_ankle_chain",
    "bilateral_limb_relationship",
    "trunk_pelvis",
    "upper_body",
)

STORY_CATEGORY_LABELS = {
    "movement_path": "Movement Path",
    "hip_knee_ankle_chain": "Lower Body",
    "bilateral_limb_relationship": "Relationship Between the Two Legs",
    "trunk_pelvis": "Trunk & Pelvis",
    "upper_body": "Upper Body",
}

TECHNICAL_CATEGORY_LABELS = {
    "movement_path": "Projected movement path",
    "hip_knee_ankle_chain": "Projected hip-knee-ankle chain",
    "bilateral_limb_relationship": "Projected bilateral HKA relationship",
    "trunk_pelvis": "Projected trunk, hip-line, and shoulder-line geometry",
    "upper_body": "Projected elbow and upper-arm geometry",
}

EVIDENCE_WEIGHTS = {
    "HIGH": 1.0,
    "GOOD": 0.9,
    "SUPPORTED": 0.9,
    "MODERATE": 0.65,
    "LIMITED": 0.45,
    "LOW": 0.2,
    "UNAVAILABLE": 0.0,
}


def build_movement_visual_story(
    *,
    movement_story: dict,
    metric_explorer: dict,
    processed_pose: pd.DataFrame | None = None,
    laterality_mapping: dict[str, str] | None = None,
) -> dict:
    """Build deterministic, scope-aware visual story records for the Results UI.

    The story ranks movement families by supported within-phase change, evidence
    quality, contribution to the incoming phase boundary, and distinctiveness
    from the preceding phase. It does not use clinical labels or learned text.
    """

    phases = movement_story.get("phases", [])
    transitions = movement_story.get("transitions", [])
    pose_index = _pose_frame_index(processed_pose)
    visual_phases = []
    for index, phase in enumerate(phases):
        previous_phase = phases[index - 1] if index else None
        incoming = _incoming_transition(phase, transitions)
        ranked_observations = _phase_observations(
            phase=phase,
            previous_phase=previous_phase,
            incoming_transition=incoming,
            metric_explorer=metric_explorer,
            limit=None,
        )
        observations = _limited_phase_observations(ranked_observations)
        visual_phases.append(
            {
                "phase_id": phase.get("phase_id"),
                "phase_index": phase.get("phase_index"),
                "title": phase.get("title"),
                "scope_label": _phase_scope_label(phase, len(phases)),
                "comparison_sentence": _comparison_sentence(phase, observations, incoming),
                "observations": observations,
                "other_observations": ranked_observations[len(observations) :],
                "snapshot_frames": _snapshot_frames(
                    phase,
                    pose_index,
                    observations=observations,
                    metric_explorer=metric_explorer,
                ),
                "visuals": _phase_visuals(phase, observations, metric_explorer, laterality_mapping),
            }
        )
    return {
        "story_version": PHASE_VISUAL_STORY_VERSION,
        "salience_algorithm": (
            "score = 0.40*within-phase change magnitude + 0.25*evidence quality + "
            "0.25*incoming transition contribution + 0.10*distinctiveness from the "
            "preceding phase; categories are capped so technical metric count cannot "
            "make a family salient by itself."
        ),
        "category_labels": STORY_CATEGORY_LABELS,
        "technical_category_labels": TECHNICAL_CATEGORY_LABELS,
        "laterality_mapping": laterality_mapping or {},
        "whole_movement": _whole_movement_story(movement_story, visual_phases, metric_explorer),
        "phases": visual_phases,
    }


def _phase_observations(
    *,
    phase: dict,
    previous_phase: dict | None,
    incoming_transition: dict | None,
    metric_explorer: dict | None = None,
    limit: int | None = 4,
) -> list[dict]:
    ranked = []
    for category in STORY_CATEGORY_ORDER:
        summary = (phase.get("category_summaries") or {}).get(category)
        if not summary:
            continue
        evidence = str(summary.get("evidence_status", "UNAVAILABLE"))
        evidence_score = EVIDENCE_WEIGHTS.get(evidence, 0.0)
        if evidence_score <= 0:
            continue
        magnitude, metric_summary = _category_magnitude(category, summary)
        support = _support_detail(
            phase,
            summary,
            metric_summary,
            metric_explorer=metric_explorer or {},
            category=category,
        )
        transition_score = _transition_fraction(category, incoming_transition)
        if magnitude < 0.08 and transition_score < 0.12:
            continue
        if magnitude <= 0 and evidence_score < 0.8:
            continue
        distinctiveness = _distinctiveness(category, summary, previous_phase)
        score = (
            0.40 * magnitude
            + 0.25 * evidence_score
            + 0.25 * transition_score
            + 0.10 * distinctiveness
        )
        if score <= 0.15:
            continue
        ranked.append(
            {
                "category": category,
                "display_label": STORY_CATEGORY_LABELS[category],
                "technical_label": TECHNICAL_CATEGORY_LABELS[category],
                "evidence_status": evidence,
                "summary": str(summary.get("summary", "")),
                "plain_language": _plain_language(category, summary, metric_summary),
                "salience_score": round(float(score), 4),
                "score_components": {
                    "within_phase_change": round(float(magnitude), 4),
                    "evidence_quality": round(float(evidence_score), 4),
                    "transition_contribution": round(float(transition_score), 4),
                    "distinctiveness_from_previous": round(float(distinctiveness), 4),
                },
                "primary_metrics": metric_summary,
                "support": support,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["salience_score"],
            -STORY_CATEGORY_ORDER.index(item["category"]),
        ),
        reverse=True,
    )
    return ranked if limit is None else _limited_phase_observations(ranked, limit=limit)


def _limited_phase_observations(ranked: list[dict], *, limit: int = 4) -> list[dict]:
    supported = ranked[:limit]
    if len(supported) < 2:
        fallback = [item for item in ranked if item not in supported]
        supported.extend(fallback[: 2 - len(supported)])
    return supported[:limit]


def _category_magnitude(category: str, summary: dict) -> tuple[float, dict]:
    metrics = summary.get("metrics") or {}
    if category == "movement_path":
        heading_change = _optional_float(metrics.get("heading_change_deg"))
        speed_change = _optional_float(metrics.get("speed_change_normalized_per_s"))
        magnitude = min(
            1.0,
            abs(heading_change or 0.0) / 120.0 + abs(speed_change or 0.0) / 6.0,
        )
        return magnitude, {
            "heading_change_deg": heading_change,
            "speed_change_normalized_per_s": speed_change,
            "mean_normalized_projected_speed_per_s": _optional_float(
                metrics.get("mean_normalized_projected_speed_per_s")
            ),
        }
    if category == "bilateral_limb_relationship":
        signed_change = _optional_float(metrics.get("signed_difference_change_deg"))
        absolute_change = _optional_float(metrics.get("absolute_difference_change_deg"))
        max_abs = _optional_float(metrics.get("maximum_absolute_hka_difference_deg"))
        magnitude = min(1.0, max(abs(signed_change or 0.0), abs(absolute_change or 0.0)) / 55.0)
        return magnitude, {
            "signed_difference_start_deg": _optional_float(metrics.get("signed_difference_start_deg")),
            "signed_difference_end_deg": _optional_float(metrics.get("signed_difference_end_deg")),
            "signed_difference_change_deg": signed_change,
            "absolute_difference_change_deg": absolute_change,
            "maximum_absolute_hka_difference_deg": max_abs,
            "relationship_pattern": metrics.get("relationship_pattern"),
        }
    changes = _nested_changes(metrics)
    largest_name, largest_change = _largest_change(changes)
    divisor = 75.0 if category in {"hip_knee_ankle_chain", "upper_body"} else 120.0
    magnitude = min(1.0, abs(largest_change or 0.0) / divisor)
    return magnitude, {
        "largest_metric_name": largest_name,
        "largest_metric_label": _metric_label(largest_name),
        "largest_change": largest_change,
        "all_changes": changes,
    }


def _support_detail(
    phase: dict,
    summary: dict,
    metrics: dict,
    *,
    metric_explorer: dict,
    category: str,
) -> dict:
    frame_count = int((phase.get("evidence_summary") or {}).get("frame_count") or 0)
    if frame_count <= 0:
        frame_count = int(phase.get("end_frame", 0)) - int(phase.get("start_frame", 0)) + 1
    raw_metrics = summary.get("metrics") or {}
    supported_samples = _optional_float(raw_metrics.get("supported_samples"))
    supported_fraction = _optional_float(raw_metrics.get("supported_fraction"))
    if supported_samples is None:
        largest = metrics.get("largest_metric_name")
        if largest and isinstance(raw_metrics.get(largest), dict):
            supported_samples = _optional_float(raw_metrics[largest].get("supported_samples"))
            supported_fraction = _optional_float(raw_metrics[largest].get("supported_fraction"))
    if supported_samples is None:
        metric_name = _support_metric_for_category(category, metrics)
        if metric_name:
            supported_samples = _supported_series_count(metric_explorer, metric_name, phase)
    if supported_fraction is None and supported_samples is not None and frame_count > 0:
        supported_fraction = supported_samples / frame_count
    if supported_samples is None and supported_fraction is not None and frame_count > 0:
        supported_samples = round(supported_fraction * frame_count)
    reason = _support_reason(
        status=str(summary.get("evidence_status", "UNAVAILABLE")),
        supported_samples=supported_samples,
        frame_count=frame_count,
    )
    return {
        "supported_samples": int(supported_samples) if supported_samples is not None else None,
        "relevant_frames": frame_count,
        "supported_fraction": supported_fraction,
        "reason": reason,
    }


def _support_metric_for_category(category: str, metrics: dict) -> str | None:
    if category == "movement_path":
        return "path:projected_heading_deg"
    if category == "bilateral_limb_relationship":
        return "hka_projected_bilateral_difference_deg"
    return metrics.get("largest_metric_name")


def _supported_series_count(metric_explorer: dict, metric_name: str, phase: dict) -> int | None:
    rows = metric_explorer.get("series", {}).get(metric_name)
    if not rows:
        return None
    start = int(phase.get("start_frame", 0))
    end = int(phase.get("end_frame", start))
    return sum(
        1
        for row in rows
        if start <= int(row["source_frame_index"]) <= end
        and row.get("evidence_status") == "SUPPORTED"
        and _optional_float(row.get("value")) is not None
    )


def _support_reason(
    *,
    status: str,
    supported_samples: float | None,
    frame_count: int,
) -> str:
    if supported_samples is None or frame_count <= 0:
        return "Compact support count unavailable; open Research Measurement for frame-level evidence."
    missing = max(frame_count - int(supported_samples), 0)
    if missing == 0:
        return f"All {frame_count} phase frames support this measurement."
    if status == "GOOD":
        return f"{int(supported_samples)} of {frame_count} phase frames support this measurement; isolated unsupported samples remain excluded."
    if status == "MODERATE":
        return f"{int(supported_samples)} of {frame_count} phase frames support this measurement; some frames are unavailable or uncertain."
    if status == "LIMITED":
        return f"{int(supported_samples)} of {frame_count} phase frames support this measurement; a substantial portion is unavailable."
    return f"{int(supported_samples)} of {frame_count} phase frames support this measurement."


def _plain_language(category: str, summary: dict, metrics: dict) -> str:
    evidence = str(summary.get("evidence_status", "UNAVAILABLE")).lower()
    if category == "movement_path":
        heading = metrics.get("heading_change_deg")
        speed = metrics.get("speed_change_normalized_per_s")
        parts = []
        if heading is not None:
            parts.append(f"projected heading changed by {heading:.1f} degrees")
        if speed is not None:
            direction = "increased" if speed > 0 else "decreased"
            parts.append(f"projected speed {direction} by {abs(speed):.2f} body-scale units/s")
        return _sentence("During this phase, " + " and ".join(parts), evidence)
    if category == "bilateral_limb_relationship":
        start = metrics.get("signed_difference_start_deg")
        end = metrics.get("signed_difference_end_deg")
        change = metrics.get("signed_difference_change_deg")
        if start is not None and end is not None:
            return _sentence(
                "The projected relationship between injured and contralateral HKA "
                f"moved from {start:.1f} degrees to {end:.1f} degrees",
                evidence,
            )
        if change is not None:
            return _sentence(
                f"The projected bilateral HKA relationship changed by {change:.1f} degrees",
                evidence,
            )
    if category == "hip_knee_ankle_chain":
        label = metrics.get("largest_metric_label") or "the projected HKA chain"
        change = metrics.get("largest_change")
        if change is not None:
            return _sentence(f"{label} changed by {change:.1f} degrees", evidence)
    if category == "trunk_pelvis":
        label = metrics.get("largest_metric_label") or "projected trunk/pelvis orientation"
        change = metrics.get("largest_change")
        if change is not None:
            return _sentence(f"{label} changed by {change:.1f} degrees", evidence)
    if category == "upper_body":
        label = metrics.get("largest_metric_label") or "projected upper-body geometry"
        change = metrics.get("largest_change")
        if change is not None:
            return _sentence(f"{label} changed by {change:.1f} degrees", evidence)
    return str(summary.get("summary", "Supported movement evidence is available for this phase."))


def _sentence(text: str, evidence: str) -> str:
    if not text:
        return ""
    text = text.rstrip(".")
    if evidence in {"limited", "low", "unavailable"}:
        text += f" with {evidence} evidence"
    return f"{text}."


def _comparison_sentence(phase: dict, observations: list[dict], incoming_transition: dict | None) -> str:
    if not observations:
        return "This phase has limited supported movement evidence."
    labels = [item["display_label"].lower() for item in observations[:3]]
    if incoming_transition and incoming_transition.get("dominant_feature_families"):
        contributors = [
            STORY_CATEGORY_LABELS.get(item, item.replace("_", " "))
            for item in incoming_transition["dominant_feature_families"][:3]
        ]
        return (
            "Compared with the preceding phase, this boundary was driven mainly by "
            f"{_join_plain(contributors).lower()}."
        )
    if int(phase.get("phase_index", 1)) == 1:
        return f"This opening phase is described mainly by {_join_plain(labels)}."
    return f"Compared with the preceding phase, the most salient supported changes are {_join_plain(labels)}."


def _phase_visuals(
    phase: dict,
    observations: list[dict],
    metric_explorer: dict,
    laterality_mapping: dict[str, str] | None,
) -> list[dict]:
    visuals = []
    observed_categories = {item["category"] for item in observations}
    if "movement_path" in observed_categories:
        visuals.append(
            {
                "category": "movement_path",
                "visual_type": "projected_path",
                "title": "Projected movement path",
                "points": _path_points(metric_explorer, phase),
                "metrics": _category_metrics(phase, "movement_path"),
            }
        )
        visuals.append(
            {
                "category": "movement_path",
                "visual_type": "projected_speed_sparkline",
                "title": "Projected speed",
                "points": _metric_points(
                    metric_explorer,
                    "path:normalized_projected_speed_per_s",
                    phase,
                ),
            }
        )
    if "hip_knee_ankle_chain" in observed_categories:
        visuals.append(
            {
                "category": "hip_knee_ankle_chain",
                "visual_type": "hka_start_end",
                "title": "Projected hip-knee-ankle chain",
                "metrics": _category_metrics(phase, "hip_knee_ankle_chain"),
                "laterality_mapping": laterality_mapping or {},
            }
        )
    if "bilateral_limb_relationship" in observed_categories:
        visuals.append(
            {
                "category": "bilateral_limb_relationship",
                "visual_type": "bilateral_difference",
                "title": "Injured vs contralateral projected HKA",
                "metrics": _category_metrics(phase, "bilateral_limb_relationship"),
            }
        )
    if "trunk_pelvis" in observed_categories:
        visuals.append(
            {
                "category": "trunk_pelvis",
                "visual_type": "trunk_pelvis_axes",
                "title": "Projected trunk and pelvis axes",
                "metrics": _category_metrics(phase, "trunk_pelvis"),
            }
        )
    if "upper_body" in observed_categories:
        visuals.append(
            {
                "category": "upper_body",
                "visual_type": "upper_body_pose",
                "title": "Projected upper-body orientation",
                "metrics": _category_metrics(phase, "upper_body"),
            }
        )
    return visuals


def _snapshot_frames(
    phase: dict,
    pose_index: dict[int, dict],
    *,
    observations: list[dict] | None = None,
    metric_explorer: dict | None = None,
) -> list[dict]:
    start = int(phase.get("start_frame", 0))
    end = int(phase.get("end_frame", start))
    length = end - start + 1
    if length >= 12:
        targets = (
            ("Phase start", start),
            ("25%", round(start + (end - start) * 0.25)),
            ("50%", round(start + (end - start) * 0.50)),
            ("75%", round(start + (end - start) * 0.75)),
            ("Phase end", end),
        )
    else:
        targets = (
            ("Phase start", start),
            ("Mid-phase", round((start + end) / 2)),
            ("Phase end", end),
        )
    snapshots = [
        _nearest_snapshot(label, int(target), start, end, pose_index)
        for label, target in targets
    ]
    salient = _salient_snapshots(
        phase=phase,
        observations=observations or [],
        metric_explorer=metric_explorer or {},
        pose_index=pose_index,
    )
    salient_by_frame = {int(item["source_frame_index"]): item for item in salient}
    for snapshot in snapshots:
        frame = int(snapshot["source_frame_index"])
        if frame in salient_by_frame:
            snapshot.update(
                {
                    key: value
                    for key, value in salient_by_frame[frame].items()
                    if key.startswith("change_")
                }
            )
    seen: set[int] = set()
    unique = []
    for snapshot in snapshots:
        frame = int(snapshot["source_frame_index"])
        if frame in seen and snapshot["label"] not in {"Phase start", "Phase end"}:
            continue
        seen.add(frame)
        unique.append(snapshot)
    return unique


def _salient_snapshots(
    *,
    phase: dict,
    observations: list[dict],
    metric_explorer: dict,
    pose_index: dict[int, dict],
) -> list[dict]:
    start = int(phase.get("start_frame", 0))
    end = int(phase.get("end_frame", start))
    if end - start < 3:
        return []
    candidates = []
    for observation in observations:
        category = str(observation.get("category", ""))
        if EVIDENCE_WEIGHTS.get(str(observation.get("evidence_status", "UNAVAILABLE")), 0.0) < 0.45:
            continue
        if float((observation.get("score_components") or {}).get("within_phase_change", 0.0)) < 0.12:
            continue
        metric_name = _salient_metric_for_category(category, observation)
        if not metric_name:
            continue
        candidates.extend(
            _metric_change_candidates(
                metric_explorer=metric_explorer,
                metric_name=metric_name,
                phase=phase,
                category=category,
                display_label=str(observation.get("display_label", "")),
            )
        )
    if not candidates:
        return []
    candidates.sort(key=lambda item: item["score"], reverse=True)
    separation = max(3, round((end - start + 1) / 5))
    selected = []
    for candidate in candidates:
        if len(selected) >= 3:
            break
        frame = int(candidate["source_frame_index"])
        if frame in {start, end}:
            continue
        if any(abs(frame - int(item["source_frame_index"])) < separation for item in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: int(item["source_frame_index"]))
    snapshots = []
    for candidate in selected:
        snapshot = _nearest_snapshot(
            candidate["label"],
            int(candidate["source_frame_index"]),
            start,
            end,
            pose_index,
        )
        snapshot.update(
            {
                "change_category": candidate["category"],
                "change_reason": candidate["reason"],
                "change_score": round(float(candidate["score"]), 4),
                "change_intensity": _change_intensity(candidate["score"]),
            }
        )
        snapshots.append(snapshot)
    return snapshots


def _salient_metric_for_category(category: str, observation: dict) -> str | None:
    metrics = observation.get("primary_metrics") or {}
    if category == "movement_path":
        return "path:projected_heading_deg"
    if category == "bilateral_limb_relationship":
        return "hka_projected_bilateral_difference_deg"
    return metrics.get("largest_metric_name")


def _metric_change_candidates(
    *,
    metric_explorer: dict,
    metric_name: str,
    phase: dict,
    category: str,
    display_label: str,
) -> list[dict]:
    rows = [
        row
        for row in metric_explorer.get("series", {}).get(metric_name, [])
        if int(phase["start_frame"]) <= int(row["source_frame_index"]) <= int(phase["end_frame"])
    ]
    candidates = []
    previous_value: float | None = None
    previous_frame: int | None = None
    for row in rows:
        frame = int(row["source_frame_index"])
        value = _optional_float(row.get("value"))
        if row.get("evidence_status") != "SUPPORTED" or value is None:
            previous_value = None
            previous_frame = None
            continue
        if previous_value is not None and previous_frame is not None:
            change = abs(value - previous_value)
            score = change / _salient_scale(metric_name)
            if score >= 0.08:
                candidates.append(
                    {
                        "source_frame_index": frame,
                        "score": min(score, 2.5),
                        "category": category,
                        "label": f"Major change - {display_label}",
                        "reason": _salient_reason(category, metric_name),
                    }
                )
        previous_value = value
        previous_frame = frame
    return candidates


def _salient_scale(metric_name: str) -> float:
    if metric_name.startswith("path:"):
        return 25.0
    if "normalized" in metric_name:
        return 0.25
    return 18.0


def _salient_reason(category: str, metric_name: str) -> str:
    if category == "movement_path":
        return "Strongest supported directional transition in this phase."
    if category == "hip_knee_ankle_chain":
        return "Largest supported HKA-chain change in this phase."
    if category == "bilateral_limb_relationship":
        return "Largest supported injured-contralateral HKA difference change in this phase."
    if category == "trunk_pelvis":
        return "Largest supported trunk/pelvis orientation change in this phase."
    if category == "upper_body":
        return "Largest supported upper-body geometry change in this phase."
    return f"Largest supported change in {metric_name}."


def _change_intensity(score: float) -> str:
    if score >= 1.0:
        return "largest"
    if score >= 0.35:
        return "larger"
    return "lower"


def _nearest_snapshot(
    label: str,
    target_frame: int,
    start_frame: int,
    end_frame: int,
    pose_index: dict[int, dict],
) -> dict:
    candidates = [
        frame
        for frame, record in pose_index.items()
        if start_frame <= frame <= end_frame and record["usable_for_snapshot"]
    ]
    if not candidates:
        candidates = list(range(start_frame, end_frame + 1))
    frame = min(candidates, key=lambda item: (abs(item - target_frame), item))
    record = pose_index.get(frame, {})
    return {
        "label": label,
        "requested_source_frame_index": int(target_frame),
        "source_frame_index": int(frame),
        "frame_status": record.get("frame_status", "UNKNOWN"),
        "selected_nearest_supported": int(frame) != int(target_frame),
        "landmarks": record.get("landmarks", {}),
    }


def _pose_frame_index(processed_pose: pd.DataFrame | None) -> dict[int, dict]:
    if processed_pose is None or processed_pose.empty:
        return {}
    output: dict[int, dict] = {}
    for frame, rows in processed_pose.groupby("source_frame_index", sort=True):
        landmarks = {}
        valid_count = 0
        for _, row in rows.iterrows():
            x = _optional_float(row.get("smoothed_x"))
            y = _optional_float(row.get("smoothed_y"))
            if x is None or y is None:
                x = _optional_float(row.get("clean_x"))
                y = _optional_float(row.get("clean_y"))
            rejected = bool(row.get("rejected"))
            if x is not None and y is not None and not rejected:
                valid_count += 1
                landmarks[str(row["landmark_name"])] = {
                    "x": x,
                    "y": y,
                    "status": str(row.get("landmark_status", "")),
                    "interpolated": bool(row.get("interpolated")),
                    "smoothed": bool(row.get("smoothed")),
                }
        frame_status = str(rows.iloc[0].get("frame_status", "UNKNOWN"))
        output[int(frame)] = {
            "frame_status": frame_status,
            "valid_landmark_count": valid_count,
            "usable_for_snapshot": frame_status == "VALID_TARGET" and valid_count >= 6,
            "landmarks": landmarks,
        }
    return output


def _whole_movement_story(movement_story: dict, visual_phases: list[dict], metric_explorer: dict) -> dict:
    window = (movement_story.get("metadata") or {}).get("movement_window", {})
    duration_s = _optional_float(window.get("duration_ms"))
    phases = movement_story.get("phases", [])
    scope_label = (
        f"Viewing: Whole Movement - {duration_s / 1000.0:.2f} s"
        if duration_s is not None
        else "Viewing: Whole Movement"
    )
    sequence = [
        {
            "phase_id": phase.get("phase_id"),
            "phase_index": phase.get("phase_index"),
            "title": phase.get("title"),
            "duration_ms": phase.get("duration_ms"),
            "summary": (visual.get("observations") or [{"plain_language": ""}])[0]["plain_language"],
        }
        for phase, visual in zip(phases, visual_phases, strict=False)
    ]
    return {
        "scope_label": scope_label,
        "sequence_summary": movement_story.get("sequence_summary", ""),
        "phase_sequence": sequence,
        "path_points": _path_points(metric_explorer, None),
    }


def _path_points(metric_explorer: dict, phase: dict | None) -> list[dict]:
    x_rows = metric_explorer.get("series", {}).get("path:compensated_x", [])
    y_rows = metric_explorer.get("series", {}).get("path:compensated_y", [])
    y_by_frame = {int(row["source_frame_index"]): row for row in y_rows}
    points = []
    for x_row in x_rows:
        frame = int(x_row["source_frame_index"])
        if phase and not int(phase["start_frame"]) <= frame <= int(phase["end_frame"]):
            continue
        y_row = y_by_frame.get(frame)
        x = _optional_float(x_row.get("value"))
        y = _optional_float(y_row.get("value") if y_row else None)
        if x is None or y is None:
            continue
        points.append(
            {
                "source_frame_index": frame,
                "movement_end_relative_ms": _optional_float(x_row.get("movement_end_relative_ms")),
                "x": x,
                "y": y,
                "path_segment_id": x_row.get("path_segment_id") or "",
            }
        )
    return points


def _metric_points(metric_explorer: dict, metric_name: str, phase: dict) -> list[dict]:
    rows = metric_explorer.get("series", {}).get(metric_name, [])
    output = []
    for row in rows:
        frame = int(row["source_frame_index"])
        if int(phase["start_frame"]) <= frame <= int(phase["end_frame"]):
            output.append(
                {
                    "source_frame_index": frame,
                    "movement_end_relative_ms": _optional_float(row.get("movement_end_relative_ms")),
                    "value": _optional_float(row.get("value")),
                    "evidence_status": row.get("evidence_status"),
                }
            )
    return output


def _category_metrics(phase: dict, category: str) -> dict:
    return ((phase.get("category_summaries") or {}).get(category) or {}).get("metrics") or {}


def _incoming_transition(phase: dict, transitions: list[dict]) -> dict | None:
    phase_id = phase.get("phase_id")
    for transition in transitions:
        if transition.get("to_phase_id") == phase_id:
            return transition
    return None


def _transition_fraction(category: str, transition: dict | None) -> float:
    if not transition:
        return 0.0
    contribution = (transition.get("feature_family_contributions") or {}).get(category) or {}
    return min(max(float(contribution.get("fraction", 0.0) or 0.0), 0.0), 1.0)


def _distinctiveness(category: str, summary: dict, previous_phase: dict | None) -> float:
    if not previous_phase:
        return 0.0
    previous_summary = (previous_phase.get("category_summaries") or {}).get(category)
    if not previous_summary:
        return 0.0
    current, _ = _category_magnitude(category, summary)
    previous, _ = _category_magnitude(category, previous_summary)
    return min(abs(current - previous), 1.0)


def _nested_changes(metrics: dict) -> dict[str, float]:
    changes = {}
    for name, values in metrics.items():
        if not isinstance(values, dict):
            continue
        change = _optional_float(values.get("change"))
        if change is not None:
            changes[str(name)] = change
    return changes


def _largest_change(changes: dict[str, float]) -> tuple[str | None, float | None]:
    if not changes:
        return None, None
    name = max(changes, key=lambda key: abs(changes[key]))
    return name, changes[name]


def _metric_label(metric_name: str | None) -> str:
    if not metric_name:
        return ""
    labels = {
        "injured_hka_angle_2d_deg": "Injured projected HKA",
        "contralateral_hka_angle_2d_deg": "Contralateral projected HKA",
        "left_hka_angle_2d_deg": "Left projected HKA",
        "right_hka_angle_2d_deg": "Right projected HKA",
        "projected_trunk_axis_angle_deg": "Projected trunk axis",
        "projected_hip_line_angle_deg": "Projected hip-line orientation",
        "projected_shoulder_line_angle_deg": "Projected shoulder-line orientation",
        "projected_shoulder_pelvis_orientation_difference_deg": (
            "Projected shoulder-pelvis orientation difference"
        ),
        "left_elbow_angle_2d_deg": "Left projected elbow angle",
        "right_elbow_angle_2d_deg": "Right projected elbow angle",
        "left_upper_arm_orientation_2d_deg": "Left projected upper-arm orientation",
        "right_upper_arm_orientation_2d_deg": "Right projected upper-arm orientation",
    }
    return labels.get(metric_name, metric_name.replace("_", " "))


def _phase_scope_label(phase: dict, phase_count: int) -> str:
    return (
        f"Viewing: Phase {phase.get('phase_index')} of {phase_count} - "
        f"{float(phase.get('duration_ms', 0.0)) / 1000.0:.2f} s"
    )


def _join_plain(items: list[str]) -> str:
    clean = [str(item) for item in items if item]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
