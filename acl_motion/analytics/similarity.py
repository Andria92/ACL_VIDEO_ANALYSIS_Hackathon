"""Evidence-aware similarity ranking for analysed movement cases.

The engine consumes case-level records produced by the exploration layer. It never
treats frames or replay views as independent cases, and it compares only descriptors
that are supported for both members of a pair.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from acl_motion.geometry.angular_semantics import (
    ANGULAR_STATISTICS_VERSION,
    range_semantics_for_metric,
)

SIMILARITY_ENGINE_VERSION = "movement_similarity_v5_angular_ranges"
COMPARISON_STATISTICS_VERSION = "comparison_statistics_v1_supported_intervals"
DEFAULT_SIMILARITY_LENS = "overall_movement_difference"
MINIMUM_SHARED_DESCRIPTORS = 6
MINIMUM_SHARED_FAMILIES = 2
MINIMUM_SCALER_CASES = 3
RESAMPLING_ITERATIONS = 200
RESAMPLING_FEATURE_DROPOUT = 0.10
SUPPORTED_REFERENCE_PHASE_STATUSES = {
    "SUPPORTED",
    "SUPPORTED_PARTIAL_WINDOW",
    "SUPPORTED_EVIDENCE_INTERVAL",
}

SIMILARITY_LENSES = (
    {
        "id": "overall_movement_difference",
        "label": "Weighted robust movement difference",
        "measure": "Reliability-weighted, robustly scaled L1 distance",
    },
    {
        "id": "movement_pattern_direction",
        "label": "Weighted cosine sensitivity",
        "measure": "Reliability-weighted, robustly scaled cosine similarity (sensitivity lens)",
    },
    {
        "id": "relationship_aware_pattern",
        "label": "Soft-cosine sensitivity",
        "measure": "Reliability-weighted soft cosine similarity (sensitivity lens)",
    },
    {
        "id": "large_difference_focus",
        "label": "Weighted Euclidean sensitivity",
        "measure": "Reliability-weighted, robustly scaled Euclidean distance (sensitivity lens)",
    },
)

_DESCRIPTOR_STATISTICS = (
    ("mean", "Overall level", "geometry_analytics_eligible", "geometry_completeness"),
    ("range", "Movement range", "geometry_analytics_eligible", "geometry_completeness"),
    (
        "pre_late_change",
        "Early-to-late change",
        "dynamic_analytics_eligible",
        "dynamic_completeness",
    ),
)
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12}",
    re.IGNORECASE,
)


class SimilarityComputationCancelled(RuntimeError):
    """Raised when a newer interactive comparison supersedes this computation."""


def build_similarity_payload(
    records: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    view_records: Iterable[Mapping[str, Any]] | None = None,
    selected_case_id: str = "",
    result_limit: int = 6,
    reference_scalers: Mapping[str, Mapping[str, float]] | None = None,
    scaler_provenance: str = "",
    resampling_iterations: int = RESAMPLING_ITERATIONS,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build player choices and per-lens nearest-neighbour rankings."""

    record_list = [dict(record) for record in records]
    event_list = [dict(event) for event in events]
    if cancelled is not None and cancelled():
        raise SimilarityComputationCancelled("Comparison superseded by a newer request.")
    if resampling_iterations < 0:
        raise ValueError("resampling_iterations cannot be negative.")
    vectors, descriptors = _case_vectors(record_list)
    if view_records is None:
        view_vectors, view_metadata, case_view_ids = _synthetic_view_vectors(vectors)
    else:
        view_vectors, view_metadata, case_view_ids, view_descriptors = _view_vectors(
            [dict(record) for record in view_records]
        )
        descriptors.update(view_descriptors)
    event_lookup = {str(event.get("case_id", "")): event for event in event_list}
    unscaled_public_cases = _public_cases(event_lookup, vectors)
    public_ids = {case["case_id"] for case in unscaled_public_cases}
    unscaled_public_case_lookup = {
        case["case_id"]: case for case in unscaled_public_cases
    }
    selected_id = str(selected_case_id or "")
    if selected_id not in unscaled_public_case_lookup:
        selected_id = ""
    reference_ids = {
        case["case_id"]
        for case in unscaled_public_cases
        if case["reference_pool_eligible"]
    }
    if reference_scalers is None:
        scaler_reference_ids = reference_ids.difference({selected_id})
        query_excluded = bool(selected_id and selected_id in reference_ids)
        if not query_excluded:
            scaler_reference_ids = reference_ids
        scalers = _descriptor_scalers(
            {
                case_id: vectors[case_id]
                for case_id in scaler_reference_ids
                if case_id in vectors
            }
        )
        scaling = {
            "status": (
                "QUERY_EXCLUDED_EXPLORATORY_ESTIMATE"
                if query_excluded
                else "EXPLORATORY_POOL_ESTIMATE"
            ),
            "label": (
                "Query-excluded eligible-pool scaling"
                if query_excluded
                else "Current eligible-pool scaling"
            ),
            "source": (
                "completed event-covered reference cases excluding the selected query"
                if query_excluded
                else "completed event-covered reference cases"
            ),
            "reference_case_count": len(scaler_reference_ids),
            "frozen": False,
            "query_excluded": query_excluded,
            "note": (
                "The selected query does not influence its own feature scales. Scaling is still "
                "estimated from the current eligible reference pool and must be replaced by "
                "frozen parameters from a separate, substantially larger cohort before "
                "external validation."
                if query_excluded
                else "Scaling is estimated from the current eligible reference pool. It must be "
                "replaced by frozen parameters from a separate, substantially larger cohort "
                "before external validation."
            ),
        }
    else:
        scalers = _validated_reference_scalers(reference_scalers)
        scaling = {
            "status": "FROZEN_EXTERNAL_REFERENCE",
            "label": "Frozen external-reference scaling",
            "source": str(scaler_provenance or "separate reference cohort"),
            "reference_case_count": None,
            "frozen": True,
            "query_excluded": None,
            "note": "Scaling parameters were supplied independently of query and candidate cases.",
        }
    vectors = {
        case_id: {key: value for key, value in vector.items() if key in scalers}
        for case_id, vector in vectors.items()
        if case_id in public_ids
    }
    view_vectors = {
        view_id: {key: value for key, value in vector.items() if key in scalers}
        for view_id, vector in view_vectors.items()
        if str(view_metadata.get(view_id, {}).get("case_id") or "") in public_ids
    }
    case_view_ids = {
        case_id: [view_id for view_id in view_ids if view_id in view_vectors]
        for case_id, view_ids in case_view_ids.items()
        if case_id in public_ids
    }
    public_cases = _public_cases(event_lookup, vectors)
    public_case_lookup = {case["case_id"]: case for case in public_cases}
    reference_ids = {
        case_id
        for case_id, case in public_case_lookup.items()
        if case["reference_pool_eligible"]
    }

    pair_scores: dict[tuple[str, str], dict[str, Any]] = {}
    comparable_pair_count = 0
    case_ids = sorted(public_case_lookup)
    for left_index, left_id in enumerate(case_ids):
        if cancelled is not None and cancelled():
            raise SimilarityComputationCancelled("Comparison superseded by a newer request.")
        for right_id in case_ids[left_index + 1 :]:
            if left_id not in reference_ids and right_id not in reference_ids:
                continue
            comparison = _compare_best_view_pair(
                left_id,
                right_id,
                view_vectors,
                view_metadata,
                case_view_ids,
                descriptors,
                scalers,
                len(reference_ids),
                reference_ids,
            )
            pair_scores[(left_id, right_id)] = comparison
            if (
                comparison["available"]
                and left_id in reference_ids
                and right_id in reference_ids
            ):
                comparable_pair_count += 1

    if selected_id not in public_case_lookup:
        selected_id = ""
    rankings = (
        _rankings_for_case(
            selected_id,
            public_case_lookup,
            reference_ids,
            pair_scores,
            vectors,
            view_vectors,
            view_metadata,
            case_view_ids,
            descriptors,
            scalers,
            result_limit=max(1, int(result_limit)),
            resampling_iterations=int(resampling_iterations),
            cancelled=cancelled,
        )
        if selected_id
        else {lens["id"]: [] for lens in SIMILARITY_LENSES}
    )
    comparable_case_count = sum(
        1
        for case in public_cases
        if case["reference_pool_eligible"]
        and case["comparable_descriptor_count"] >= MINIMUM_SHARED_DESCRIPTORS
    )
    available = comparable_case_count >= 2 and comparable_pair_count > 0
    return {
        "engine_version": SIMILARITY_ENGINE_VERSION,
        "analysis_unit": "registered_injury_case",
        "available": available,
        "status": "AVAILABLE" if available else "INSUFFICIENT_COMPARABLE_CASES",
        "reason": (
            "A selected case may be query-only when phases are unavailable. Rankings compare its supported whole-movement measurements only against completed, phase-supported reference views whose visible event is covered."
            if available
            else (
                f"At least {MINIMUM_SCALER_CASES} completed, event-covered reference cases "
                "with enough shared measurements are required."
            )
        ),
        "cases": public_cases,
        "selected_case": public_case_lookup.get(selected_id),
        "rankings": rankings,
        "lenses": [dict(lens) for lens in SIMILARITY_LENSES],
        "summary": {
            "analysed_case_count": len(public_cases),
            "reference_pool_case_count": len(reference_ids),
            "query_only_case_count": len(public_cases) - len(reference_ids),
            "comparable_case_count": comparable_case_count,
            "comparable_pair_count": comparable_pair_count,
            "eligible_descriptor_count": len(scalers),
        },
        "index_note": (
            "The lens-specific index runs from 0 to 1. It is not a percentage or probability, "
            "and values from different lenses should not be compared directly. Views are not "
            "averaged: the displayed index and measurements come from the single eligible view "
            "pair with the highest primary-lens similarity."
        ),
        "view_selection_note": (
            "Each injury is listed once. For a query-only case, all analysed views with enough "
            "supported measurements can be checked; reference views must additionally have "
            "supported phases and event coverage. The pair with the highest weighted robust "
            "movement similarity is retained, and the number checked is disclosed per result."
        ),
        "evidence_support_note": (
            "Evidence support is a categorical audit of overlap, measurement support, camera-view "
            "compatibility, and library size. Stability is reported separately from deterministic "
            "feature-dropout resampling. Neither is diagnostic confidence."
        ),
        "scaling": scaling,
        "validation": {
            "status": "NOT_EXTERNALLY_VALIDATED",
            "required": [
                "repeated independent annotation agreement",
                "laboratory or multi-camera reference comparison",
                "blinded expert pairwise similarity judgements",
                "held-out player ranking evaluation",
                "frozen scalers from a separate larger reference cohort",
            ],
        },
    }


def similarity_readiness(
    records: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    view_records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return compact readiness fields for the home and exploration pages."""

    payload = build_similarity_payload(records, events, view_records=view_records)
    summary = payload["summary"]
    return {
        "available": payload["available"],
        "status": payload["status"],
        "comparable_case_count": summary["comparable_case_count"],
        "reference_pool_case_count": summary["reference_pool_case_count"],
        "query_only_case_count": summary["query_only_case_count"],
        "pairwise_output_count": summary["comparable_pair_count"],
        "eligible_descriptor_count": summary["eligible_descriptor_count"],
        "reason": payload["reason"],
        "scientific_note": (
            "Movement similarity is a descriptive comparison of mutually supported "
            "projected measurements, not evidence of the same ACL mechanism or biological cause."
        ),
    }


def _case_vectors(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, str]]]:
    vectors: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    descriptors: dict[str, dict[str, str]] = {}
    for record in records:
        case_id = str(record.get("statistical_unit_id") or record.get("case_id") or "")
        feature_name = str(record.get("feature_name") or "")
        if not case_id or not feature_name:
            continue
        family = str(record.get("body_region") or record.get("feature_family") or "other")
        perspective = str(record.get("perspective") or "unknown")
        for statistic, statistic_label, eligible_key, completeness_key in _DESCRIPTOR_STATISTICS:
            if not _descriptor_provenance_supported(record, feature_name, statistic):
                continue
            value = _finite(record.get(statistic))
            if value is None or not bool(record.get(eligible_key, False)):
                continue
            if statistic == "pre_late_change" and str(
                record.get("phase_status") or "UNKNOWN"
            ).upper() != "SUPPORTED":
                continue
            descriptor_id = f"{feature_name}::{statistic}"
            quality = _clip(_finite(record.get(completeness_key)) or 0.0)
            vectors[case_id][descriptor_id] = {
                "value": value,
                "quality": quality,
                "perspective": perspective,
            }
            descriptors[descriptor_id] = {
                "feature_name": feature_name,
                "feature_label": _feature_label(feature_name),
                "statistic": statistic,
                "statistic_label": statistic_label,
                "family": family,
                "family_label": _family_label(family),
                "unit": _descriptor_unit(record, feature_name),
            }
    return dict(vectors), descriptors


def _view_vectors(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, dict[str, str]],
]:
    """Build one intact feature vector per eligible camera view."""

    vectors: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}
    case_view_ids: dict[str, list[str]] = defaultdict(list)
    descriptors: dict[str, dict[str, str]] = {}
    for record in records:
        case_id = str(record.get("statistical_unit_id") or record.get("case_id") or "")
        source_id = str(record.get("source_id") or "")
        feature_name = str(record.get("feature_name") or "")
        if not case_id or not source_id or not feature_name:
            continue
        if source_id not in metadata:
            phase_status = str(record.get("phase_status") or "UNKNOWN").upper()
            reference_eligibility_supplied = (
                "phase_status" in record or "event_comparison_eligible" in record
            )
            metadata[source_id] = {
                "case_id": case_id,
                "source_id": source_id,
                "case_slug": str(record.get("case_slug") or ""),
                "view_label": str(record.get("view_label") or source_id),
                "perspective": str(record.get("perspective") or "unknown"),
                "primary_view": bool(record.get("primary_view", False)),
                "event_interval_review_decision": str(
                    record.get("event_interval_review_decision") or ""
                ),
                "event_interval_review_status": str(
                    record.get("event_interval_review_status") or ""
                ),
                "phase_status": phase_status,
                "reference_view_eligible": (
                    phase_status in SUPPORTED_REFERENCE_PHASE_STATUSES
                    and bool(record.get("event_comparison_eligible", False))
                    if reference_eligibility_supplied
                    else True
                ),
            }
            case_view_ids[case_id].append(source_id)
        family = str(record.get("body_region") or record.get("feature_family") or "other")
        perspective = str(record.get("perspective") or "unknown")
        for statistic, statistic_label, eligible_key, completeness_key in _DESCRIPTOR_STATISTICS:
            if not _descriptor_provenance_supported(record, feature_name, statistic):
                continue
            value = _finite(record.get(statistic))
            if value is None or not bool(record.get(eligible_key, False)):
                continue
            if statistic == "pre_late_change" and str(
                record.get("phase_status") or "UNKNOWN"
            ).upper() != "SUPPORTED":
                continue
            descriptor_id = f"{feature_name}::{statistic}"
            vectors[source_id][descriptor_id] = {
                "value": value,
                "quality": _clip(_finite(record.get(completeness_key)) or 0.0),
                "perspective": perspective,
            }
            descriptors[descriptor_id] = {
                "feature_name": feature_name,
                "feature_label": _feature_label(feature_name),
                "statistic": statistic,
                "statistic_label": statistic_label,
                "family": family,
                "family_label": _family_label(family),
                "unit": _descriptor_unit(record, feature_name),
            }
    return dict(vectors), metadata, dict(case_view_ids), descriptors


def _synthetic_view_vectors(
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    """Preserve the public API for callers that already supply one vector per case."""

    output: dict[str, dict[str, dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    case_view_ids: dict[str, list[str]] = {}
    for case_id, vector in vectors.items():
        view_id = f"{case_id}::eligible_view"
        output[view_id] = {key: dict(value) for key, value in vector.items()}
        metadata[view_id] = {
            "case_id": case_id,
            "source_id": view_id,
            "case_slug": "",
            "view_label": "Eligible analysed view",
            "perspective": "unknown",
            "primary_view": True,
            "event_interval_review_decision": "yes",
            "event_interval_review_status": "LEGACY_CALLER",
            "phase_status": "SUPPORTED",
            "reference_view_eligible": True,
        }
        case_view_ids[case_id] = [view_id]
    return output, metadata, case_view_ids


def _descriptor_unit(record: Mapping[str, Any], feature_name: str) -> str:
    unit = str(record.get("unit") or "").strip()
    if unit:
        return unit
    if feature_name.endswith("_deg") or "angle" in feature_name:
        return "deg"
    if "normalized" in feature_name:
        return "normalized units"
    return ""


def _descriptor_provenance_supported(
    record: Mapping[str, Any],
    feature_name: str,
    statistic: str,
) -> bool:
    if (
        str(record.get("comparison_statistics_version") or "")
        != COMPARISON_STATISTICS_VERSION
    ):
        return False
    expected = range_semantics_for_metric(feature_name)
    if statistic == "mean" and expected != "linear":
        return False
    if statistic != "range":
        return True
    if expected == "linear":
        return True
    return (
        str(record.get("range_semantics") or "") == expected
        and str(record.get("angular_statistics_version") or "")
        == ANGULAR_STATISTICS_VERSION
    )


def _descriptor_scalers(
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for vector in vectors.values():
        for descriptor_id, item in vector.items():
            values[descriptor_id].append(float(item["value"]))
    scalers: dict[str, dict[str, float]] = {}
    for descriptor_id, descriptor_values in values.items():
        if len(descriptor_values) < MINIMUM_SCALER_CASES:
            continue
        array = np.asarray(descriptor_values, dtype=float)
        center = float(np.median(array))
        mad_scale = float(np.median(np.abs(array - center)) * 1.4826)
        q25, q75 = np.quantile(array, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        standard_scale = float(np.std(array))
        scale = next(
            (candidate for candidate in (mad_scale, iqr_scale, standard_scale) if candidate > 1e-9),
            0.0,
        )
        if scale <= 0.0:
            continue
        q10, q90 = np.quantile(array, [0.10, 0.90])
        robust_range = max(float(q90 - q10), 4.0 * scale, 1e-9)
        scalers[descriptor_id] = {
            "center": center,
            "scale": scale,
            "robust_range": robust_range,
            "support_count": len(descriptor_values),
        }
    return scalers


def _validated_reference_scalers(
    reference_scalers: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Validate externally frozen scalers before they enter a comparison."""

    scalers: dict[str, dict[str, float]] = {}
    for descriptor_id, source in reference_scalers.items():
        center = _finite(source.get("center"))
        scale = _finite(source.get("scale"))
        robust_range = _finite(source.get("robust_range"))
        if center is None or scale is None or robust_range is None:
            continue
        if scale <= 1e-9 or robust_range <= 1e-9:
            continue
        scalers[str(descriptor_id)] = {
            "center": center,
            "scale": scale,
            "robust_range": robust_range,
            "support_count": int(source.get("support_count") or 0),
        }
    return scalers


def _compare_pair(
    left_id: str,
    right_id: str,
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    descriptors: Mapping[str, Mapping[str, str]],
    scalers: Mapping[str, Mapping[str, float]],
    case_count: int,
) -> dict[str, Any]:
    left = vectors.get(left_id, {})
    right = vectors.get(right_id, {})
    shared = sorted(set(left).intersection(right).intersection(scalers))
    union = set(left).union(right).intersection(scalers)
    families = {descriptors[item]["family"] for item in shared}
    union_families = {descriptors[item]["family"] for item in union}
    if len(shared) < MINIMUM_SHARED_DESCRIPTORS or len(families) < MINIMUM_SHARED_FAMILIES:
        return {
            "available": False,
            "reason": (
                f"Only {len(shared)} shared supported measurements across "
                f"{len(families)} movement areas were available."
            ),
            "shared_descriptor_count": len(shared),
            "shared_family_count": len(families),
            "indices": {},
        }

    family_counts: dict[str, int] = defaultdict(int)
    for descriptor_id in shared:
        family_counts[descriptors[descriptor_id]["family"]] += 1
    raw_weights = []
    left_z = []
    right_z = []
    gower_gaps = []
    perspective_scores = []
    measurement_rows = []
    for descriptor_id in shared:
        left_item = left[descriptor_id]
        right_item = right[descriptor_id]
        descriptor = descriptors[descriptor_id]
        scaler = scalers[descriptor_id]
        pair_quality = math.sqrt(float(left_item["quality"]) * float(right_item["quality"]))
        raw_weight = pair_quality / max(1, family_counts[descriptor["family"]])
        left_value = float(left_item["value"])
        right_value = float(right_item["value"])
        left_scaled = _clip_value((left_value - scaler["center"]) / scaler["scale"], -5.0, 5.0)
        right_scaled = _clip_value((right_value - scaler["center"]) / scaler["scale"], -5.0, 5.0)
        gap = min(abs(left_value - right_value) / scaler["robust_range"], 1.0)
        perspective_score = _perspective_compatibility(
            str(left_item["perspective"]), str(right_item["perspective"])
        )
        raw_weights.append(raw_weight)
        left_z.append(left_scaled)
        right_z.append(right_scaled)
        gower_gaps.append(gap)
        perspective_scores.append(perspective_score)
        measurement_rows.append(
            {
                "descriptor_id": descriptor_id,
                "label": f'{descriptor["feature_label"]} — {descriptor["statistic_label"]}',
                "family": descriptor["family_label"],
                "left_value": round(left_value, 3),
                "right_value": round(right_value, 3),
                "absolute_difference": round(abs(left_value - right_value), 3),
                "unit": str(descriptor.get("unit") or ""),
                "relative_gap": round(gap * 100.0, 1),
                "pair_quality": pair_quality,
            }
        )

    weights = np.asarray(raw_weights, dtype=float)
    if float(weights.sum()) <= 1e-12:
        return {
            "available": False,
            "reason": "The shared measurements did not have enough evidence support.",
            "shared_descriptor_count": len(shared),
            "shared_family_count": len(families),
            "indices": {},
        }
    weights /= float(weights.sum())
    left_array = np.asarray(left_z, dtype=float)
    right_array = np.asarray(right_z, dtype=float)
    gaps = np.asarray(gower_gaps, dtype=float)

    l1_distance = float(np.sum(weights * gaps))
    euclidean_distance = float(np.sqrt(np.sum(weights * np.square(left_array - right_array))))
    cosine = _weighted_cosine(left_array, right_array, weights)
    soft_cosine = _soft_cosine(left_array, right_array, weights, shared, descriptors)
    indices = {
        "overall_movement_difference": round(1.0 - l1_distance, 3),
        "large_difference_focus": round(1.0 / (1.0 + euclidean_distance), 3),
    }
    if cosine is not None:
        indices["movement_pattern_direction"] = round(0.5 * (cosine + 1.0), 3)
    if soft_cosine is not None:
        indices["relationship_aware_pattern"] = round(0.5 * (soft_cosine + 1.0), 3)

    overlap = len(shared) / max(1, len(union))
    evidence = float(np.average([row["pair_quality"] for row in measurement_rows], weights=weights))
    family_coverage = len(families) / max(1, len(union_families))
    view_compatibility = float(np.average(perspective_scores, weights=weights))
    library_support = min(max(case_count - 1, 0) / 15.0, 1.0)
    closest = sorted(measurement_rows, key=lambda row: (row["relative_gap"], row["label"]))[:3]
    different = sorted(
        measurement_rows,
        key=lambda row: (row["relative_gap"] * row["pair_quality"], row["label"]),
        reverse=True,
    )[:3]
    return {
        "available": bool(indices),
        "reason": "",
        "shared_descriptor_count": len(shared),
        "shared_family_count": len(families),
        "indices": indices,
        "base_evidence_factors": {
            "shared_information": overlap,
            "measurement_quality": evidence,
            "body_coverage": family_coverage,
            "camera_view_match": view_compatibility,
            "library_support": library_support,
        },
        "closest_measurements": closest,
        "largest_differences": different,
    }


def _compare_best_view_pair(
    left_case_id: str,
    right_case_id: str,
    view_vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_metadata: Mapping[str, Mapping[str, Any]],
    case_view_ids: Mapping[str, list[str]],
    descriptors: Mapping[str, Mapping[str, str]],
    scalers: Mapping[str, Mapping[str, float]],
    case_count: int,
    reference_ids: set[str],
) -> dict[str, Any]:
    """Retain one coherent, most-similar eligible view pair for two injuries."""

    def comparison_views(case_id: str) -> list[str]:
        view_ids = sorted(case_view_ids.get(case_id, []))
        if case_id not in reference_ids:
            return view_ids
        return [
            view_id
            for view_id in view_ids
            if bool(view_metadata.get(view_id, {}).get("reference_view_eligible", True))
        ]

    left_views = comparison_views(left_case_id)
    right_views = comparison_views(right_case_id)
    eligible_pair_count = len(left_views) * len(right_views)
    comparisons: list[tuple[str, str, dict[str, Any]]] = []
    for left_view_id in left_views:
        for right_view_id in right_views:
            comparison = _compare_pair(
                left_view_id,
                right_view_id,
                view_vectors,
                descriptors,
                scalers,
                case_count,
            )
            comparisons.append((left_view_id, right_view_id, comparison))

    comparable = [item for item in comparisons if item[2].get("available")]
    if comparable:
        left_view_id, right_view_id, best = min(
            comparable,
            key=lambda item: (
                -float(
                    item[2].get("indices", {}).get(DEFAULT_SIMILARITY_LENS, -1.0)
                ),
                -int(item[2].get("shared_descriptor_count") or 0),
                item[0],
                item[1],
            ),
        )
    elif comparisons:
        left_view_id, right_view_id, best = min(
            comparisons,
            key=lambda item: (
                -int(item[2].get("shared_descriptor_count") or 0),
                -int(item[2].get("shared_family_count") or 0),
                item[0],
                item[1],
            ),
        )
    else:
        return {
            "available": False,
            "reason": "Neither injury has an eligible completed view for event comparison.",
            "shared_descriptor_count": 0,
            "shared_family_count": 0,
            "indices": {},
            "eligible_view_pair_count": eligible_pair_count,
            "comparable_view_pair_count": 0,
            "selected_views": {},
            "view_selection_rule": "highest_primary_lens_similarity",
        }

    output = dict(best)
    output.update(
        {
            "eligible_view_pair_count": eligible_pair_count,
            "comparable_view_pair_count": len(comparable),
            "selected_views": {
                left_case_id: dict(view_metadata.get(left_view_id, {})),
                right_case_id: dict(view_metadata.get(right_view_id, {})),
            },
            "view_selection_rule": "highest_primary_lens_similarity",
        }
    )
    for field in ("closest_measurements", "largest_differences"):
        rows = []
        for row in output.get(field, []):
            item = dict(row)
            item["case_values"] = {
                left_case_id: item.pop("left_value", None),
                right_case_id: item.pop("right_value", None),
            }
            rows.append(item)
        output[field] = rows
    return output


def _rankings_for_case(
    selected_id: str,
    cases: Mapping[str, Mapping[str, Any]],
    reference_ids: set[str],
    pair_scores: Mapping[tuple[str, str], Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_metadata: Mapping[str, Mapping[str, Any]],
    case_view_ids: Mapping[str, list[str]],
    descriptors: Mapping[str, Mapping[str, str]],
    scalers: Mapping[str, Mapping[str, float]],
    *,
    result_limit: int,
    resampling_iterations: int,
    cancelled: Callable[[], bool] | None,
) -> dict[str, list[dict[str, Any]]]:
    lens_rankings: dict[str, list[dict[str, Any]]] = {}
    raw_rankings: dict[str, list[tuple[str, Mapping[str, Any], float]]] = {}
    candidate_ids = sorted(reference_ids.difference({selected_id}))
    resampling = _resampled_top_rank_frequency(
        selected_id,
        candidate_ids,
        view_vectors,
        view_metadata,
        case_view_ids,
        reference_ids,
        descriptors,
        scalers,
        iterations=resampling_iterations,
        cancelled=cancelled,
    )
    for lens in SIMILARITY_LENSES:
        lens_id = lens["id"]
        candidates = []
        for candidate_id in candidate_ids:
            pair = pair_scores.get(_pair_key(selected_id, candidate_id), {})
            index = pair.get("indices", {}).get(lens_id)
            if pair.get("available") and index is not None:
                candidates.append((candidate_id, pair, float(index)))
        candidates.sort(key=lambda item: (-item[2], str(cases[item[0]]["player_name"])))
        raw_rankings[lens_id] = candidates

    for lens in SIMILARITY_LENSES:
        lens_id = lens["id"]
        matches = []
        for rank, (candidate_id, pair, index) in enumerate(raw_rankings[lens_id], start=1):
            evidence_support = _evidence_support(
                pair["base_evidence_factors"],
                lens_id=lens_id,
                case_count=len(reference_ids),
                shared_count=int(pair["shared_descriptor_count"]),
                shared_family_count=int(pair["shared_family_count"]),
            )
            matches.append(
                {
                    "rank": rank,
                    "case": dict(cases[candidate_id]),
                    "similarity_index": index,
                    "evidence_support": evidence_support,
                    "stability": _stability_summary(
                        resampling.get(lens_id, {}), candidate_id
                    ),
                    "shared_descriptor_count": pair["shared_descriptor_count"],
                    "shared_family_count": pair["shared_family_count"],
                    "selected_view_pair": {
                        "selected_case": dict(
                            pair.get("selected_views", {}).get(selected_id, {})
                        ),
                        "candidate_case": dict(
                            pair.get("selected_views", {}).get(candidate_id, {})
                        ),
                        "eligible_view_pair_count": int(
                            pair.get("eligible_view_pair_count") or 0
                        ),
                        "comparable_view_pair_count": int(
                            pair.get("comparable_view_pair_count") or 0
                        ),
                        "selection_rule": str(pair.get("view_selection_rule") or ""),
                    },
                    "closest_measurements": _orient_measurements(
                        pair["closest_measurements"], selected_id, candidate_id
                    ),
                    "largest_differences": _orient_measurements(
                        pair["largest_differences"], selected_id, candidate_id
                    ),
                }
            )
        lens_rankings[lens_id] = matches[:result_limit]
    return lens_rankings


def _orient_measurements(
    rows: Iterable[Mapping[str, Any]], selected_id: str, candidate_id: str
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        values = item.pop("case_values", {})
        item["selected_value"] = values.get(selected_id)
        item["candidate_value"] = values.get(candidate_id)
        output.append(item)
    return output


def _evidence_support(
    base_factors: Mapping[str, float],
    *,
    lens_id: str,
    case_count: int,
    shared_count: int,
    shared_family_count: int,
) -> dict[str, Any]:
    factors = {key: _clip(float(value)) for key, value in base_factors.items()}
    strong = all(
        factors[key] >= threshold
        for key, threshold in {
            "shared_information": 0.80,
            "measurement_quality": 0.80,
            "body_coverage": 0.75,
            "camera_view_match": 0.80,
            "library_support": 0.80,
        }.items()
    )
    adequate = all(
        factors[key] >= threshold
        for key, threshold in {
            "shared_information": 0.60,
            "measurement_quality": 0.60,
            "body_coverage": 0.50,
            "camera_view_match": 0.60,
        }.items()
    )
    if lens_id == "relationship_aware_pattern" or case_count < 20:
        status = "PROVISIONAL"
        label = "Provisional"
    elif strong:
        status = "HIGH"
        label = "High"
    elif adequate:
        status = "MODERATE"
        label = "Moderate"
    else:
        status = "LOW"
        label = "Low"

    limitations = []
    if factors["library_support"] < 0.80:
        limitations.append("the reference library is still small")
    if factors["measurement_quality"] < 0.65:
        limitations.append("some shared measurements have limited support")
    if factors["camera_view_match"] < 0.75:
        limitations.append("the camera views are not closely matched")
    if factors["shared_information"] < 0.70:
        limitations.append("some measurements do not overlap")
    if lens_id == "relationship_aware_pattern":
        limitations.insert(0, "the soft-cosine relationship map still needs validation")
    limitation_text = "; ".join(limitations[:2])
    explanation = (
        f"{shared_count} shared measurements across {shared_family_count} movement areas "
        f"support this result"
    )
    explanation += f", but {limitation_text}." if limitation_text else "."
    return {
        "status": status,
        "label": label,
        "explanation": explanation,
        "factor_levels": {key: _support_level(value) for key, value in factors.items()},
    }


def _resampled_top_rank_frequency(
    selected_id: str,
    candidate_ids: list[str],
    view_vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
    view_metadata: Mapping[str, Mapping[str, Any]],
    case_view_ids: Mapping[str, list[str]],
    reference_ids: set[str],
    descriptors: Mapping[str, Mapping[str, str]],
    scalers: Mapping[str, Mapping[str, float]],
    *,
    iterations: int = RESAMPLING_ITERATIONS,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure top-rank sensitivity under deterministic 10% feature dropout."""

    output = {
        lens["id"]: {
            "iterations": iterations,
            "valid_iterations": 0,
            "top_counts": {candidate_id: 0 for candidate_id in candidate_ids},
        }
        for lens in SIMILARITY_LENSES
    }
    feature_names = sorted(
        {
            descriptors[descriptor_id]["feature_name"]
            for descriptor_id in scalers
            if descriptor_id in descriptors
        }
    )
    if not selected_id or not candidate_ids or not feature_names:
        return output
    keep_count = max(1, round(len(feature_names) * (1.0 - RESAMPLING_FEATURE_DROPOUT)))
    seed_bytes = hashlib.sha256(
        f"{SIMILARITY_ENGINE_VERSION}:{selected_id}".encode()
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    for _ in range(iterations):
        if cancelled is not None and cancelled():
            raise SimilarityComputationCancelled("Comparison superseded by a newer request.")
        kept = set(rng.choice(feature_names, size=keep_count, replace=False).tolist())
        sampled_scalers = {
            descriptor_id: scaler
            for descriptor_id, scaler in scalers.items()
            if descriptor_id in descriptors
            and descriptors[descriptor_id]["feature_name"] in kept
        }
        per_lens: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for candidate_id in candidate_ids:
            comparison = _compare_best_view_pair(
                selected_id,
                candidate_id,
                view_vectors,
                view_metadata,
                case_view_ids,
                descriptors,
                sampled_scalers,
                len(candidate_ids) + 1,
                reference_ids,
            )
            if not comparison["available"]:
                continue
            for lens_id, index in comparison["indices"].items():
                per_lens[lens_id].append((candidate_id, float(index)))
        for lens_id, candidates in per_lens.items():
            if not candidates:
                continue
            winner = min(candidates, key=lambda item: (-item[1], item[0]))[0]
            output[lens_id]["valid_iterations"] += 1
            output[lens_id]["top_counts"][winner] += 1
    return output


def _stability_summary(result: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    iterations = int(result.get("iterations", RESAMPLING_ITERATIONS))
    valid = int(result.get("valid_iterations") or 0)
    count = int(result.get("top_counts", {}).get(candidate_id, 0))
    frequency = count / valid if valid else None
    if frequency is None:
        label = "Unavailable"
    elif frequency == 0.0:
        label = "Not observed as top rank"
    elif frequency >= 0.80:
        label = "Stable top rank"
    elif frequency >= 0.60:
        label = "Moderately stable top rank"
    else:
        label = "Top rank is sensitive"
    return {
        "method": "10% feature-dropout resampling",
        "iterations": iterations,
        "valid_iterations": valid,
        "top_rank_count": count,
        "top_rank_frequency": round(frequency, 3) if frequency is not None else None,
        "label": label,
        "explanation": (
            f"Ranked first in {count} of {valid} valid feature-dropout checks."
            if valid
            else "No valid feature-dropout checks were available."
        ),
    }


def _support_level(value: float) -> str:
    if value >= 0.80:
        return "strong"
    if value >= 0.60:
        return "moderate"
    return "limited"


def _public_cases(
    event_lookup: Mapping[str, Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    cases = []
    for case_id, vector in vectors.items():
        event = event_lookup.get(case_id, {})
        player_name = _display_player_name(event.get("player_name"), case_id)
        if player_name == "Player not recorded":
            continue
        cases.append(
            {
                "case_id": case_id,
                "player_name": player_name,
                "team": str(event.get("team") or ""),
                "competition": str(event.get("competition") or ""),
                "injury_date": str(event.get("injury_date") or ""),
                "position_group": str(event.get("position_group") or "unknown"),
                "analysed_view_count": int(event.get("analysed_view_count") or 0),
                "comparable_descriptor_count": len(vector),
                "reference_pool_eligible": bool(event.get("reference_pool_eligible", False)),
                "reference_pool_reason": str(
                    event.get("reference_pool_reason")
                    or (
                        "Phase-support evidence was not supplied; this case is query-only and "
                        "may be compared with eligible references when enough measurements overlap."
                    )
                ),
                "phase_supported_view_count": int(
                    event.get("phase_supported_view_count") or 0
                ),
                "event_covered_view_count": int(
                    event.get("event_covered_view_count") or 0
                ),
                "event_excluded_view_count": int(
                    event.get("event_excluded_view_count") or 0
                ),
            }
        )
    return sorted(cases, key=lambda item: (item["player_name"], item["case_id"]))


def _weighted_cosine(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float | None:
    numerator = float(np.sum(weights * left * right))
    denominator = math.sqrt(
        float(np.sum(weights * np.square(left)))
        * float(np.sum(weights * np.square(right)))
    )
    if denominator <= 1e-12:
        return None
    return _clip_value(numerator / denominator, -1.0, 1.0)


def _soft_cosine(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    descriptor_ids: list[str],
    descriptors: Mapping[str, Mapping[str, str]],
) -> float | None:
    size = len(descriptor_ids)
    relationship = np.eye(size, dtype=float)
    for left_index, left_id in enumerate(descriptor_ids):
        for right_index in range(left_index, size):
            right_id = descriptor_ids[right_index]
            value = 0.0
            if descriptors[left_id]["feature_name"] == descriptors[right_id]["feature_name"]:
                value += 0.25
            if descriptors[left_id]["family"] == descriptors[right_id]["family"]:
                value += 0.08
            relationship[left_index, right_index] += value
            if left_index != right_index:
                relationship[right_index, left_index] += value
    diagonal = np.sqrt(np.diag(relationship))
    relationship = relationship / np.outer(diagonal, diagonal)
    weighted_left = np.sqrt(weights) * left
    weighted_right = np.sqrt(weights) * right
    numerator = float(weighted_left @ relationship @ weighted_right)
    left_norm = float(weighted_left @ relationship @ weighted_left)
    right_norm = float(weighted_right @ relationship @ weighted_right)
    denominator = math.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0))
    if denominator <= 1e-12:
        return None
    return _clip_value(numerator / denominator, -1.0, 1.0)


def _perspective_compatibility(left: str, right: str) -> float:
    left_value = left.strip().lower()
    right_value = right.strip().lower()
    unknown = {"", "unknown", "none", "not recorded"}
    if left_value in unknown or right_value in unknown:
        return 0.75
    return 1.0 if left_value == right_value else 0.60


def _display_player_name(value: object, case_id: str) -> str:
    source = str(value or "").strip()
    raw = source.replace("_", " ").strip()
    lowered = raw.lower()
    if (
        not raw
        or _UUID_PATTERN.search(source)
        or lowered.startswith(("screen recording", "imported "))
    ):
        return "Player not recorded"
    if raw == case_id.replace("_", " ").title():
        return "Player not recorded"
    return raw


def _feature_label(feature_name: str) -> str:
    tokens = feature_name.split("_")
    is_projected_2d = "2d" in tokens
    tokens = [token for token in tokens if token not in {"2d", "deg"}]
    replacements = {
        "hka": "HKA",
        "injured": "injured-side",
        "contralateral": "opposite-side",
    }
    label = " ".join(replacements.get(token, token) for token in tokens)
    if is_projected_2d:
        label = f"{label} (2D)"
    return label[:1].upper() + label[1:]


def _family_label(value: str) -> str:
    return value.replace("_", " ").replace("/", " / ").strip().title() or "Other"


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _clip(value: float) -> float:
    return _clip_value(value, 0.0, 1.0)


def _clip_value(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
