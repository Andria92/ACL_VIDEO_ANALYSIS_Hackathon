"""Internal, non-clinical validation audits for the similarity engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from acl_motion.analytics.similarity import (
    DEFAULT_SIMILARITY_LENS,
    SIMILARITY_LENSES,
    build_similarity_payload,
)

SIMILARITY_VALIDATION_VERSION = "similarity_internal_validation_v1"


def build_internal_similarity_validation_report(
    records: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit ranking sensitivity without claiming external or clinical validation."""

    record_list = [dict(record) for record in records]
    event_list = [dict(event) for event in events]
    index = build_similarity_payload(
        record_list,
        event_list,
        resampling_iterations=0,
    )
    case_ids = [str(case["case_id"]) for case in index["cases"]]
    reference_ids = {
        str(case["case_id"])
        for case in index["cases"]
        if bool(case["reference_pool_eligible"])
    }
    audits = [
        _audit_query(
            query_id,
            record_list,
            event_list,
            reference_ids,
        )
        for query_id in case_ids
    ]
    primary_rows = [
        audit["lenses"][DEFAULT_SIMILARITY_LENS]
        for audit in audits
        if audit["lenses"][DEFAULT_SIMILARITY_LENS]["baseline_top_case_id"]
    ]
    jackknife_frequencies = [
        float(row["jackknife_top_retention_frequency"])
        for row in primary_rows
        if row["jackknife_top_retention_frequency"] is not None
    ]
    lens_agreements = [
        float(audit["cross_lens_top_agreement"])
        for audit in audits
        if audit["cross_lens_top_agreement"] is not None
    ]
    return {
        "validation_version": SIMILARITY_VALIDATION_VERSION,
        "status": "INTERNAL_AUDIT_ONLY" if audits else "INSUFFICIENT_CASES",
        "externally_validated": False,
        "held_out_players": False,
        "summary": {
            "audited_query_count": len(audits),
            "reference_pool_case_count": len(reference_ids),
            "primary_jackknife_evaluable_query_count": len(jackknife_frequencies),
            "median_primary_top_retention": _rounded_median(jackknife_frequencies),
            "median_cross_lens_top_agreement": _rounded_median(lens_agreements),
            "query_excluded_scaling": True,
        },
        "case_audits": audits,
        "interpretation": (
            "This report checks whether rankings change when individual reference cases are "
            "removed and whether different lenses choose the same top case. It does not show "
            "agreement with experts, laboratory measurements, injury mechanisms, or future cases. "
            "A query is omitted from the jackknife summary when removing another reference leaves "
            "fewer than three cases able to define robust feature scales."
        ),
        "remaining_requirements": [
            "independent repeated annotations",
            "laboratory or multi-camera pose reference",
            "blinded expert pairwise judgements",
            "evaluation on genuinely new held-out players",
            "frozen scalers from a separate larger cohort",
        ],
    }


def _audit_query(
    query_id: str,
    records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    reference_ids: set[str],
) -> dict[str, Any]:
    baseline = build_similarity_payload(
        records,
        events,
        selected_case_id=query_id,
        resampling_iterations=0,
    )
    reduced_payloads = {}
    for excluded_id in sorted(reference_ids.difference({query_id})):
        reduced_events = [
            {
                **event,
                "reference_pool_eligible": False,
                "reference_pool_reason": "Temporarily removed for reference-case jackknife audit.",
            }
            if str(event.get("case_id", "")) == excluded_id
            else dict(event)
            for event in events
        ]
        reduced_payloads[excluded_id] = build_similarity_payload(
            records,
            reduced_events,
            selected_case_id=query_id,
            resampling_iterations=0,
        )

    lens_rows = {}
    baseline_winners = []
    for lens in SIMILARITY_LENSES:
        lens_id = str(lens["id"])
        baseline_matches = baseline["rankings"].get(lens_id, [])
        baseline_winner = (
            str(baseline_matches[0]["case"]["case_id"])
            if baseline_matches
            else ""
        )
        if baseline_winner:
            baseline_winners.append(baseline_winner)
        retained = 0
        eligible_checks = 0
        alternate_winners: Counter[str] = Counter()
        for excluded_id, payload in reduced_payloads.items():
            if excluded_id == baseline_winner:
                continue
            matches = payload["rankings"].get(lens_id, [])
            if not matches:
                continue
            eligible_checks += 1
            winner = str(matches[0]["case"]["case_id"])
            alternate_winners[winner] += 1
            retained += int(winner == baseline_winner)
        lens_rows[lens_id] = {
            "label": str(lens["label"]),
            "baseline_top_case_id": baseline_winner or None,
            "baseline_top_index": (
                float(baseline_matches[0]["similarity_index"])
                if baseline_matches
                else None
            ),
            "jackknife_valid_checks": eligible_checks,
            "jackknife_top_retention_count": retained,
            "jackknife_top_retention_frequency": (
                round(retained / eligible_checks, 3) if eligible_checks else None
            ),
            "jackknife_winner_counts": dict(sorted(alternate_winners.items())),
        }

    winner_counts = Counter(baseline_winners)
    cross_lens_agreement = (
        max(winner_counts.values()) / len(baseline_winners)
        if baseline_winners
        else None
    )
    return {
        "query_case_id": query_id,
        "query_reference_pool_eligible": bool(
            baseline.get("selected_case", {}).get("reference_pool_eligible", False)
        ),
        "scaling_status": str(baseline.get("scaling", {}).get("status", "")),
        "scaler_reference_case_count": baseline.get("scaling", {}).get(
            "reference_case_count"
        ),
        "cross_lens_top_agreement": (
            round(cross_lens_agreement, 3)
            if cross_lens_agreement is not None
            else None
        ),
        "cross_lens_winner_counts": dict(sorted(winner_counts.items())),
        "lenses": lens_rows,
    }


def _rounded_median(values: list[float]) -> float | None:
    return round(float(np.median(values)), 3) if values else None
