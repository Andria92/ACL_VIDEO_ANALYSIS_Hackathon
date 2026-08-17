"""Guardrails for deferred cross-case analytics."""

from __future__ import annotations

CROSS_CASE_ANALYTIC_NAMES = (
    "similarity",
    "umap",
    "clustering",
    "association_rules",
)


class CrossCaseAnalyticsUnavailable(RuntimeError):
    """Raised when cross-case analytics are requested before enough human cases exist."""


def require_cross_case_analytics_ready(
    *,
    human_validated_case_count: int,
    analytic_name: str,
) -> None:
    """Refuse V1 cross-case analytics until at least two human cases are available."""

    if analytic_name not in CROSS_CASE_ANALYTIC_NAMES:
        allowed = ", ".join(CROSS_CASE_ANALYTIC_NAMES)
        raise ValueError(f"Unknown cross-case analytic '{analytic_name}'. Expected one of: {allowed}")
    if human_validated_case_count < 2:
        raise CrossCaseAnalyticsUnavailable(
            f"{analytic_name} requires at least two human-validated cases; "
            f"received {human_validated_case_count}."
        )
