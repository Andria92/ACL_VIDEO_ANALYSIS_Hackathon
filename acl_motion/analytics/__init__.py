"""Analytics guardrails for ACL Movement Analytics Lab."""

from acl_motion.analytics.exploration import (
    CONTACT_MECHANISMS,
    EXPLORATION_VERSION,
    SUMMARY_STATISTICS,
    assess_group_test_eligibility,
    load_exploration_payload,
)
from acl_motion.analytics.guard import (
    CROSS_CASE_ANALYTIC_NAMES,
    CrossCaseAnalyticsUnavailable,
    require_cross_case_analytics_ready,
)
from acl_motion.analytics.similarity import (
    DEFAULT_SIMILARITY_LENS,
    SIMILARITY_ENGINE_VERSION,
    SIMILARITY_LENSES,
    build_similarity_payload,
    similarity_readiness,
)

__all__ = [
    "CONTACT_MECHANISMS",
    "CROSS_CASE_ANALYTIC_NAMES",
    "DEFAULT_SIMILARITY_LENS",
    "EXPLORATION_VERSION",
    "SIMILARITY_ENGINE_VERSION",
    "SIMILARITY_LENSES",
    "SUMMARY_STATISTICS",
    "CrossCaseAnalyticsUnavailable",
    "assess_group_test_eligibility",
    "build_similarity_payload",
    "load_exploration_payload",
    "require_cross_case_analytics_ready",
    "similarity_readiness",
]
