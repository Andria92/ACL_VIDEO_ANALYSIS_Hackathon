"""Blinded pairwise expert judgements and similarity-ranking concordance."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

from acl_motion.analytics.similarity import SIMILARITY_LENSES, build_similarity_payload

EXPERT_JUDGEMENT_PROTOCOL_VERSION = "expert_pairwise_similarity_v1"


class PairwiseChoice(StrEnum):
    """Permitted responses for one blinded A/B movement comparison."""

    OPTION_A = "OPTION_A"
    OPTION_B = "OPTION_B"
    ABOUT_THE_SAME = "ABOUT_THE_SAME"
    UNABLE_TO_JUDGE = "UNABLE_TO_JUDGE"


@dataclass(frozen=True, slots=True)
class ExpertPairwiseJudgement:
    """One assessor response, blinded to algorithm output and case metadata."""

    judgement_id: str
    assignment_id: str
    assessor_id: str
    query_case_id: str
    option_a_case_id: str
    option_b_case_id: str
    choice: PairwiseChoice
    created_at: str
    notes: str = ""
    protocol_version: str = EXPERT_JUDGEMENT_PROTOCOL_VERSION
    blinded_to_algorithm: bool = True

    def __post_init__(self) -> None:
        identifiers = {
            self.query_case_id,
            self.option_a_case_id,
            self.option_b_case_id,
        }
        if len(identifiers) != 3:
            raise ValueError("Query, Option A, and Option B must be three different cases.")
        if not self.assessor_id.strip():
            raise ValueError("assessor_id is required.")
        object.__setattr__(self, "choice", PairwiseChoice(self.choice))

    @classmethod
    def create(
        cls,
        *,
        assignment: Mapping[str, Any],
        assessor_id: str,
        choice: PairwiseChoice | str,
        notes: str = "",
    ) -> ExpertPairwiseJudgement:
        return cls(
            judgement_id=str(uuid4()),
            assignment_id=str(assignment["assignment_id"]),
            assessor_id=assessor_id.strip(),
            query_case_id=str(assignment["query_case_id"]),
            option_a_case_id=str(assignment["option_a_case_id"]),
            option_b_case_id=str(assignment["option_b_case_id"]),
            choice=PairwiseChoice(choice),
            created_at=datetime.now(UTC).isoformat(),
            notes=notes.strip(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExpertPairwiseJudgement:
        return cls(
            judgement_id=str(data["judgement_id"]),
            assignment_id=str(data["assignment_id"]),
            assessor_id=str(data["assessor_id"]),
            query_case_id=str(data["query_case_id"]),
            option_a_case_id=str(data["option_a_case_id"]),
            option_b_case_id=str(data["option_b_case_id"]),
            choice=PairwiseChoice(str(data["choice"])),
            created_at=str(data["created_at"]),
            notes=str(data.get("notes", "")),
            protocol_version=str(
                data.get("protocol_version", EXPERT_JUDGEMENT_PROTOCOL_VERSION)
            ),
            blinded_to_algorithm=bool(data.get("blinded_to_algorithm", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgement_id": self.judgement_id,
            "assignment_id": self.assignment_id,
            "assessor_id": self.assessor_id,
            "query_case_id": self.query_case_id,
            "option_a_case_id": self.option_a_case_id,
            "option_b_case_id": self.option_b_case_id,
            "choice": self.choice.value,
            "created_at": self.created_at,
            "notes": self.notes,
            "protocol_version": self.protocol_version,
            "blinded_to_algorithm": self.blinded_to_algorithm,
        }


def build_blinded_assignments(
    case_sources: Iterable[Mapping[str, Any]],
    reference_case_ids: set[str],
    *,
    assessor_id: str,
) -> list[dict[str, Any]]:
    """Create deterministic, assessor-specific A/B ordering without algorithm scores."""

    assessor = assessor_id.strip()
    if not assessor:
        raise ValueError("assessor_id is required.")
    source_lookup = {
        str(source["case_id"]): dict(source)
        for source in case_sources
        if source.get("case_id") and source.get("slug")
    }
    assignments = []
    for query_id in sorted(source_lookup):
        candidates = sorted(reference_case_ids.intersection(source_lookup).difference({query_id}))
        for first_id, second_id in combinations(candidates, 2):
            assignment_id = _assignment_id(query_id, first_id, second_id)
            orientation = _digest_int(f"{assessor}:{assignment_id}:orientation") % 2
            option_a, option_b = (
                (first_id, second_id) if orientation == 0 else (second_id, first_id)
            )
            assignments.append(
                {
                    "assignment_id": assignment_id,
                    "query_case_id": query_id,
                    "query_slug": str(source_lookup[query_id]["slug"]),
                    "option_a_case_id": option_a,
                    "option_a_slug": str(source_lookup[option_a]["slug"]),
                    "option_b_case_id": option_b,
                    "option_b_slug": str(source_lookup[option_b]["slug"]),
                }
            )
    return sorted(
        assignments,
        key=lambda item: _digest_int(
            f"{assessor}:{item['assignment_id']}:presentation-order"
        ),
    )


def next_blinded_assignment(
    assignments: Iterable[Mapping[str, Any]],
    judgements: Iterable[ExpertPairwiseJudgement],
    *,
    assessor_id: str,
) -> dict[str, Any] | None:
    completed = {
        judgement.assignment_id
        for judgement in judgements
        if judgement.assessor_id == assessor_id
    }
    return next(
        (dict(assignment) for assignment in assignments if assignment["assignment_id"] not in completed),
        None,
    )


def load_expert_judgements(path: str | Path) -> tuple[ExpertPairwiseJudgement, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    judgements = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            judgements.append(ExpertPairwiseJudgement.from_dict(json.loads(line)))
    return tuple(judgements)


def append_expert_judgement(
    path: str | Path,
    judgement: ExpertPairwiseJudgement,
) -> Path:
    output = Path(path)
    existing = load_expert_judgements(output)
    if any(
        item.assessor_id == judgement.assessor_id
        and item.assignment_id == judgement.assignment_id
        for item in existing
    ):
        raise ValueError("This assessor has already completed that blinded assignment.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgement.to_dict(), sort_keys=True) + "\n")
    return output


def evaluate_expert_judgements(
    records: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    judgements: Iterable[ExpertPairwiseJudgement],
) -> dict[str, Any]:
    """Compare algorithm pair ordering with judgements without training on them."""

    record_list = [dict(record) for record in records]
    event_list = [dict(event) for event in events]
    judgement_list = list(judgements)
    scored_choices = [
        item
        for item in judgement_list
        if item.choice in {PairwiseChoice.OPTION_A, PairwiseChoice.OPTION_B}
    ]
    payload_cache = {}
    lens_counts = {
        str(lens["id"]): {"correct": 0, "evaluated": 0, "unavailable": 0}
        for lens in SIMILARITY_LENSES
    }
    for judgement in scored_choices:
        if judgement.query_case_id not in payload_cache:
            payload_cache[judgement.query_case_id] = build_similarity_payload(
                record_list,
                event_list,
                selected_case_id=judgement.query_case_id,
                result_limit=max(len(event_list), 1),
                resampling_iterations=0,
            )
        payload = payload_cache[judgement.query_case_id]
        expert_winner = (
            judgement.option_a_case_id
            if judgement.choice is PairwiseChoice.OPTION_A
            else judgement.option_b_case_id
        )
        for lens in SIMILARITY_LENSES:
            lens_id = str(lens["id"])
            scores = {
                str(match["case"]["case_id"]): float(match["similarity_index"])
                for match in payload["rankings"].get(lens_id, [])
            }
            if (
                judgement.option_a_case_id not in scores
                or judgement.option_b_case_id not in scores
            ):
                lens_counts[lens_id]["unavailable"] += 1
                continue
            algorithm_winner = max(
                (judgement.option_a_case_id, judgement.option_b_case_id),
                key=lambda case_id: (scores[case_id], case_id),
            )
            lens_counts[lens_id]["evaluated"] += 1
            lens_counts[lens_id]["correct"] += int(algorithm_winner == expert_winner)

    lenses = {}
    for lens in SIMILARITY_LENSES:
        lens_id = str(lens["id"])
        counts = lens_counts[lens_id]
        evaluated = counts["evaluated"]
        accuracy = counts["correct"] / evaluated if evaluated else None
        lenses[lens_id] = {
            "label": str(lens["label"]),
            **counts,
            "concordance": round(accuracy, 3) if accuracy is not None else None,
            "wilson_95_interval": (
                [round(value, 3) for value in _wilson_interval(counts["correct"], evaluated)]
                if evaluated
                else None
            ),
        }
    return {
        "protocol_version": EXPERT_JUDGEMENT_PROTOCOL_VERSION,
        "status": "CURRENT_CASE_CONCORDANCE" if scored_choices else "NO_JUDGEMENTS",
        "judgement_count": len(judgement_list),
        "scored_for_concordance_count": len(scored_choices),
        "assessor_count": len({item.assessor_id for item in judgement_list}),
        "query_case_count": len({item.query_case_id for item in judgement_list}),
        "lenses": lenses,
        "query_excluded_scaling": True,
        "held_out_players": False,
        "interpretation": (
            "Concordance compares the engine with blinded expert choices for current cases. "
            "The judgements are not used to fit the engine. Genuinely new players are still "
            "required for final held-out evaluation."
        ),
        "assessor_agreement": _assessor_agreement(judgement_list),
    }


def _assessor_agreement(
    judgements: list[ExpertPairwiseJudgement],
) -> dict[str, Any]:
    grouped: dict[str, list[ExpertPairwiseJudgement]] = defaultdict(list)
    for item in judgements:
        grouped[item.assignment_id].append(item)
    agreeing_pairs = 0
    compared_pairs = 0
    for items in grouped.values():
        for left, right in combinations(items, 2):
            if left.assessor_id == right.assessor_id:
                continue
            compared_pairs += 1
            agreeing_pairs += int(
                _canonical_choice(left) == _canonical_choice(right)
            )
    return {
        "status": "AVAILABLE" if compared_pairs else "INSUFFICIENT_REPEAT_RATINGS",
        "assessor_pair_comparisons": compared_pairs,
        "exact_agreement_count": agreeing_pairs,
        "exact_agreement": (
            round(agreeing_pairs / compared_pairs, 3) if compared_pairs else None
        ),
        "note": (
            "This is raw exact agreement; chance-corrected agreement should be added after "
            "enough assessors and repeated assignments are available."
        ),
    }


def _canonical_choice(judgement: ExpertPairwiseJudgement) -> str:
    if judgement.choice is PairwiseChoice.OPTION_A:
        return judgement.option_a_case_id
    if judgement.choice is PairwiseChoice.OPTION_B:
        return judgement.option_b_case_id
    return judgement.choice.value


def _assignment_id(query_id: str, first_id: str, second_id: str) -> str:
    canonical = ":".join(
        [EXPERT_JUDGEMENT_PROTOCOL_VERSION, query_id, *sorted((first_id, second_id))]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _digest_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
