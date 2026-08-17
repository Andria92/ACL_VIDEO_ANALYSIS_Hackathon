"""Bilateral projected HKA relationship descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class WindowRelationshipSummary:
    """Bilateral relationship summary for a Movement-End-relative window."""

    window_name: str
    start_ms: float
    end_ms: float
    supported_samples: int
    evidence_status: str
    mean_absolute_difference_deg: float | None
    signed_start_deg: float | None
    signed_end_deg: float | None
    signed_change_deg: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BilateralHkaSummary:
    """Projected injured-vs-contralateral HKA relationship across time."""

    evidence_status: str
    supported_samples: int
    mean_absolute_hka_bilateral_difference_deg: float | None
    peak_absolute_hka_bilateral_difference_deg: float | None
    time_peak_absolute_hka_bilateral_difference_ms: float | None
    source_frame_peak_absolute_hka_bilateral_difference: int | None
    relationship_pattern: str
    pattern_explanation: str
    start_signed_difference_deg: float | None
    end_signed_difference_deg: float | None
    signed_change_deg: float | None
    window_summaries: tuple[WindowRelationshipSummary, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        output = asdict(self)
        output["window_summaries"] = [item.to_dict() for item in self.window_summaries]
        return output


def signed_difference(injured_value: float, contralateral_value: float) -> float:
    """Return injured minus contralateral projected HKA."""

    return injured_value - contralateral_value


def absolute_difference(injured_value: float, contralateral_value: float) -> float:
    """Return absolute projected HKA relationship."""

    return abs(signed_difference(injured_value, contralateral_value))


def compute_bilateral_hka_summary(
    dynamic_df: pd.DataFrame,
    *,
    minimum_supported_samples: int = 3,
    meaningful_change_deg: float = 10.0,
) -> BilateralHkaSummary:
    """Summarize supported projected bilateral HKA relationship samples."""

    signed_rows = _supported_feature(dynamic_df, "hka_projected_bilateral_difference_deg")
    absolute_rows = _supported_feature(
        dynamic_df,
        "hka_projected_bilateral_absolute_difference_deg",
    )
    if len(signed_rows) < minimum_supported_samples or len(absolute_rows) < minimum_supported_samples:
        return BilateralHkaSummary(
            evidence_status="UNAVAILABLE",
            supported_samples=int(min(len(signed_rows), len(absolute_rows))),
            mean_absolute_hka_bilateral_difference_deg=None,
            peak_absolute_hka_bilateral_difference_deg=None,
            time_peak_absolute_hka_bilateral_difference_ms=None,
            source_frame_peak_absolute_hka_bilateral_difference=None,
            relationship_pattern="INSUFFICIENT_EVIDENCE",
            pattern_explanation="Fewer than the configured supported bilateral HKA samples were available.",
            start_signed_difference_deg=None,
            end_signed_difference_deg=None,
            signed_change_deg=None,
            window_summaries=(),
        )

    absolute_rows = absolute_rows.sort_values("movement_end_relative_ms")
    signed_rows = signed_rows.sort_values("movement_end_relative_ms")
    peak = absolute_rows.loc[absolute_rows["feature_value"].abs().idxmax()]
    start_signed = float(signed_rows.iloc[0]["feature_value"])
    end_signed = float(signed_rows.iloc[-1]["feature_value"])
    signed_change = end_signed - start_signed
    pattern, explanation = _relationship_pattern(
        signed_rows,
        meaningful_change_deg=meaningful_change_deg,
    )
    return BilateralHkaSummary(
        evidence_status="SUPPORTED",
        supported_samples=int(min(len(signed_rows), len(absolute_rows))),
        mean_absolute_hka_bilateral_difference_deg=float(absolute_rows["feature_value"].abs().mean()),
        peak_absolute_hka_bilateral_difference_deg=float(abs(peak["feature_value"])),
        time_peak_absolute_hka_bilateral_difference_ms=float(peak["movement_end_relative_ms"]),
        source_frame_peak_absolute_hka_bilateral_difference=int(peak["source_frame_index"]),
        relationship_pattern=pattern,
        pattern_explanation=explanation,
        start_signed_difference_deg=start_signed,
        end_signed_difference_deg=end_signed,
        signed_change_deg=float(signed_change),
        window_summaries=tuple(
            _window_summary(
                signed_rows,
                absolute_rows,
                name,
                start_ms,
                minimum_supported_samples,
            )
            for name, start_ms in (
                ("final_1000ms", -1000.0),
                ("final_500ms", -500.0),
                ("final_250ms", -250.0),
            )
        ),
    )


def _supported_feature(dynamic_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["feature_status"].eq("SUPPORTED")
        & dynamic_df["feature_value"].notna()
    ].copy()


def _window_summary(
    signed_rows: pd.DataFrame,
    absolute_rows: pd.DataFrame,
    name: str,
    start_ms: float,
    minimum_supported_samples: int,
) -> WindowRelationshipSummary:
    signed = signed_rows[
        signed_rows["movement_end_relative_ms"].between(start_ms, 0.0, inclusive="both")
    ]
    absolute = absolute_rows[
        absolute_rows["movement_end_relative_ms"].between(start_ms, 0.0, inclusive="both")
    ]
    supported = int(min(len(signed), len(absolute)))
    if supported < minimum_supported_samples:
        return WindowRelationshipSummary(
            window_name=name,
            start_ms=start_ms,
            end_ms=0.0,
            supported_samples=supported,
            evidence_status="UNAVAILABLE",
            mean_absolute_difference_deg=None,
            signed_start_deg=None,
            signed_end_deg=None,
            signed_change_deg=None,
        )
    signed = signed.sort_values("movement_end_relative_ms")
    return WindowRelationshipSummary(
        window_name=name,
        start_ms=start_ms,
        end_ms=0.0,
        supported_samples=supported,
        evidence_status="SUPPORTED",
        mean_absolute_difference_deg=float(absolute["feature_value"].abs().mean()),
        signed_start_deg=float(signed.iloc[0]["feature_value"]),
        signed_end_deg=float(signed.iloc[-1]["feature_value"]),
        signed_change_deg=float(signed.iloc[-1]["feature_value"] - signed.iloc[0]["feature_value"]),
    )


def _relationship_pattern(
    signed_rows: pd.DataFrame,
    *,
    meaningful_change_deg: float,
) -> tuple[str, str]:
    signed_rows = signed_rows.sort_values("movement_end_relative_ms")
    signed_values = signed_rows["feature_value"].astype(float)
    absolute_values = signed_values.abs()
    if (signed_values.iloc[0] < 0 < signed_values.iloc[-1]) or (
        signed_values.iloc[0] > 0 > signed_values.iloc[-1]
    ):
        return "CROSSING", "The signed injured-minus-contralateral relationship changed sign."
    absolute_change = float(absolute_values.iloc[-1] - absolute_values.iloc[0])
    if absolute_change >= meaningful_change_deg:
        return "DIVERGING", "The absolute projected bilateral HKA difference increased."
    if absolute_change <= -meaningful_change_deg:
        return "CONVERGING", "The absolute projected bilateral HKA difference decreased."
    return "RELATIVELY_STABLE", "The absolute relationship changed less than the analysis threshold."
