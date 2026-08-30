# EDA and Visualization Audit V1

Audit date: 2026-08-23  
Scope: existing HUMAN case summaries, case-level explorer, per-case Results views, generated diagnostic plots, and similarity presentation. No videos were processed and no scientific feature, pose, tracking, smoothing, annotation, ingestion, or similarity calculation was changed.

## Current data observed

- 5 registered injury events represented by 9 analysed views and 9 case-summary files.
- 36 projected measurement types in the cross-case summaries.
- Geometry summaries are supported for 36/36 measurement types in each currently grouped case, but dynamic support varies materially by case (8/36 to 32/36 in the rendered explorer).
- Injured side is recorded for 3 of the 5 rendered case groups; contact mechanism is not recorded for any current case group.
- 2 case-movement signature artifacts exist. No pairwise similarity, nearest-neighbour, distance, or feature-contribution artifact exists.
- Replays are grouped into their registered injury event and are not counted as independent cases.

## Existing views retained

- The per-case Results trajectory view: retains measurement units, unsupported-frame gaps, source-frame traceability, and existing movement-phase boundaries.
- The per-case bilateral and phase views: retain injured-side versus opposite-side projected comparisons. These are within-case comparisons, not controls or normative baselines.
- Movement Story, evidence cards, source provenance, and measurement-support drilldowns.
- The cross-case Overview, named case comparison, two-measurement comparison, and group-test readiness views.
- Generated PNG diagnostics for pose/landmark QC, geometry, robust dynamics, event-relative trajectories, phase timelines, and movement profiles. These remain engineering/research diagnostics rather than the primary hackathon presentation surface.
- The disabled/planned similarity entry point and Results placeholder. No score is shown without an implemented deterministic result.

## Problems found

1. `null` was coerced to numeric zero in the browser, so an unavailable summary could be plotted and formatted as `0`.
2. Early-to-late change was gated by geometry eligibility rather than dynamic eligibility.
3. Cross-case canvases had axis titles but no numeric tick labels, limiting interpretation.
4. The case overview hid the large difference between geometry support and robust-dynamic support.
5. Comparison tables listed only contributing values, making omitted cases and the true denominator hard to see.
6. The even-sample median calculation selected the upper middle value instead of averaging the two middle values.
7. The product context describes a similarity engine, but this checkout has only future-clustering signatures and a placeholder renderer; no nearest-neighbour calculation or explanation output is implemented.
8. One bilateral phase sentence described the opposite limb as “uninjured-side,” which could imply a healthy control.
9. Several retained static diagnostic plots use code-oriented labels such as `timestamp_ms`, `event_relative_ms`, and raw feature identifiers. Some older titles use “event” even though the human workflow aligns to Movement End. They are useful for technical QC but are not self-explanatory researcher-facing figures.

## Views modified or added

| View | Change | Scientific question answered |
| --- | --- | --- |
| Case Library | Added geometry support, dynamic support, and median geometry/dynamic frame coverage per registered event. | Which cases have enough supported evidence, and is support geometric or dynamic? |
| Measurement Coverage | Added geometry-supported, dynamic-supported, unavailable, limited, and median frame-coverage counts. | Which projected measurements are supported across the case library, and where is evidence missing? |
| Named Case Comparison | Added numeric axes, unit-aware restrained formatting, exact contributing/total case count, correct median line, and rows for omitted cases. | How does one supported case-level descriptor vary among the currently named cases? |
| Two-Measurement Comparison | Added numeric axes and units, contributing/total case count, and a row for every case with inclusion status. | Which cases jointly support both descriptors, and what descriptive relationship is visible without implying causation? |
| Movement Similarity status | Added a deterministic artifact-readiness panel: 2 signatures, 0 pairwise outputs, no displayed score or explanation. | Is an auditable case-similarity comparison actually available in this build? |
| Bilateral phase copy | Replaced “uninjured-side” with “opposite-side.” | How do the two measured limbs compare within this case without treating one as a healthy control? |

No visualization or workflow was removed.

## Visual integrity decisions

- Unavailable and limited values are never converted to or plotted as zero; a true recorded zero remains visible as zero.
- Degree and pixel values use one decimal place; body-scale values use two; coverage uses whole percentages.
- Every cross-case comparison reports the independent-case numerator and denominator.
- The named case comparison is presented as a small-sample case comparison, not a population distribution.
- Spearman rho remains behind technical details and is described as exploratory only.
- Colours remain restrained and are accompanied by labels, counts, line style, or table text rather than carrying meaning alone.

## Known limitations and human review

- Five independent injury events are insufficient for population claims, normative ranges, stable outlier classification, or confirmatory inference.
- Contact mechanism metadata is absent, and laterality is missing for two rendered case groups. Those fields require human source verification; they must not be inferred from footage.
- Feature-by-feature evidence-view selection is defensible for coverage but means different measurements from one registered event may come from different camera views. The explorer discloses selected views and never averages projected angles across views.
- Current summary artifacts contain numeric geometry summaries for all 36 measurement types, so real-data UI review does not exercise every possible geometry-missing state. Synthetic tests cover unavailable, limited, and true-zero separation.
- The per-case Results UI already makes phases and unsupported intervals visible. No new event landmark was introduced.
- Similarity explanations cannot be reviewed because no similarity implementation or pairwise output exists. Implementing a metric, normalization, missing-data policy, and contribution decomposition would be a new analytical feature and is outside this pass.

## Deliberately parked for post-hackathon work

- Design, scientific review, and version a genuine pairwise similarity method before enabling nearest cases, scores, matrices, or explanations. Required outputs should include mutually used dimensions, missing dimensions, per-feature/family contributions, and explicit normalization provenance.
- Consolidate or archive redundant generated PNG diagnostics after researchers identify which technical QC plots remain operationally necessary.
- Replace code-oriented axis labels in retained static diagnostics with researcher-facing labels when those plots become product surfaces; also audit every historical “event” label against Movement End semantics before regeneration.
- Add a dedicated case-by-measurement missingness heatmap only when the library has enough heterogeneous missingness for it to answer more than the coverage table already does.
- Do not add UMAP, t-SNE, PCA, clustering, or inferential tests until a reviewed case matrix, missing-data policy, and adequate independent-case count exist.

## Validation performed

- Targeted EDA/home tests passed.
- Full repository test suite passed.
- The explorer was rendered against current repository data in the local browser; Overview, named case comparison, dynamic-gated comparison, and two-measurement comparison were exercised.
- The existing Christen Press Results page loaded after the changes.
- No browser console warnings or errors were observed on the tested explorer and Results pages.
- Python compilation passed. Ruff was not available in the active environment.

