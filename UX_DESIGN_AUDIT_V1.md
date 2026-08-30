# ACL Movement Analytics Lab — UX Design Audit V1

Date: 23 August 2026  
Scope: Focused Design, UX, accessibility, and responsive tightening pass. This is not a redesign.

## What was inspected

- Workflow home, completed-case selection, and case switching.
- Results loading, loaded, partial-support, unavailable-measurement, and error presentation.
- Video/frame inspection, Movement Story, phase detail, selected measurements, measurement support, cross-case readiness, and provenance.
- Desktop and narrow responsive behavior in the live local application.
- Existing result-page and home-page test coverage.

## Original problems found

### First-use comprehension and hierarchy

- The home page opened with internal workflow tools but did not quickly explain what the product analyzes, what the judge would see, or why the evidence is useful.
- The results page led with a technical case identifier and a large video. The evidence-backed Movement Story and support state were below the first viewport.
- Detailed phase cards competed with the whole-movement summary. A judge had to read researcher-level detail before reaching the selected measurement.
- The strongest measurement evidence appeared one feature at a time and lacked a small first-level scan of key supported measurements.

### Case selection

- Case selection was visually equal to clip preparation and annotation, even though opening a completed case is the appropriate first action for a judge.
- Options relied heavily on optgroup context and view labels; the selected player/case context was less obvious in the option text itself.
- The results header did not label itself as the selected-case state and could overflow when case identifiers were long.

### Data-support and similarity states

- Supported, limited, and unavailable measurements existed in the underlying UI, but the distinction was not visible in the first-level summary.
- Similarity was deliberately unavailable in the backend, but the active results page had no integrated empty state explaining why.
- The home described similarity as merely planned, rather than evidence-gated by independent human-validated cases and mutually available measurements.
- Unsupported values were correctly retained as missing, but the first-level UI did not explicitly say that unavailable values were not converted to zero.

### Density and responsive behavior

- Phase-level evidence was always expanded, producing substantial scrolling before measurements.
- Long identifiers and header actions could push beyond the viewport at common desktop widths.
- Tables needed an explicit narrow-screen overflow strategy.
- Touch targets were smaller than the preferred 44 px minimum in parts of the workflow/results UI.

### Accessibility

- There was no skip link to the primary case/analysis content.
- Focus visibility relied mainly on browser defaults.
- The trajectory canvas had no programmatic image label or description relationship.
- Frame-step buttons exposed terse visual labels (`-5`, `+5`) without descriptive accessible names.
- Loading and changing-frame states needed clearer live-region semantics.
- Reduced-motion preferences were not considered for programmatic smooth scrolling.

## Changes made

### Home / 30-second explanation

- Added a restrained product introduction answering: what it is, what it analyzes, what the user sees, and why it is useful.
- Moved completed analyses to the first workflow position and made the case options include both player/case and view text.
- Kept video preparation, annotation, statistical exploration, and similarity gating available without changing framework or workflow logic.
- Reworded similarity as evidence-gated and explained the independent-case requirement.

### Results hierarchy

- Added a prominent selected-case overview above the video.
- The overview now presents:
  - the current case and view;
  - analysis completion / measurement-support state;
  - a concise Movement Story;
  - up to three prioritized observable measurements with support labels and supported start/end values;
  - direct routes to watch the movement, explore phase detail, and inspect measurements;
  - a concise projected-2D / non-diagnostic scope statement.
- Kept the existing video, phase engine, measurement graphs, filmstrip, phase comparison, QC, and provenance intact.
- Put detailed phase evidence behind a native disclosure control for progressive disclosure. When phases are withheld, the evidence explanation remains opened so the limitation is not hidden.

### Measurement support

- Added first-level `GOOD`, `LIMITED`, and `UNAVAILABLE` text labels; meaning is not encoded by color alone.
- Unavailable overview measurements say that no zero is substituted.
- Kept unsupported trajectory intervals visible as gaps and retained the existing frame-level explanation.

### Similar cases

- Added an integrated Similar Cases section.
- When the existing backend withholds similarity, the section shows a calm no-data state using the backend reason rather than fake scores or placeholders.
- Added concise language that similarity is limited to mutually supported measured movement and does not imply identical injury mechanism, biological cause, tissue loading, or clinical condition.
- No similarity algorithm, score, or scientific feature was invented in this pass.

### States and error handling

- Added a results loading state with polite live-region semantics.
- Added a user-facing load error state with retry and case-selection routes.
- Prevented analysis-regeneration failures from displaying backend exception text in the normal UI; developer diagnostics remain server-side.
- Kept explicit partial-support, unavailable-measurement, withheld-phase, and unsupported-interval states.

### Accessibility and responsive behavior

- Added skip links and strong keyboard focus indicators.
- Increased primary control/touch target height to 44 px.
- Added descriptive accessible names to frame-step controls.
- Added an accessible label and description relationship to the trajectory canvas.
- Added polite announcements for loading and frame changes.
- Respected reduced-motion preferences for smooth-scrolling behavior.
- Added narrow-screen stacking for overview, measurements, phase cards, actions, audit content, and similarity content.
- Added responsive table overflow and a medium-width stacked results header to prevent long-case overflow.

## Important hierarchy decisions

1. Completed-case selection is the judge entry point; preparation and annotation remain researcher workflow tools.
2. The results overview answers the judge’s first questions before researcher-level phase and trajectory detail.
3. Movement Story remains interpretive and is explicitly separated from measured values and support state.
4. A small prioritized measurement set appears first; the complete existing feature selector remains available afterward.
5. Similarity remains visible even when unavailable, because an honest evidence-gated state is more useful than a missing product section.
6. Measurement support and Responsible AI guidance remain close to the results rather than becoming a detached warning page.

## Deliberately deferred

- A production similarity ranking and “why similar / where different” output. The current backend intentionally reports this unavailable; adding it would be a scientific feature, not a UX tightening.
- New biomechanical inference, automatic mechanism classification, 3D reconstruction, force/load estimates, diagnosis, prognosis, or injury-risk scoring.
- Framework migration, authentication, routing redesign, or backend reorganization.
- A bespoke mobile research-chart redesign. The V1 behavior stacks content and preserves horizontal table access; deeper mobile visualization work should follow researcher testing.
- Replacing imported technical identifiers with player names when verified metadata does not provide a clearer name. No metadata was invented.

## Scientific-language review

- Preserved association/observation language and avoided causal injury claims.
- Preserved the distinction between projected 2D measures and 3D biomechanics.
- Preserved the distinction between measurement support and diagnostic confidence.
- Preserved the distinction between movement similarity and a shared mechanism/cause.
- Preserved missing values and unsupported intervals rather than presenting fake zeros.

## Validation record

- Live desktop review: home and a completed human analysis.
- Live case switching: exercised from the home page into a completed analysis.
- Live results: video/frame controls, Movement Story, selected measurement, support state, and similarity empty state inspected.
- Responsive review: desktop and narrow viewport, including overflow checks.
- Automated focused checks: `tests/test_home_ui.py` and `tests/test_results_ui.py`.
- Full test suite: see final implementation report for the executed result.

