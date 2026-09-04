# ACL Movement Analytics Lab — Four-Minute Demonstration

This is a presenter script, not application copy. The demonstration follows the product’s actual hierarchy: workflow, case library, responsible restraint, and cross-case evidence.

## What the audience should understand

By the end, the audience should be able to repeat four ideas:

1. A human establishes the athlete, case context, and visible movement interval.
2. Every movement claim remains traceable to video frames, measurements, and evidence support.
3. Uncertainty changes the result: the system can narrow, report no transition, or withhold.
4. Cross-case findings use independent injury events and mutually supported measurements—not extra frames as extra people.

## Three visualizations to protect

These are the visual proof of the product. If time slips, shorten the narration rather than skipping them:

1. **Movement Story:** the interactive measurement trajectory, phase markers, and supporting source frame.
2. **Comparison:** the selected-case similarity spectrum and its shared-evidence support.
3. **Explore Data:** one Case Breakdowns bar chart showing the cohort as injury events.

## Before presenting

1. Start the unified application:

   ```bash
   .venv/bin/python scripts/run_annotation_ui.py
   ```

2. Open `http://127.0.0.1:8765/` and wait for the case count to appear.
3. Warm these pages in advance:
   - Andi Sullivan supported result: `/results?case=imported_andi_sullivan_2024_10_06_view_02`
   - Comparison: `/compare?case=imported_andi_sullivan_2024_10_06_acl_candidate`
   - Dataset visualization: `/explore`
4. Return to the home page and leave the Injury Case Library search empty.
5. Do not generate a new analysis, save an annotation, delete a case, or cut a new clip during the timed presentation.

## The four-minute script

### 0:00–0:30 — Why ACL injuries matter

**Screen:** Home-page hero.

**Say:**

> ACL stands for anterior cruciate ligament, one of the key ligaments stabilising the knee. In football, a rupture can require reconstruction and keep a player out for up to a year. Exposure-adjusted research has found women footballers’ ACL injury rate is about twice men’s. That burden motivates this work: making documented injury movement observable and auditable without pretending broadcast video can diagnose the cause.

Sources for the wording: [AAOS knee anatomy](https://orthoinfo.aaos.org/globalassets/pdfs/about-your-knee.pdf), [NHS ACL-surgery recovery guidance](https://www.nhs.uk/tests-and-treatments/acl-anterior-cruciate-ligament-surgery/recovering/), and the [football ACL-incidence systematic review and meta-analysis](https://pubmed.ncbi.nlm.nih.gov/29599121/).

### 0:30–0:50 — Product promise and boundary

**Screen:** Stay on the hero.

**Point to:** “Human-guided,” “Evidence-led,” and the scope strip.

**Say:**

> We turn documented injury-event video into traceable, human-guided 2D movement evidence. This is not diagnosis, risk scoring, or a claim about what caused the injury. When the video cannot support a conclusion, the application narrows or withholds it.

### 0:50–1:05 — The operational workflow

**Screen:** Scroll to **Create, annotate, and review a case**.

**Point across the four cards:** Create → Video Cutter → Annotation → Movement Story.

**Say:**

> One workflow connects case creation, the Video Cutter, human annotation, and review. I’ll use a completed case so we can spend our time on the evidence rather than processing.

Do not open the Video Cutter or Annotation Workspace in the base four-minute version. Their buttons prove the connected workflow; use the optional operational proof below if asked.

### 1:05–1:25 — Select one documented case

**Screen:** Scroll to **Injury Case Library**.

**Do:**

1. Search for `Andi Sullivan`.
2. Select the case.
3. Point to the case facts, attached view, analysis progress, and **Review analysis** action.

**Say:**

> This library is organised by injury event, not by frame. The selected case keeps its player information, source views, annotation state, and analysis status together. That preserves provenance and prevents multiple replay angles from becoming fake sample size.

### 1:25–2:20 — Visualize one Movement Story

**Do:** Open Andi Sullivan’s completed analysis.

**Show only:**

1. The selected source movement.
2. The interactive measurement trajectory.
3. The blue phase-start and green phase-end markers.
4. One graph point or phase snapshot and its supporting source frame.
5. The evidence limitation beside it.

**Say:**

> The source movement and measurement trajectory stay together. Here, sufficient continuous 2D evidence supports three sustained movement phases, marked directly on the graph. Selecting a point returns us to the supporting source frame. Unsupported samples remain visible gaps; the visualization never turns absence into zero or certainty.

Avoid opening multiple measurements, technical tables, or the full QC timeline.

### 2:20–2:45 — Responsible AI through real outcomes

**Screen:** Return home and scroll to **Knowing when not to give an absolute answer**.

**Point across the four cases without opening them:**

- Andi Sullivan — supported full Movement Window.
- Charlotte Newsham — phases limited to a continuous supported interval.
- Jordyn Huitema — sufficient evidence, but no supported transition.
- Caroline Weir — insufficient continuous evidence, so phase analysis is withheld.

**Say:**

> Responsible AI here changes the answer. Across real cases, the system can report a supported result, narrow the interval, say no transition was detected, or withhold the result. A human still verifies the event context and supported interval.

**Anchor line:**

> The responsible result is sometimes less information—not a more confident guess.

### 2:45–3:20 — Visualize comparison using shared evidence

**Screen:** Open **Compare Movements** with Andi Sullivan selected.

**Show only:**

1. **Simple** mode.
2. The **Selected-case similarity spectrum** and its support indicator.
3. The first ranked profile and its shared-measurement support.

**Say:**

> This spectrum is not a probability of the same injury. It visualizes a transparent movement ranking based only on measurements both cases support. The support shown beside the index matters: a visually strong match with limited shared evidence must be interpreted cautiously. We can then open the first result to see the actual similarities and differences behind its position.

Do not open Scientific mode or the all-case matrix unless asked.

### 3:20–3:45 — Visualize the cohort

**Screen:** Open **Explore Data** and stay on **Case Breakdowns**.

**Show:** Select one clear category, such as player position or contact mechanism, and show its bar chart. Briefly point to the chart summary or source controls.

**Say:**

> This chart widens the view from one movement to the cohort. Each bar counts documented injury events, so one event still counts once even when it has several replay views or thousands of frames. The same workspace also exposes provenance, evidence coverage, distributions, correlations, and readiness limits.

### 3:45–4:00 — Close

**Say:**

> The innovation is not automatic certainty. It is an auditable path from human-reviewed video to visual evidence, responsible comparison, and cohort exploration—with a system designed to know when not to answer.

Stop there. Do not reopen another screen during the final line.

## Optional operational proof — only if asked

Use this after the timed presentation or replace the 15-second workflow explanation with it.

### Video Cutter

1. Open **Video Cutter**.
2. Choose **Add video views to an existing case**.
3. Select a case and point to its preserved player and injury information.
4. Select a source video and show that Set In, Set Out, Review Cut, and **Cut and add view** belong to that same case.
5. Do not export during a question unless the audience explicitly requests it.

### Annotation Workspace

1. Open a saved case rather than a blank one.
2. Show the target-athlete ROI, Movement Window, documented injured side, and position.
3. Point to the sequence: annotate → verify → validate → generate.
4. Emphasise that saving human review and generating analysis are separate actions.

## If a page is slow

- Keep the warmed pages open and switch to them instead of waiting on camera.
- Continue speaking while a result page loads; do not click repeatedly.
- If the Movement Story is still loading, use the warmed trajectory and supporting-frame view.
- If comparison is still loading, state the shared-evidence rule and move to the warmed Case Breakdowns chart.
- Never start model processing as a recovery step.

## Question-led material

Open these only when relevant to a question:

- all-case comparison matrix;
- Scientific comparison mode;
- complete measurement catalogue;
- frame-by-frame QC timeline;
- similarity validation;
- statistical group tests;
- case deletion or regeneration controls.

## Final presenter check

- The server is responding on port 8765.
- The home page shows case counts.
- Andi Sullivan’s result shows its measurement trajectory, three phases, and supporting frames.
- Comparison shows the selected-case similarity spectrum and a completed ranking.
- Explore Data has loaded the Case Breakdowns bar chart.
- Browser zoom is 100% and no menus are left open.
