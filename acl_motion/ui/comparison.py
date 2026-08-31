"""Dedicated movement-comparison workspace."""

from __future__ import annotations

import json

from acl_motion.ui.app_shell import app_shell_css, app_site_header

COMPARISON_EXPERIENCE = {
    "default_mode": "simple",
    "default_lens": "overall_movement_difference",
    "lenses": [
        {
            "id": "overall_movement_difference",
            "label": "Weighted robust movement difference",
            "measure": "Reliability-weighted, robustly scaled L1 distance",
            "simple": {
                "interpretation": (
                    "Looks at how different two movements are across measurements available "
                    "in both cases."
                ),
                "advantages": (
                    "Easy to explain, works when some measurements are missing, and shows "
                    "which measurements created the difference."
                ),
                "disadvantages": (
                    "May not capture the order in which the movement changed over time."
                ),
            },
            "scientific": {
                "interpretation": (
                    "Combines robustly scaled absolute differences over mutually eligible features "
                    "using evidence and movement-family weights."
                ),
                "advantages": (
                    "Supports pairwise missingness and deterministic feature contributions; "
                    "robust scaling limits domination by feature units."
                ),
                "disadvantages": (
                    "Does not model temporal order or feature covariance automatically and "
                    "depends on predefined feature scales."
                ),
            },
        },
        {
            "id": "movement_pattern_direction",
            "label": "Weighted cosine sensitivity",
            "measure": "Reliability-weighted, robustly scaled cosine similarity",
            "simple": {
                "interpretation": (
                    "Checks whether the supported body-measurement pattern points in a similar "
                    "direction after scaling and quality weighting."
                ),
                "advantages": (
                    "Can find a shared movement pattern even when one movement is stronger "
                    "than the other."
                ),
                "disadvantages": (
                    "A weak movement and a strong movement can sometimes look more similar "
                    "than expected."
                ),
            },
            "scientific": {
                "interpretation": (
                    "Uses weighted cosine similarity after robust feature scaling, mutual-feature masking, "
                    "reliability weighting, and movement-family balancing."
                ),
                "advantages": (
                    "Measures multivariate directional agreement without allowing large-unit "
                    "features to dominate."
                ),
                "disadvantages": (
                    "Is largely insensitive to magnitude and can be unstable for near-zero "
                    "vectors or limited feature overlap."
                ),
            },
        },
        {
            "id": "relationship_aware_pattern",
            "label": "Soft-cosine sensitivity",
            "measure": "Reliability-weighted soft cosine similarity",
            "simple": {
                "interpretation": "Allows closely related movement measurements to support each other.",
                "advantages": (
                    "Can recognize similarities that a standard feature-by-feature comparison "
                    "might miss."
                ),
                "disadvantages": (
                    "Depends on correctly deciding which measurements are genuinely related."
                ),
            },
            "scientific": {
                "interpretation": (
                    "Uses a conservative predefined feature-relationship matrix within a "
                    "reliability-weighted soft-cosine calculation."
                ),
                "advantages": (
                    "Allows justified cross-feature correspondence and partial semantic overlap "
                    "between descriptors."
                ),
                "disadvantages": (
                    "The current relationship matrix is provisional and requires scientific "
                    "validation; attribution is less direct."
                ),
            },
        },
        {
            "id": "large_difference_focus",
            "label": "Weighted Euclidean sensitivity",
            "measure": "Reliability-weighted, robustly scaled Euclidean distance",
            "simple": {
                "interpretation": (
                    "Gives extra importance to measurements where the cases are very different."
                ),
                "advantages": "Good at highlighting one or two major movement differences.",
                "disadvantages": (
                    "A noisy or unusual measurement can affect the result more strongly."
                ),
            },
            "scientific": {
                "interpretation": (
                    "Combines squared, robustly scaled feature differences with reliability "
                    "and movement-family weights."
                ),
                "advantages": (
                    "Penalizes large residuals and provides sensitivity analysis against the "
                    "primary absolute-distance ranking."
                ),
                "disadvantages": (
                    "More sensitive to outliers, scale misspecification, and correlated-feature "
                    "duplication than the primary robust L1 distance."
                ),
            },
        },
        {
            "id": "trajectory_shape",
            "label": "Movement over time",
            "measure": "Reliability-weighted constrained Soft-DTW",
            "future": True,
            "simple": {
                "interpretation": "Compares how the full movement develops from beginning to end.",
                "advantages": "Can compare movements performed at different speeds.",
                "disadvantages": (
                    "It is harder to explain and can hide timing differences if alignment is "
                    "allowed to stretch too far."
                ),
            },
            "scientific": {
                "interpretation": (
                    "Compares supported time-normalized trajectories with constrained soft "
                    "dynamic time warping."
                ),
                "advantages": (
                    "Uses sequential shape information and tolerates bounded temporal variation."
                ),
                "disadvantages": (
                    "Alignment constraints and temporal cost require validation; unconstrained "
                    "warping can erase meaningful timing differences."
                ),
            },
        },
    ],
    "evidence_factors": [
        {
            "label": "Shared information",
            "simple": {
                "evaluation": "Did both cases provide enough of the same measurements?",
                "example": "Fourteen shared measurements were available in both cases.",
            },
            "scientific": {
                "evaluation": (
                    "Weighted overlap of mutually eligible features after evidence and "
                    "compatibility masking."
                ),
                "example": "Report weighted feature overlap and the core-feature overlap gate.",
            },
        },
        {
            "label": "Body coverage",
            "simple": {
                "evaluation": "Did the comparison cover several areas of the body?",
                "example": "The comparison covered lower limb, trunk, upper body, and timing.",
            },
            "scientific": {
                "evaluation": "Coverage of predefined movement-feature families.",
                "example": "Report eligible and expected feature families for the pair.",
            },
        },
        {
            "label": "Measurement quality",
            "simple": {
                "evaluation": "Was the player visible and tracked consistently?",
                "example": "Most shared measurements had good support across the movement.",
            },
            "scientific": {
                "evaluation": (
                    "Pairwise aggregation of pose, landmark, dynamic, and descriptor coverage "
                    "under existing quality rules."
                ),
                "example": "Report weighted joint evidence support for compared features.",
            },
        },
        {
            "label": "Camera-view match",
            "simple": {
                "evaluation": "Were the movements filmed from views that can be compared fairly?",
                "example": "The views were usable, but they were not a perfect match.",
            },
            "scientific": {
                "evaluation": "Compatibility between each descriptor's required and observed view.",
                "example": "Apply descriptor-level view gates before pairwise scoring.",
            },
        },
        {
            "label": "Library support",
            "simple": {
                "evaluation": "Were enough independent, comparable cases available?",
                "example": "Replays of the same injury count as one case, not several.",
            },
            "scientific": {
                "evaluation": (
                    "Number and local density of independent eligible reference cases after "
                    "case-level deduplication."
                ),
                "example": "Report eligible reference count and effective independent case count.",
            },
        },
    ],
    "evidence_states": [
        {
            "id": "HIGH",
            "label": "High",
            "simple": (
                "Strong shared information, good measurement quality, compatible views, and "
                "a stable result."
            ),
            "scientific": (
                "All predefined overlap, evidence, compatibility, and reference-adequacy "
                "criteria are satisfied."
            ),
        },
        {
            "id": "MODERATE",
            "label": "Moderate",
            "simple": (
                "The comparison is useful, but some information is missing or one part is limited."
            ),
            "scientific": (
                "Minimum gates pass, but one or more non-critical evidence components remain limited."
            ),
        },
        {
            "id": "LOW",
            "label": "Low",
            "simple": (
                "Important information is limited, or the result changes noticeably after "
                "small adjustments."
            ),
            "scientific": (
                "The pair passes minimum availability gates but has weak evidence coverage "
                "or view compatibility."
            ),
        },
        {
            "id": "PROVISIONAL",
            "label": "Provisional",
            "simple": (
                "A result can be calculated, but the reference library or validation is not "
                "yet strong enough."
            ),
            "scientific": (
                "Pairwise computation is possible, but reference-size or external-validation "
                "criteria have not been met."
            ),
        },
        {
            "id": "UNAVAILABLE",
            "label": "Unavailable",
            "simple": "The minimum requirements for a responsible comparison were not met.",
            "scientific": (
                "At least one hard eligibility gate failed, so no similarity index or rank is emitted."
            ),
        },
    ],
}


def render_comparison_page() -> str:
    """Return the dedicated movement-comparison experience."""

    return (
        COMPARISON_HTML.replace("__APP_SHELL_CSS__", app_shell_css())
        .replace("__APP_SITE_HEADER__", app_site_header("Compare Movements"))
        .replace("__COMPARISON_EXPERIENCE__", json.dumps(COMPARISON_EXPERIENCE))
    )


COMPARISON_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Compare Movements - ACL Movement Analytics Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7f9;
      --panel: #ffffff;
      --ink: #1f2a33;
      --muted: #617080;
      --line: #d5dee7;
      --accent: #215f9a;
      --accent-soft: #e9f2fb;
      --green: #18744a;
      --green-soft: #e9f6ef;
      --amber: #8a6200;
      --amber-soft: #fff4cf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .tool-header {
      min-height: 60px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .button, button, select, input {
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    .button, button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 13px;
      font-weight: 780;
      text-decoration: none;
      cursor: pointer;
    }
    button:hover, .button:hover, select:hover, input:hover { border-color: var(--accent); }
    main { width: min(1240px, calc(100% - 28px)); margin: 0 auto; padding: 28px 0 48px; }
    h1 { margin: 0; font-size: 28px; }
    h2 { margin: 0 0 6px; font-size: 20px; }
    h3 { margin: 0 0 5px; font-size: 16px; }
    p { line-height: 1.5; }
    .lede { margin: 7px 0 20px; color: var(--muted); }
    .panel {
      margin-bottom: 14px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .controls {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
    }
    .control { display: grid; gap: 5px; min-width: min(100%, 330px); }
    .control-label, label { color: var(--muted); font-size: 12px; font-weight: 800; }
    select { width: 100%; padding: 7px 10px; }
    .mode-toggle {
      display: inline-flex;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #f6f8fa;
    }
    .mode-toggle button { min-height: 34px; border: 0; background: transparent; color: var(--muted); }
    .mode-toggle button[aria-pressed="true"] {
      background: var(--panel);
      color: var(--accent);
      box-shadow: 0 1px 3px rgba(31, 42, 51, 0.14);
    }
    .lens-help { margin: 12px 0 0; color: var(--muted); }
    .comparison-layout {
      display: grid;
      grid-template-columns: minmax(245px, .34fr) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .player-picker { position: sticky; top: 12px; }
    .search-control { display: grid; gap: 6px; margin-top: 12px; }
    .search-control input { width: 100%; padding: 8px 10px; }
    .player-list { display: grid; gap: 7px; max-height: 560px; margin-top: 10px; overflow-y: auto; }
    .player-choice {
      width: 100%;
      min-height: 0;
      display: grid;
      gap: 3px;
      justify-items: start;
      padding: 10px;
      text-align: left;
      background: #fbfcfd;
    }
    .player-choice[aria-pressed="true"] {
      border-color: var(--accent);
      background: var(--accent-soft);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .player-choice strong { font-size: 14px; }
    .player-choice span { color: var(--muted); font-size: 11px; font-weight: 650; }
    .empty-state { padding: 14px; border: 1px dashed var(--line); border-radius: 8px; color: var(--muted); }
    .results-column { min-width: 0; }
    .selection-header { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
    .selection-header h2 { margin-bottom: 3px; }
    .selection-meta { margin: 0; color: var(--muted); font-size: 13px; }
    .engine-note { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
    .ranking-list { display: grid; gap: 10px; }
    .match-card {
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr) 116px;
      gap: 13px;
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .match-card.simple { grid-template-columns: 44px minmax(0, 1fr); }
    .match-rank {
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 900;
    }
    .match-name { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .match-name h3 { margin: 0; font-size: 18px; }
    .match-meta { margin: 4px 0 8px; color: var(--muted); font-size: 12px; }
    .reliability-copy, .stability-copy, .view-pair-copy { margin: 7px 0 0; color: #40505f; font-size: 13px; }
    .view-pair-copy { padding: 7px 9px; border-left: 3px solid var(--green); background: var(--green-soft); }
    .score-box { text-align: right; }
    .score-box strong { display: block; color: var(--accent); font-size: 30px; line-height: 1; }
    .score-box span { color: var(--muted); font-size: 11px; font-weight: 760; }
    .comparison-details { grid-column: 2 / -1; display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .measurement-group { padding: 10px; border-radius: 8px; background: #f6f8fa; }
    .measurement-group h4 { margin: 0 0 5px; font-size: 12px; }
    .measurement-group ul { margin: 0; padding-left: 17px; color: var(--muted); font-size: 12px; line-height: 1.5; }
    .stability-copy { padding: 7px 9px; border-left: 3px solid var(--accent); background: var(--accent-soft); }
    .pool-label { margin-top: 3px; font-size: 11px; font-weight: 800; }
    .pool-label.reference { color: var(--green); }
    .pool-label.query { color: var(--amber); }
    .help-panel summary { cursor: pointer; font-weight: 850; font-size: 16px; }
    .help-content { margin-top: 14px; }
    .result-grid { display: grid; grid-template-columns: 1.25fr .75fr; gap: 12px; }
    .result-card {
      min-height: 150px;
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 9px;
      background: #fbfcfd;
    }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--amber-soft);
      color: var(--amber);
      font-size: 12px;
      font-weight: 850;
    }
    .badge.good { background: var(--green-soft); color: var(--green); }
    .badge.neutral { background: #eef1f4; color: #526170; }
    .measure { color: var(--accent); font-weight: 780; }
    .readiness-counts { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
    .readiness-counts span { color: var(--muted); font-size: 12px; }
    .readiness-counts strong { display: block; color: var(--ink); font-size: 22px; }
    .section-copy { margin: 0 0 10px; color: var(--muted); }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 9px; }
    table { width: 100%; min-width: 820px; border-collapse: collapse; font-size: 13px; }
    .reliability-table { min-width: 690px; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; line-height: 1.45; }
    th { background: #f4f7fa; color: #40505f; font-size: 12px; }
    tr:last-child td { border-bottom: 0; }
    td:first-child { font-weight: 820; }
    tr.selected td { background: #f2f7fc; }
    .future { display: inline-block; margin-top: 4px; color: var(--muted); font-size: 11px; }
    .states { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; }
    .state { padding: 10px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; font-size: 12px; line-height: 1.45; }
    .state .badge { margin-bottom: 5px; }
    .scope-note { padding: 11px 13px; border-left: 4px solid var(--accent); background: var(--accent-soft); color: #28465f; }
    @media (max-width: 760px) {
      .tool-header { align-items: flex-start; flex-direction: column; padding: 11px 14px; }
      main { width: min(100% - 20px, 640px); padding-top: 20px; }
      .comparison-layout, .controls, .result-grid, .states, .comparison-details { grid-template-columns: 1fr; }
      .controls { display: grid; }
      .control { width: 100%; }
      .mode-toggle { width: 100%; }
      .mode-toggle button { flex: 1; }
      .player-picker { position: static; }
      .player-list { max-height: 280px; }
      .match-card { grid-template-columns: 40px minmax(0, 1fr); }
      .score-box { grid-column: 2; text-align: left; }
      .comparison-details { grid-column: 1 / -1; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="app-skip-link" href="#mainContent">Skip to movement comparison</a>
  __APP_SITE_HEADER__
  <header class="tool-header">
    <strong>Compare Movements</strong>
    <a class="button" href="/">Main menu</a>
  </header>
  <main id="mainContent" class="app-page-main" tabindex="-1">
    <h1>Compare Movements</h1>
    <p class="lede">Choose an injury case with a completed view that covers the visible injury event. Each injury is listed once, even when it has several camera views.</p>

    <div class="comparison-layout">
      <aside class="panel player-picker" aria-labelledby="choosePlayerHeading">
        <h2 id="choosePlayerHeading">Choose an injury case</h2>
        <p class="section-copy">Search the analysed injury cases available for comparison.</p>
        <div class="search-control">
          <label for="playerSearch">Find a player</label>
          <input id="playerSearch" type="search" placeholder="Search by player or team" autocomplete="off" />
        </div>
        <div class="player-list" id="playerList" role="listbox" aria-label="Analysed injury cases">
          <div class="app-football-loader compact" role="status" aria-live="polite">
            <span class="app-loader-pitch" aria-hidden="true"><span class="app-loader-ball">⚽</span></span>
            <span class="app-loader-copy"><strong>Checking the match-ups…</strong><small>Finding cases with mutually supported measurements.</small></span>
          </div>
        </div>
        <p class="engine-note" id="poolSummary"></p>
      </aside>

      <div class="results-column">
        <section class="panel" aria-labelledby="selectedPlayerHeading">
          <div class="selection-header">
            <div>
              <p class="control-label">Selected injury case</p>
              <h2 id="selectedPlayerHeading">Choose an injury case to begin</h2>
              <p class="selection-meta" id="selectedPlayerMeta"></p>
            </div>
            <span class="badge neutral" id="engineStatus">Loading</span>
          </div>
          <div class="controls" style="margin-top:14px">
            <div class="control">
              <label for="comparisonLens">Comparison lens</label>
              <select id="comparisonLens" aria-describedby="lensHelp"></select>
            </div>
            <div class="control">
              <span class="control-label" id="modeLabel">Explanation</span>
              <div class="mode-toggle" role="group" aria-labelledby="modeLabel">
                <button id="simpleMode" type="button" aria-pressed="true">Simple</button>
                <button id="scientificMode" type="button" aria-pressed="false">Scientific</button>
              </div>
            </div>
          </div>
          <p class="lens-help" id="lensHelp" aria-live="polite"></p>
          <p class="engine-note" id="engineNote" hidden></p>
        </section>

        <section aria-labelledby="rankingHeading">
          <div class="selection-header" style="margin:18px 2px 10px">
            <div>
              <h2 id="rankingHeading">Closest injury-event movement profiles</h2>
              <p class="selection-meta">Every eligible view pair is checked, but only the single most similar pair supplies the displayed index and measurements. Views are never averaged. It does not determine whether two players experienced the same ACL injury mechanism.</p>
              <p class="selection-meta" id="rankingSummary">Choose an injury case to see ranked results.</p>
            </div>
          </div>
          <div class="ranking-list" id="rankingList" aria-live="polite">
            <div class="empty-state">No player selected.</div>
          </div>
        </section>
      </div>
    </div>

    <details class="panel help-panel" style="margin-top:18px">
      <summary>How the comparison works and how to interpret it</summary>
      <div class="help-content">
        <section aria-labelledby="lensTableHeading">
          <h2 id="lensTableHeading">How each comparison lens works</h2>
          <p class="section-copy" id="modeDescription"></p>
          <div class="table-wrap" id="lensTable"></div>
        </section>
        <section aria-labelledby="reliabilityHeading" style="margin-top:16px">
          <h2 id="reliabilityHeading">How evidence support is judged</h2>
          <p class="section-copy">The similarity index describes closeness under one lens. Evidence support describes whether the pair has enough comparable information. Rank stability is reported separately.</p>
          <div class="table-wrap" id="reliabilityTable"></div>
          <div class="states" id="reliabilityStates"></div>
        </section>
        <section aria-labelledby="validationHeading" style="margin-top:16px">
          <h2 id="validationHeading">Validation work</h2>
          <p class="section-copy">Query-excluded scaling and internal reference-case sensitivity audits are now implemented. The current rankings remain exploratory and are not externally validated measurements of biomechanical or clinical similarity.</p>
          <ul>
            <li>Repeat annotations independently and quantify agreement.</li>
            <li>Compare projected measurements with laboratory or multi-camera references.</li>
            <li>Collect blinded expert pairwise similarity judgements.</li>
            <li>Test rankings on players held out from development.</li>
            <li>Freeze scaling parameters from a separate, substantially larger reference cohort.</li>
          </ul>
          <a class="button" href="/validate-similarity">Open blinded expert review</a>
        </section>
      </div>
    </details>

    <p class="scope-note">The lens-specific index is not a percentage or probability. A case with more eligible views has more opportunities to produce a close match, so the number of view pairs checked is shown with every result. Evidence support and resampling stability are not diagnostic confidence or evidence of a shared injury mechanism.</p>
  </main>
  <script>
    const experience = __COMPARISON_EXPERIENCE__;
    const $ = id => document.getElementById(id);
    let mode = experience.default_mode;
    let selectedLensId = experience.default_lens;
    const app = {
      index: null,
      comparison: null,
      selectedCaseId: new URLSearchParams(window.location.search).get("case") || "",
      comparisonRequestId: 0,
      comparisonAbortController: null,
      comparisonClientId: (window.crypto && typeof window.crypto.randomUUID === "function")
        ? window.crypto.randomUUID()
        : `comparison-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      search: "",
    };

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
      }[char]));
    }
    function selectedLens() {
      return experience.lenses.find(lens => lens.id === selectedLensId) || experience.lenses[0];
    }
    function setLens(lensId) {
      selectedLensId = lensId;
      $("comparisonLens").value = lensId;
      renderEducation();
      renderRankings();
    }
    function setMode(nextMode) {
      mode = nextMode === "scientific" ? "scientific" : "simple";
      renderEducation();
      renderRankings();
    }
    function statusClass(status) {
      if (status === "HIGH") return "good";
      if (["MODERATE", "PROVISIONAL"].includes(status)) return "";
      return "neutral";
    }
    function caseMeta(item) {
      return [item.team, item.injury_date, item.competition].filter(Boolean).join(" · ") ||
        `${item.analysed_view_count || 1} analysed video ${item.analysed_view_count === 1 ? "view" : "views"}`;
    }
    function renderEducation() {
      const lens = selectedLens();
      const copy = lens[mode];
      $("simpleMode").setAttribute("aria-pressed", String(mode === "simple"));
      $("scientificMode").setAttribute("aria-pressed", String(mode === "scientific"));
      $("lensHelp").textContent = `${lens.label}: ${copy.interpretation}`;
      $("engineNote").hidden = mode !== "scientific";
      $("modeDescription").textContent = mode === "simple"
        ? "Everyday language about what each comparison can and cannot show."
        : "Technical information about scaling, weighting, overlap, and stability.";

      $("lensTable").innerHTML = `<table><thead><tr><th>Comparison lens</th><th>Measure</th><th>${mode === "simple" ? "What it tells you" : "Interpretation"}</th><th>Advantages</th><th>Disadvantages</th></tr></thead><tbody>` +
        experience.lenses.map(item => {
          const itemCopy = item[mode];
          return `<tr class="${item.id === selectedLensId ? "selected" : ""}"><td>${escapeHtml(item.label)}${item.future ? '<br><span class="future">Future</span>' : ""}</td><td><span class="measure">${escapeHtml(item.measure)}</span></td><td>${escapeHtml(itemCopy.interpretation)}</td><td>${escapeHtml(itemCopy.advantages)}</td><td>${escapeHtml(itemCopy.disadvantages)}</td></tr>`;
        }).join("") + `</tbody></table>`;

      $("reliabilityTable").innerHTML = `<table class="reliability-table"><thead><tr><th>Evidence factor</th><th>${mode === "simple" ? "What it looks at" : "What it evaluates"}</th><th>${mode === "simple" ? "Simple example" : "Scientific assessment"}</th></tr></thead><tbody>` +
        experience.evidence_factors.map(factor => {
          const factorCopy = factor[mode];
          return `<tr><td>${escapeHtml(factor.label)}</td><td>${escapeHtml(factorCopy.evaluation)}</td><td>${escapeHtml(factorCopy.example)}</td></tr>`;
        }).join("") + `</tbody></table>`;

      $("reliabilityStates").innerHTML = experience.evidence_states.map(state =>
        `<div class="state"><span class="badge ${statusClass(state.id)}">${escapeHtml(state.label)}</span><br>${escapeHtml(state[mode])}</div>`
      ).join("");
    }
    function renderPlayerList() {
      if (!app.index) return;
      const query = app.search.trim().toLowerCase();
      const filtered = app.index.cases.filter(item =>
        [item.player_name, item.team, item.competition, item.injury_date]
          .join(" ").toLowerCase().includes(query)
      );
      if (!filtered.length) {
        $("playerList").innerHTML = '<div class="empty-state">No analysed injury cases match that search.</div>';
        return;
      }
      $("playerList").innerHTML = filtered.map(item =>
        `<button class="player-choice" type="button" role="option" data-case-id="${escapeHtml(item.case_id)}" aria-selected="${item.case_id === app.selectedCaseId}" aria-pressed="${item.case_id === app.selectedCaseId}">` +
          `<strong>${escapeHtml(item.player_name)}</strong>` +
          `<span>${escapeHtml(caseMeta(item))}</span>` +
          `<span>${item.comparable_descriptor_count} supported movement measurements</span>` +
          `<span class="pool-label ${item.reference_pool_eligible ? "reference" : "query"}">${item.reference_pool_eligible ? "Eligible injury-event case" : "Not eligible as a reference"}</span>` +
        `</button>`
      ).join("");
      $("playerList").querySelectorAll("[data-case-id]").forEach(button => {
        button.addEventListener("click", () => selectCase(button.dataset.caseId));
      });
    }
    function measurementList(items) {
      if (!items || !items.length) return "<li>No supported explanation available.</li>";
      return items.map(item => {
        const unit = item.unit ? ` ${item.unit}` : "";
        const values = item.selected_value == null || item.candidate_value == null
          ? ""
          : `: ${Number(item.selected_value).toFixed(3)}${unit} vs ${Number(item.candidate_value).toFixed(3)}${unit}`;
        const difference = item.absolute_difference == null
          ? ""
          : ` · difference ${item.absolute_difference}${unit}`;
        return `<li>${escapeHtml(item.label)}${escapeHtml(values)} <span class="future">${escapeHtml(item.family)}${escapeHtml(difference)}</span></li>`;
      }).join("");
    }
    function renderRankings() {
      const data = app.comparison;
      const lens = selectedLens();
      if (!data || !data.selected_case) {
        $("selectedPlayerHeading").textContent = "Choose an injury case to begin";
        $("selectedPlayerMeta").textContent = "";
        $("rankingSummary").textContent = "Choose an injury case to see ranked results.";
        $("rankingList").innerHTML = '<div class="empty-state">No injury case selected.</div>';
        return;
      }
      const selected = data.selected_case;
      const matches = data.rankings[selectedLensId] || [];
      $("selectedPlayerHeading").textContent = selected.player_name;
      $("selectedPlayerMeta").textContent = `${caseMeta(selected)} · ${selected.reference_pool_eligible ? "eligible reference case" : "query-only case; excluded from reference scaling and candidates"}`;
      $("engineStatus").className = `badge ${data.available ? "good" : "neutral"}`;
      $("engineStatus").textContent = data.available ? "Comparison ready" : "Unavailable";
      $("engineNote").textContent = mode === "scientific"
        ? `${lens.measure}. ${data.index_note} ${data.scaling?.note || ""}`
        : "";
      $("rankingSummary").textContent = matches.length
        ? `Ranked against ${data.summary.reference_pool_case_count} event-covered injury cases using ${lens.label.toLowerCase()}. Each injury appears once.`
        : "No other case met the minimum shared-information requirements for this lens.";
      if (!matches.length) {
        $("rankingList").innerHTML = '<div class="empty-state">No responsible comparison is available for this injury case and lens.</div>';
        return;
      }
      $("rankingList").innerHTML = matches.map(match => {
        const item = match.case;
        const support = match.evidence_support;
        const stability = match.stability || {};
        const viewPair = match.selected_view_pair || {};
        const selectedView = viewPair.selected_case || {};
        const candidateView = viewPair.candidate_case || {};
        const pairCount = Number(viewPair.eligible_view_pair_count || 0);
        const pairWord = pairCount === 1 ? "pair" : "pairs";
        const viewPairText = `Best eligible-view match: ${selectedView.view_label || "eligible view"} ↔ ${candidateView.view_label || "eligible view"} · selected from ${pairCount} eligible view ${pairWord}.`;
        return `<article class="match-card ${mode}">` +
          `<div class="match-rank" aria-label="Rank ${match.rank}">#${match.rank}</div>` +
          `<div>` +
            `<div class="match-name"><h3>${escapeHtml(item.player_name)}</h3><span class="badge ${statusClass(support.status)}">${escapeHtml(support.label)} evidence</span></div>` +
            `<p class="match-meta">${escapeHtml(caseMeta(item))}</p>` +
            `<p class="view-pair-copy"><strong>${escapeHtml(viewPairText)}</strong> No view average is used.</p>` +
            `<p class="reliability-copy">${escapeHtml(support.explanation)}</p>` +
            `<p class="stability-copy"><strong>${escapeHtml(stability.label || "Stability unavailable")}:</strong> ${escapeHtml(stability.explanation || "No resampling result was available.")}</p>` +
          `</div>` +
          (mode === "scientific" ? `<div class="score-box"><strong>${Number(match.similarity_index).toFixed(3)}</strong><span>Best-view index (0–1)</span></div>` : "") +
          `<div class="comparison-details">` +
            `<div class="measurement-group"><h4>${mode === "simple" ? "Most alike" : "Smallest robustly scaled gaps"}</h4><ul>${measurementList(match.closest_measurements)}</ul></div>` +
            `<div class="measurement-group"><h4>${mode === "simple" ? "Largest differences" : "Largest reliability-weighted gaps"}</h4><ul>${measurementList(match.largest_differences)}</ul></div>` +
          `</div>` +
        `</article>`;
      }).join("");
    }
    function syncComparisonUrl(caseId, historyMode = "replace") {
      const url = new URL(window.location.href);
      if (caseId) url.searchParams.set("case", caseId);
      else url.searchParams.delete("case");
      const method = historyMode === "push" ? "pushState" : "replaceState";
      window.history[method]({caseId: caseId || ""}, "", url);
    }
    async function selectCase(caseId, {historyMode = "push"} = {}) {
      if (!caseId) return;
      if (app.comparisonAbortController) app.comparisonAbortController.abort();
      const controller = new AbortController();
      app.comparisonAbortController = controller;
      const requestId = ++app.comparisonRequestId;
      app.selectedCaseId = caseId;
      if (historyMode) syncComparisonUrl(caseId, historyMode);
      renderPlayerList();
      const pendingCase = app.index?.cases?.find(item => item.case_id === caseId);
      if (pendingCase) {
        $("selectedPlayerHeading").textContent = pendingCase.player_name;
        $("selectedPlayerMeta").textContent = `${caseMeta(pendingCase)} · loading comparison…`;
      }
      $("engineStatus").className = "badge neutral";
      $("engineStatus").textContent = "Comparing";
      $("rankingList").innerHTML = '<div class="app-football-loader" role="status" aria-live="polite"><span class="app-loader-pitch" aria-hidden="true"><span class="app-loader-ball">⚽</span></span><span class="app-loader-copy"><strong>Checking the match-ups…</strong><small>Comparing only mutually supported movement measurements.</small></span></div>';
      try {
        const response = await fetch(
          `/api/movement-comparison?case=${encodeURIComponent(caseId)}`
            + `&client_id=${encodeURIComponent(app.comparisonClientId)}`
            + `&request_id=${encodeURIComponent(requestId)}`,
          {signal: controller.signal},
        );
        if (!response.ok) throw new Error("Movement comparison could not be loaded.");
        const comparison = await response.json();
        if (requestId !== app.comparisonRequestId || caseId !== app.selectedCaseId) return;
        app.comparison = comparison;
        renderRankings();
      } catch (error) {
        if (error.name === "AbortError") return;
        if (requestId !== app.comparisonRequestId || caseId !== app.selectedCaseId) return;
        app.comparison = null;
        $("engineStatus").className = "badge neutral";
        $("engineStatus").textContent = "Unavailable";
        $("rankingList").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
      }
    }
    async function loadComparisonIndex() {
      try {
        const response = await fetch('/api/movement-comparison');
        if (!response.ok) throw new Error("Analysed players could not be loaded.");
        app.index = await response.json();
        app.comparison = null;
        const summary = app.index.summary || {};
        $("poolSummary").textContent = `${summary.reference_pool_case_count || 0} eligible reference cases · ${summary.query_only_case_count || 0} query-only cases.`;
        const availableCaseIds = new Set((app.index.cases || []).map(item => item.case_id));
        if (!availableCaseIds.has(app.selectedCaseId)) app.selectedCaseId = "";
        renderPlayerList();
        if (app.index.cases.length) await selectCase(
          app.selectedCaseId || app.index.cases[0].case_id,
          {historyMode: "replace"},
        );
        else {
          $("engineStatus").textContent = "Unavailable";
          $("playerList").innerHTML = '<div class="empty-state">No analysed injury cases are ready for comparison.</div>';
          renderRankings();
        }
      } catch (error) {
        app.comparison = null;
        $("engineStatus").textContent = "Unavailable";
        $("playerList").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
        renderRankings();
      }
    }
    $("comparisonLens").innerHTML = experience.lenses.filter(lens => !lens.future).map(lens =>
      `<option value="${escapeHtml(lens.id)}">${escapeHtml(lens.label)}</option>`
    ).join("");
    $("comparisonLens").value = selectedLensId;
    $("comparisonLens").addEventListener("change", event => setLens(event.target.value));
    $("simpleMode").addEventListener("click", () => setMode("simple"));
    $("scientificMode").addEventListener("click", () => setMode("scientific"));
    $("playerSearch").addEventListener("input", event => {
      app.search = event.target.value;
      renderPlayerList();
    });
    window.addEventListener("popstate", () => {
      if (!app.index?.cases?.length) return;
      const requested = new URLSearchParams(window.location.search).get("case") || "";
      const available = app.index.cases.some(item => item.case_id === requested);
      const fallback = app.index.cases[0].case_id;
      selectCase(available ? requested : fallback, {
        historyMode: available ? null : "replace",
      });
    });
    renderEducation();
    loadComparisonIndex();
  </script>
</body>
</html>
"""
