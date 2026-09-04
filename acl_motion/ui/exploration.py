"""Cross-case descriptive exploration page."""

from __future__ import annotations

from acl_motion.ui.app_shell import app_shell_css, app_site_header, apply_app_brand


def render_exploration_page(*, initial_view: str = "overview") -> str:
    """Return the evidence-gated statistical exploration workspace."""

    correlation_landing = initial_view == "correlations"
    page_title = (
        "Feature Correlations - ACL Movement Analytics Lab"
        if correlation_landing
        else "Explore Data - ACL Movement Analytics Lab"
    )
    page_heading = "Feature Correlations" if correlation_landing else "Explore Data"
    page_lede = (
        "Investigate positive and negative relationships between supported movement "
        "measurements across the complete case library. This is a feature-by-feature "
        "analysis, independent of movement-similarity rankings."
        if correlation_landing
        else "Case-level descriptive analytics with visible evidence coverage and statistical "
        "safeguards. Case identity, injury status, and laterality come from supplied metadata, "
        "not video inference."
    )
    section_label = "Feature Correlations" if correlation_landing else "Explore Data"

    return apply_app_brand(r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f8fa;
      --panel: #ffffff;
      --ink: #142334;
      --muted: #586879;
      --line: #d7e0e7;
      --accent: #0F62FE;
      --accent-soft: #eef5ff;
      --green: #08766d;
      --green-soft: #dffaf4;
      --amber: #8a6200;
      --amber-soft: #fff4cf;
      --red: #9a3040;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a.button, button, select, input {
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    a.button, button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 13px;
      font-weight: 780;
      text-decoration: none;
      cursor: pointer;
    }
    button:hover, a.button:hover, select:hover { border-color: var(--accent); }
    button.active { border-color: var(--accent); background: var(--accent); color: #fff; }
    select, input { width: 100%; padding: 7px 10px; }
    main {
      width: min(1240px, calc(100% - 28px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { margin: 0; font-size: 26px; }
    h2 { margin: 0; font-size: 20px; }
    h3 { margin: 0; font-size: 16px; }
    p { line-height: 1.48; }
    .lede { margin: 6px 0 22px; color: var(--muted); }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .summary-card {
      min-height: 96px;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .summary-card span { display: block; color: var(--muted); font-size: 12px; font-weight: 750; }
    .summary-card strong { display: block; margin-top: 5px; font-size: 26px; }
    .summary-card small { display: block; margin-top: 3px; color: var(--muted); }
    .evidence-note {
      padding: 13px 15px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      line-height: 1.45;
    }
    .small-sample-banner {
      margin-top: 10px;
      padding: 13px 15px;
      border: 1px solid #e5c86d;
      border-left: 4px solid var(--amber);
      background: #fffaf0;
      line-height: 1.45;
    }
    .small-sample-banner strong { display: block; margin-bottom: 2px; }
    .tabs {
      position: sticky;
      z-index: 10;
      top: 0;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 18px 0 14px;
      padding: 8px 0;
      overflow-x: visible;
      background: var(--bg);
      box-shadow: 0 1px 0 var(--line);
    }
    .tab { flex: 0 1 auto; white-space: normal; }
    .view { display: none; }
    .view.active { display: block; }
    .band {
      margin-top: 12px;
      padding: 18px 0;
      border-top: 1px solid var(--line);
    }
    .overview-disclosure > summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 2px;
      cursor: pointer;
    }
    .overview-disclosure > summary h2 { display: inline; margin-right: 10px; }
    .overview-disclosure > summary span { color: var(--muted); }
    .overview-disclosure[open] > summary { margin-bottom: 6px; }
    .section-copy { margin: 5px 0 14px; color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    label span {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .table-wrap:focus-visible { border-color: var(--accent); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 1; background: #f7f9fb; color: var(--muted); font-size: 12px; text-transform: uppercase; }
    tbody tr:last-child td { border-bottom: 0; }
    .coverage-cell { min-width: 150px; }
    .coverage-track { width: 100%; height: 9px; margin-top: 5px; background: #edf1f4; border-radius: 4px; overflow: hidden; }
    .coverage-fill { height: 100%; background: var(--green); }
    .coverage-fill.limited { background: var(--amber); }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 23px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 850;
    }
    .badge.good { color: var(--green); background: var(--green-soft); }
    .badge.caution { color: var(--amber); background: var(--amber-soft); }
    .status-note {
      margin-top: 10px;
      padding: 14px 15px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--amber);
      border-radius: 8px;
      background: #fffaf0;
    }
    .status-note p { margin: 6px 0 0; color: var(--muted); }
    .chart-shell { padding: 10px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    canvas { display: block; width: 100%; aspect-ratio: 2.5 / 1; min-height: 280px; }
    canvas.tall-chart { aspect-ratio: auto; }
    .chart-legend {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 6px 14px;
      margin: 12px 0 0;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      list-style: none;
    }
    .chart-legend:empty { display: none; }
    .chart-legend li { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .chart-key {
      display: inline-grid;
      place-items: center;
      flex: 0 0 22px;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      color: #fff;
      background: var(--green);
      font-size: 11px;
      font-weight: 850;
    }
    .chart-summary { margin: 10px 0 0; padding: 10px 12px; border-left: 3px solid var(--accent); background: #f7f9fb; }
    .correlation-map-panel {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    .correlation-map-wrap {
      max-height: 760px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .correlation-map { width: max-content; min-width: 100%; border-collapse: separate; border-spacing: 0; }
    .correlation-map th, .correlation-map td { padding: 0; border: 0; }
    .correlation-map .correlation-corner {
      position: sticky;
      z-index: 5;
      top: 0;
      left: 0;
      min-width: 205px;
      padding: 9px 10px;
      background: #f4f7fa;
      text-align: left;
    }
    .correlation-column {
      position: sticky;
      z-index: 3;
      top: 0;
      width: 46px;
      height: 158px;
      background: #f4f7fa;
      vertical-align: bottom;
    }
    .correlation-column span {
      display: block;
      width: 46px;
      padding: 7px 5px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
    }
    .correlation-row-label {
      position: sticky;
      z-index: 2;
      left: 0;
      width: 205px;
      max-width: 205px;
      padding: 8px 10px !important;
      overflow: hidden;
      background: #f7f9fb;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .correlation-column.selected, .correlation-row-label.selected { color: var(--accent); background: var(--accent-soft); }
    .correlation-cell, .correlation-na, .correlation-diagonal {
      width: 46px;
      min-width: 46px;
      height: 46px;
      min-height: 46px;
      padding: 0;
      border: 1px solid rgba(255,255,255,.82);
      border-radius: 0;
      font-size: 10px;
      font-weight: 850;
    }
    .correlation-cell:hover, .correlation-cell:focus-visible { position: relative; z-index: 1; outline: 3px solid #263746; outline-offset: -3px; }
    .correlation-cell.selected-pair { box-shadow: inset 0 0 0 3px #263746; }
    .correlation-na {
      display: grid;
      place-items: center;
      color: #74818c;
      background: repeating-linear-gradient(135deg, #eef1f4, #eef1f4 5px, #e0e5e9 5px, #e0e5e9 7px);
    }
    .correlation-diagonal { display: grid; place-items: center; color: #657381; background: #e8edf1; }
    .correlation-legend { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; margin: 10px 0 0; color: var(--muted); font-size: 11px; }
    .correlation-gradient { display: inline-block; flex: none; width: 190px; height: 13px; border-radius: 999px; background: linear-gradient(90deg, #2e6fa3, #fff 50%, #b04a70); box-shadow: inset 0 0 0 1px #cbd5dd; }
    .correlation-na-key { display: inline-block; width: 20px; height: 13px; margin-right: 4px; vertical-align: -2px; background: repeating-linear-gradient(135deg, #eef1f4, #eef1f4 4px, #dce2e7 4px, #dce2e7 6px); }
    .correlation-meaning-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 0 0 12px; }
    .correlation-meaning { padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: #fff; font-size: 12px; line-height: 1.4; }
    .correlation-meaning strong { display: block; margin-bottom: 2px; }
    .correlation-meaning.negative { border-left: 5px solid #2e6fa3; }
    .correlation-meaning.weak { border-left: 5px solid #d7dee5; }
    .correlation-meaning.positive { border-left: 5px solid #b04a70; }
    .pair-drilldown { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--line); scroll-margin-top: 90px; }
    .correlation-safeguard { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
    .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .chart-grid h3 { margin: 2px 4px 8px; }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .stat-card { padding: 11px 12px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .stat-card span { display: block; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .stat-card strong { display: block; margin-top: 4px; font-size: 17px; }
    .stat-card small { display: block; margin-top: 3px; color: var(--muted); }
    .category-legend { margin-top: 12px; }
    .measurement-help {
      margin: -2px 0 14px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      background: #f7f9fb;
      line-height: 1.45;
    }
    .measurement-help strong { display: block; margin-bottom: 3px; }
    .measurement-help p { margin: 0; color: var(--muted); }
    .measurement-help-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .chart-reference { display: flex; align-items: center; gap: 8px; margin: 2px 4px 8px; color: var(--muted); font-size: 12px; font-weight: 750; }
    .chart-reference-line { width: 28px; border-top: 2px dashed #735b24; }
    .profile-coverage { margin: 12px 0 18px; }
    .profile-section { margin-top: 22px; }
    .profile-section h3 { margin-bottom: 4px; }
    .profile-table a { color: var(--accent); font-weight: 750; }
    .eligibility-result {
      min-height: 98px;
      margin-top: 12px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .eligibility-result.caution { border-color: #e5c86d; background: #fffaf0; }
    .eligibility-result.good { border-color: #8ec8aa; background: #f3fbf6; }
    .technical-details {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .technical-details summary {
      padding: 11px 13px;
      cursor: pointer;
      font-weight: 800;
    }
    .technical-details > div { padding: 0 13px 13px; }
    .source-toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1.5fr) repeat(2, minmax(170px, 1fr));
      gap: 10px;
      margin: 14px 0 8px;
    }
    .source-count { margin: 0 0 12px; color: var(--muted); }
    .source-list { display: grid; gap: 12px; }
    .source-card {
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .source-card-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
    }
    .source-card-head h3 { font-size: 18px; }
    .source-card-head p { margin: 3px 0 0; color: var(--muted); }
    .source-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }
    .source-assessment { margin: 13px 0 6px; }
    .source-provenance { margin: 0; color: var(--muted); font-size: 13px; }
    .report-list { display: grid; gap: 8px; margin: 14px 0 0; padding: 0; list-style: none; }
    .report-item { padding: 11px 12px; border-left: 3px solid var(--accent); background: #f7f9fb; }
    .report-item a { color: var(--accent); font-weight: 800; }
    .report-meta { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }
    .report-evidence { margin: 7px 0 0; }
    .muted { color: var(--muted); }
    .empty { padding: 28px; text-align: center; color: var(--muted); }
    @media (max-width: 820px) {
      main { width: min(100% - 20px, 640px); padding-top: 20px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .controls { grid-template-columns: 1fr; }
      .source-toolbar { grid-template-columns: 1fr; }
      .source-card-head { flex-direction: column; }
      .source-badges { justify-content: flex-start; }
      .measurement-help-grid { grid-template-columns: 1fr; }
      .correlation-meaning-grid { grid-template-columns: 1fr; }
      .chart-grid { grid-template-columns: 1fr; }
      .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      canvas { min-height: 240px; aspect-ratio: 1.5 / 1; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body data-initial-view="__INITIAL_VIEW__">
  <a class="app-skip-link" href="#mainContent">Skip to data exploration</a>
  __APP_SITE_HEADER__
  <main id="mainContent" class="app-page-main" tabindex="-1">
    <h1>__PAGE_HEADING__</h1>
    <p class="lede">__PAGE_LEDE__</p>

    <div class="summary-grid">
      <div class="summary-card"><span>Registered injury cases</span><strong id="caseCount">-</strong><small>Independent analysis units</small></div>
      <div class="summary-card"><span>Analysed views</span><strong id="viewCount">-</strong><small>Not treated as extra cases</small></div>
      <div class="summary-card"><span>Measurement types</span><strong id="featureCount">-</strong><small>Available in at least one case summary</small></div>
      <div class="summary-card"><span>Current inference status</span><strong style="font-size:17px" id="inferenceStatus">Loading</strong><small>Evidence-gated</small></div>
    </div>

    <div class="evidence-note" id="analysisUnitNote">
      <div class="app-football-loader" role="status" aria-live="polite">
        <span class="app-loader-pitch" aria-hidden="true"><span class="app-loader-ball">⚽</span></span>
        <span class="app-loader-copy"><strong>Surveying the pitch…</strong><small>Loading case-level evidence, coverage, and comparison readiness.</small></span>
      </div>
    </div>
    <div class="small-sample-banner" id="smallSampleBanner">
      <strong>Checking the case-library size</strong>
      These tools will show only comparisons supported by independent injury cases.
    </div>

    <nav class="tabs" aria-label="Data exploration views" role="tablist">
      <button class="tab active" id="overviewTab" type="button" role="tab" aria-selected="true" aria-controls="overviewView" tabindex="0" data-view="overviewView">Overview</button>
      <button class="tab" id="sourcesTab" type="button" role="tab" aria-label="Sources / Injury Reports" aria-selected="false" aria-controls="sourcesView" tabindex="-1" data-view="sourcesView">Sources</button>
      <button class="tab" id="breakdownsTab" type="button" role="tab" aria-selected="false" aria-controls="breakdownsView" tabindex="-1" data-view="breakdownsView">Case breakdowns</button>
      <button class="tab" id="profilesTab" type="button" role="tab" aria-label="Height and weight" aria-selected="false" aria-controls="profilesView" tabindex="-1" data-view="profilesView">Body metrics</button>
      <button class="tab" id="distributionTab" type="button" role="tab" aria-selected="false" aria-controls="distributionView" tabindex="-1" data-view="distributionView">Compare cases</button>
      <button class="tab" id="relationshipTab" type="button" role="tab" aria-label="Explore measurement correlations" aria-selected="false" aria-controls="relationshipView" tabindex="-1" data-view="relationshipView">Measurement correlations</button>
      <button class="tab" id="testBuilderTab" type="button" role="tab" aria-label="Can these groups be compared?" aria-selected="false" aria-controls="testBuilderView" tabindex="-1" data-view="testBuilderView">Group readiness</button>
    </nav>

    <section class="view active" id="overviewView" role="tabpanel" aria-labelledby="overviewTab">
      <details class="band overview-disclosure" open>
        <summary><h2>Case Library</h2><span>Completed analyses grouped by registered injury event.</span></summary>
        <div class="table-wrap" role="region" aria-label="Case library table" tabindex="0"><table>
          <thead><tr><th>Case</th><th>Supplied injury metadata</th><th>Team / league / competition</th><th>Views</th><th>Measurement support</th></tr></thead>
          <tbody id="caseRows"><tr><td colspan="5" class="empty">Loading cases</td></tr></tbody>
        </table></div>
      </details>
      <details class="band overview-disclosure">
        <summary><h2>Feature Coverage</h2><span>Coverage by independent injury case and measurement.</span></summary>
        <p class="section-copy">Coverage is calculated across injury cases after selecting one evidence view per case and feature.</p>
        <div class="table-wrap" role="region" aria-label="Feature coverage table" tabindex="0"><table>
          <thead><tr><th>Measurement</th><th>Region</th><th>Geometry supported</th><th>Dynamic supported</th><th>Geometry unavailable / limited</th><th>Median frame coverage</th></tr></thead>
          <tbody id="featureRows"><tr><td colspan="6" class="empty">Loading measurements</td></tr></tbody>
        </table></div>
        <p class="section-copy">Unavailable means no numeric summary exists. Limited means a value exists but does not pass the relevant evidence gate. A recorded zero remains a numeric zero.</p>
      </details>
      <details class="band overview-disclosure" open>
        <summary><h2>Movement Similarity</h2><span>Current deterministic comparison readiness.</span></summary>
        <p class="section-copy">Availability of deterministic case-to-case similarity outputs in the current repository.</p>
        <div class="status-note" id="similarityStatus">Checking similarity artifacts.</div>
      </details>
      <details class="technical-details">
        <summary>Technical identifiers</summary>
        <div><div class="table-wrap"><table>
          <thead><tr><th>Type</th><th>Display name</th><th>Internal identifier</th></tr></thead>
          <tbody id="technicalIdentifierRows"></tbody>
        </table></div></div>
      </details>
    </section>

    <section class="view" id="breakdownsView" role="tabpanel" aria-labelledby="breakdownsTab">
      <div class="band">
        <h2>Case Breakdowns</h2>
        <p class="section-copy">See how many independent injury cases fall into each recorded category. These are case counts, not risk rates.</p>
        <div class="controls">
          <label><span>Break cases down by</span><select id="breakdownVariable">
            <option value="position_group">Player position</option>
            <option value="league">Domestic league</option>
            <option value="age_group">Age at injury</option>
            <option value="contact_mechanism">Contact mechanism</option>
            <option value="preferred_foot_knee_injured">Preferred-foot knee injured</option>
          </select></label>
        </div>
        <div class="chart-grid">
          <div class="chart-shell"><h3>Bar chart</h3><canvas id="breakdownBarCanvas" role="img" aria-label="Case counts by selected category shown as bars."></canvas></div>
          <div class="chart-shell"><h3>Pie chart</h3><canvas id="breakdownPieCanvas" role="img" aria-label="Case proportions by selected category shown as a pie."></canvas></div>
        </div>
        <p class="chart-summary" id="breakdownSummary"></p>
        <ul id="breakdownLegend" class="chart-legend category-legend" aria-label="Category counts"></ul>
      </div>
    </section>

    <section class="view" id="sourcesView" role="tabpanel" aria-labelledby="sourcesTab">
      <div class="band">
        <h2>Sources / Injury Reports</h2>
        <p class="section-copy">A case-by-case audit of the injury mechanism label. Explicit source wording is separated from interpretation, and contact-unspecified reports remain unclear.</p>
        <div class="evidence-note" id="mechanismMethodology">Loading the mechanism-classification method.</div>
        <div class="source-toolbar">
          <label><span>Find a player or source</span><input id="sourceSearch" type="search" placeholder="Search player, publisher, or report" autocomplete="off" /></label>
          <label><span>Mechanism</span><select id="sourceMechanismFilter">
            <option value="all">All mechanisms</option>
            <option value="non_contact">Non-contact</option>
            <option value="indirect_contact">Indirect contact</option>
            <option value="direct_contact">Direct contact</option>
            <option value="unclear">Unclear</option>
          </select></label>
          <label><span>Evidence assessment</span><select id="sourceEvidenceFilter">
            <option value="all">All evidence levels</option>
            <option value="verified_explicit">Explicitly verified</option>
            <option value="supported_interpretation">Supported interpretation</option>
            <option value="unverified">Unverified / unclear</option>
          </select></label>
        </div>
        <p class="source-count" id="sourceCount" role="status" aria-live="polite"></p>
        <div class="source-list" id="sourceCards"><div class="empty">Loading injury-report sources.</div></div>
      </div>
    </section>

    <section class="view" id="profilesView" role="tabpanel" aria-labelledby="profilesTab">
      <div class="band">
        <h2>Player Height &amp; Weight</h2>
        <p class="section-copy">Sourced player-profile values for the independent injury cases. These are descriptive profile fields—not measurements taken at the injury date—and missing or conflicting values are never estimated or plotted as zero.</p>
        <div class="summary-grid profile-coverage">
          <div class="summary-card"><span>Height available</span><strong id="heightCoverage">-</strong><small>of independent cases</small></div>
          <div class="summary-card"><span>Weight available</span><strong id="weightCoverage">-</strong><small>of independent cases</small></div>
          <div class="summary-card"><span>Complete pairs</span><strong id="biometricPairCoverage">-</strong><small>height and weight both sourced</small></div>
          <div class="summary-card"><span>Incomplete profiles</span><strong id="biometricMissingCount">-</strong><small>kept visible below</small></div>
        </div>

        <div class="profile-section">
          <h3>Height distribution</h3>
          <p class="section-copy">The box shows the middle 50%; the diamond is the mean; every named point is one player.</p>
          <div class="stats-grid" id="heightStats" aria-label="Height descriptive statistics"></div>
          <div class="chart-shell">
            <div class="chart-reference"><span class="chart-reference-line" aria-hidden="true"></span><span>Dashed line = median; box = middle 50%; diamond = mean</span></div>
            <canvas id="heightCanvas" class="tall-chart" role="img" aria-label="Named player height distribution; values are also listed in the table below."></canvas>
          </div>
          <p class="chart-summary" id="heightSummary"></p>
        </div>

        <div class="profile-section">
          <h3>Weight distribution</h3>
          <p class="section-copy">Only sourced numeric weights are included. Conflicting and unavailable weights remain visible in the audit table.</p>
          <div class="stats-grid" id="weightStats" aria-label="Weight descriptive statistics"></div>
          <div class="chart-shell">
            <div class="chart-reference"><span class="chart-reference-line" aria-hidden="true"></span><span>Dashed line = median; box = middle 50%; diamond = mean</span></div>
            <canvas id="weightCanvas" class="tall-chart" role="img" aria-label="Named player weight distribution; values are also listed in the table below."></canvas>
          </div>
          <p class="chart-summary" id="weightSummary"></p>
        </div>

        <div class="profile-section">
          <h3>Height and weight together</h3>
          <p class="section-copy">Each numbered point is one player with a complete sourced pair. This is a descriptive profile view, not a relationship with injury risk.</p>
          <div class="chart-shell">
            <canvas id="biometricScatterCanvas" role="img" aria-label="Height versus weight for players with complete sourced profiles."></canvas>
            <ol id="biometricScatterLegend" class="chart-legend" aria-label="Player labels for numbered height and weight points"></ol>
          </div>
          <p class="chart-summary" id="biometricScatterSummary"></p>
        </div>

        <div class="profile-section">
          <h3>Source and missing-data audit</h3>
          <div class="table-wrap profile-table" role="region" aria-label="Height and weight source audit" tabindex="0"><table>
            <thead><tr><th>Player</th><th>Height</th><th>Weight</th><th>Selected source</th><th>Audit note</th></tr></thead>
            <tbody id="biometricRows"></tbody>
          </table></div>
        </div>
      </div>
    </section>

    <section class="view" id="distributionView" role="tabpanel" aria-labelledby="distributionTab">
      <div class="band">
        <h2>Compare Cases</h2>
        <p class="section-copy">Compare one supported measurement across registered injury cases. Every point is one injury case, never a video frame or replay.</p>
        <div class="controls">
          <label><span>Measurement</span><select id="distributionFeature"></select></label>
          <label><span>How to summarise each case</span><select id="distributionStatistic"></select></label>
        </div>
        <div class="measurement-help" id="distributionFeatureHelp" aria-live="polite"></div>
        <div class="stats-grid" id="distributionStats" aria-label="Descriptive statistics for supported case values"></div>
        <div class="chart-shell">
          <div class="chart-reference"><span class="chart-reference-line" aria-hidden="true"></span><span>Dashed line = median; box = middle 50%; diamond = mean</span></div>
          <canvas id="distributionCanvas" class="tall-chart" role="img" aria-label="Named case comparison chart; values are also listed in the table below."></canvas>
        </div>
        <p class="chart-summary" id="distributionSummary"></p>
        <div class="table-wrap"><table>
          <thead><tr><th>Case</th><th>Value</th><th>Support</th><th>Evidence view</th><th>Relevant coverage</th></tr></thead>
          <tbody id="distributionRows"></tbody>
        </table></div>
      </div>
    </section>

    <section class="view" id="relationshipView" role="tabpanel" aria-labelledby="relationshipTab">
      <div class="band">
        <h2>Measurement Correlation Map</h2>
        <p class="section-copy">Explore how supported measurements vary together across the complete analysed case library. Every paired value represents one independent injury event, never an extra replay, frame, or camera view.</p>
        <div class="controls">
          <label><span>How to summarise each case</span><select id="correlationStatistic"></select></label>
          <label><span>Measurement group</span><select id="correlationGroup"><option value="all">All measurement groups</option></select></label>
          <label><span>Measurement order</span><select id="correlationOrder"><option value="anatomical">Group by movement area</option><option value="association">Place related measurements together</option></select></label>
        </div>
        <div class="correlation-meaning-grid" aria-label="How to read positive and negative correlations">
          <div class="correlation-meaning negative"><strong>Negative correlation</strong>As one supported measurement increases across cases, the other tends to decrease.</div>
          <div class="correlation-meaning weak"><strong>Weak or no monotonic correlation</strong>The two measurements show little consistent rank relationship in this case library.</div>
          <div class="correlation-meaning positive"><strong>Positive correlation</strong>The two supported measurements tend to increase or decrease together across cases.</div>
        </div>
        <div class="correlation-map-panel">
          <div class="correlation-map-wrap" id="correlationMap" role="region" aria-label="Measurement correlation matrix" tabindex="0"></div>
          <div class="correlation-legend" aria-label="Correlation map colour legend"><span>−1</span><i class="correlation-gradient" aria-hidden="true"></i><span>+1</span><span><i class="correlation-na-key" aria-hidden="true"></i> Insufficient or unsuitable paired evidence</span></div>
          <p class="chart-summary" id="correlationMapSummary">Loading the independent case-level measurement relationships.</p>
          <p class="correlation-safeguard">This map reports descriptive Spearman rank correlation, not causation, injury risk, or a biomechanical mechanism. Missing and limited values are never replaced with zero. Some measurements are mathematically related, so strong cells must be interpreted alongside their definitions.</p>
        </div>

        <div class="pair-drilldown" id="relationshipDrilldown">
          <h3>Inspect one measurement pair</h3>
          <p class="section-copy">Choose a pair directly or select any available cell above to open its named-case scatter plot.</p>
          <div class="controls" style="grid-template-columns:repeat(2,minmax(0,1fr))">
          <label><span>First measurement</span><select id="relationshipX"></select></label>
          <label><span>Second measurement</span><select id="relationshipY"></select></label>
          </div>
          <div class="measurement-help-grid" id="relationshipFeatureHelp" aria-live="polite"></div>
          <div class="chart-shell">
            <canvas id="relationshipCanvas" role="img" aria-label="Two-measurement case comparison chart; values are also listed in the table below."></canvas>
            <ol id="relationshipLegend" class="chart-legend" aria-label="Case labels for numbered chart points"></ol>
          </div>
          <p class="chart-summary" id="relationshipSummary"></p>
          <div class="table-wrap"><table>
            <thead><tr><th>Case</th><th id="relationshipXHeading">First measurement</th><th id="relationshipYHeading">Second measurement</th><th>Comparison status</th></tr></thead>
            <tbody id="relationshipRows"></tbody>
          </table></div>
          <details class="technical-details">
            <summary>Technical relationship details</summary>
            <div id="relationshipTechnicalSummary"></div>
          </details>
        </div>
      </div>
    </section>

    <section class="view" id="testBuilderView" role="tabpanel" aria-labelledby="testBuilderTab">
      <div class="band">
        <h2>Can These Groups Be Compared?</h2>
        <p class="section-copy">Choose a measurement and a way to group the cases. The explorer will explain whether there are enough independent injury cases for a responsible comparison.</p>
        <div class="controls">
          <label><span>Measurement</span><select id="testFeature"></select></label>
          <label><span>How to summarise each case</span><select id="testStatistic"></select></label>
          <label><span>Compare groups by</span><select id="testGroup">
            <option value="contact_mechanism">Contact mechanism</option>
            <option value="preferred_foot_knee_injured">Preferred-foot knee injured</option>
          </select></label>
        </div>
        <div class="measurement-help" id="testFeatureHelp" aria-live="polite"></div>
        <button class="active" id="checkEligibility" type="button">Check whether comparison is possible</button>
        <div class="eligibility-result caution" id="eligibilityResult">
          Select an outcome and check whether the current evidence supports a group comparison.
        </div>
      </div>
      <div class="band">
        <h2>What the Current Library Supports</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>Analysis</th><th>Status</th><th>Reason</th></tr></thead>
          <tbody id="readinessRows"></tbody>
        </table></div>
      </div>
      <details class="technical-details">
        <summary>Technical details: possible statistical methods</summary>
        <div><div class="table-wrap"><table>
          <thead><tr><th>Research question</th><th>Candidate methods</th><th>Required reporting</th></tr></thead>
          <tbody id="testFamilyRows"></tbody>
        </table></div></div>
      </details>
    </section>
  </main>
  <script>
    const app = {data: null};
    const $ = id => document.getElementById(id);
    const statisticLabels = {
      mean: "Mean",
      range: "Observed range",
      pre_late_change: "Early-to-late change",
      geometry_completeness: "Geometry coverage",
      dynamic_completeness: "Dynamic coverage",
    };
    const breakdownLabels = {
      position_group: "player position",
      league: "domestic league",
      age_group: "age at injury",
      contact_mechanism: "contact mechanism",
      preferred_foot_knee_injured: "whether the preferred-foot knee was injured",
    };
    const chartColours = ["#215f9a", "#18744a", "#b26a1b", "#7a4fa3", "#b7435c", "#377f8c", "#6d7f32", "#795548"];

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
      }[char]));
    }

    function percent(value) {
      return valueAvailable(value) ? `${Math.round(Number(value) * 100)}%` : "Unavailable";
    }

    function valueAvailable(value) {
      return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
    }

    function featureLabel(name) {
      return app.data.features.find(item => item.feature_name === name)?.label || name;
    }

    function featureInfo(name) {
      return app.data.features.find(item => item.feature_name === name) || {
        label: name,
        description: "A supported two-dimensional measurement from the video.",
      };
    }

    function measurementHelpHtml(name, prefix) {
      const feature = featureInfo(name);
      return `<div class="measurement-help"><strong>${escapeHtml(`${prefix}: ${feature.label}`)}</strong><p>${escapeHtml(feature.description)}</p></div>`;
    }

    function renderMeasurementHelp() {
      const distribution = featureInfo($("distributionFeature").value);
      $("distributionFeatureHelp").innerHTML = `<strong>${escapeHtml(distribution.label)}</strong><p>${escapeHtml(distribution.description)}</p>`;
      $("relationshipFeatureHelp").innerHTML = measurementHelpHtml($("relationshipX").value, "First") + measurementHelpHtml($("relationshipY").value, "Second");
      const testFeature = featureInfo($("testFeature").value);
      $("testFeatureHelp").innerHTML = `<strong>${escapeHtml(testFeature.label)}</strong><p>${escapeHtml(testFeature.description)}</p>`;
    }

    function readableValue(value, fallback = "Not recorded") {
      const text = String(value ?? "").trim();
      if (!text || ["unknown", "uncertain", "unclear"].includes(text.toLowerCase())) return fallback;
      return text.replaceAll("_", " ");
    }

    function mechanismIsResolved(value) {
      return !["", "unknown", "uncertain", "unclear"].includes(String(value || "").toLowerCase());
    }

    function mechanismLabel(value) {
      return mechanismIsResolved(value)
        ? String(value).replaceAll("_", " ")
        : "Unclear";
    }

    function evidenceStatusLabel(value) {
      return ({
        verified_explicit: "Explicit source wording",
        supported_interpretation: "Supported interpretation",
        unverified: "Unverified mechanism",
      })[value] || "Not reviewed";
    }

    function evidenceReasonLabel(value) {
      const raw = String(value || "").trim();
      const upper = raw.toUpperCase();
      if (upper.includes("TARGET_IDENTITY_UNCERTAIN")) return "The athlete’s identity could not be verified reliably in this interval.";
      if (upper.includes("TARGET_NOT_FOUND")) return "The athlete could not be detected reliably in this view.";
      if (upper.includes("INVALID_TRACK_SEGMENT")) return "Athlete tracking was not continuous enough for this measurement.";
      if (upper.includes("TARGET_NOT_VISIBLE")) return "The athlete is not visible in this interval.";
      if (upper.includes("LOW_CONFIDENCE") || upper.includes("REQUIRED_LANDMARK")) return "The required body landmarks were not reliable enough for this measurement.";
      if (!raw) return "The evidence needed for this measurement is not available.";
      return raw.replaceAll("_", " ").replace(/\b[A-Z]{2,}\b/g, word => word.toLowerCase());
    }

    function changeStatusLabel(event) {
      const current = mechanismLabel(event.contact_mechanism);
      const previous = event.previous_contact_mechanism
        ? mechanismLabel(event.previous_contact_mechanism)
        : null;
      if (event.mechanism_change_status === "changed") return `Changed from ${previous || "not recorded"} to ${current}`;
      if (event.mechanism_change_status === "unchanged") return `Retained as ${current}`;
      if (event.mechanism_change_status === "newly_classified") return `Newly classified as ${current}`;
      return "No verifiable mechanism found";
    }

    function shortCaseLabel(name) {
      const words = String(name || "Case").trim().split(/\s+/).filter(Boolean);
      if (words.length < 2) return words[0] || "Case";
      return `${words[0].slice(0, 1)}. ${words.at(-1)}`;
    }

    function unitLabel(feature, statistic) {
      if (statistic.endsWith("completeness")) return "Coverage (%)";
      if (feature.includes("_deg")) return "Degrees";
      if (feature.includes("normalized")) return "Body-scale units";
      if (feature.endsWith("_px") || feature.includes("knee_line_deviation")) return "Pixels";
      return "Projected value";
    }

    function formatValue(value, feature, statistic) {
      if (!valueAvailable(value)) return "Unavailable";
      if (statistic.endsWith("completeness")) return percent(value);
      const isDegrees = feature.includes("_deg");
      const isNormalized = feature.includes("normalized");
      const isPixels = feature.endsWith("_px") || feature.includes("knee_line_deviation");
      const suffix = isDegrees ? "°" : isNormalized ? " body scale" : isPixels ? " px" : "";
      return `${Number(value).toFixed(isNormalized ? 2 : 1)}${suffix}`;
    }

    function statisticEligibility(record, statistic) {
      if (statistic === "pre_late_change") return Boolean(record.dynamic_analytics_eligible);
      if (statistic.endsWith("completeness")) return true;
      return Boolean(record.geometry_analytics_eligible);
    }

    function measurementStatus(record, statistic) {
      if (!record || !valueAvailable(record[statistic])) return "Unavailable";
      if (!statisticEligibility(record, statistic)) return "Limited evidence";
      return "Supported";
    }

    function selectedRecords(feature, statistic) {
      return app.data.records.filter(record =>
        record.feature_name === feature &&
        valueAvailable(record[statistic]) &&
        statisticEligibility(record, statistic)
      );
    }

    function median(values) {
      const sorted = values.filter(valueAvailable).map(Number).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    }

    function mean(values) {
      return values.length ? values.reduce((sum, value) => sum + Number(value), 0) / values.length : null;
    }

    function sampleStandardDeviation(values) {
      if (values.length < 2) return null;
      const centre = mean(values);
      return Math.sqrt(values.reduce((sum, value) => sum + (Number(value) - centre) ** 2, 0) / (values.length - 1));
    }

    function quantile(values, probability) {
      const sorted = values.map(Number).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const position = (sorted.length - 1) * probability;
      const lower = Math.floor(position);
      const fraction = position - lower;
      return sorted[lower + 1] === undefined
        ? sorted[lower]
        : sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
    }

    function categoryLabel(value, variable = "") {
      if (variable === "preferred_foot_knee_injured") {
        if (value === true || value === "true") return "Preferred-foot knee injured";
        if (value === false || value === "false") return "Other knee injured";
      }
      const text = readableValue(value, "Not recorded");
      return text === text.toLowerCase()
        ? text.replace(/(^|[\s-])([a-z])/g, (_, boundary, letter) => boundary + letter.toUpperCase())
        : text;
    }

    function setSelectOptions(select, items, valueKey, labelKey) {
      select.innerHTML = items.map(item =>
        `<option value="${escapeHtml(item[valueKey])}">${escapeHtml(item[labelKey])}</option>`
      ).join("");
    }

    function renderSummary() {
      const summary = app.data.summary;
      $("caseCount").textContent = summary.analysed_case_count;
      $("viewCount").textContent = summary.analysed_view_count;
      $("featureCount").textContent = summary.feature_count;
      $("inferenceStatus").textContent = "Descriptive only";
      $("analysisUnitNote").innerHTML = `<strong>Statistical unit:</strong> ${escapeHtml(app.data.analysis_unit_note)} ` +
        `Current confirmatory tests remain unavailable until the case library and analysis plan support them.`;
      const count = Number(summary.analysed_case_count || 0);
      $("smallSampleBanner").innerHTML = count < 10
        ? `<strong>Small case library: ${count} independent injury case${count === 1 ? "" : "s"}</strong>` +
          `Use these views to inspect named cases. Relationship coefficients and group tests remain exploratory, and comparisons are withheld when the independent-case counts are insufficient.`
        : `<strong>Exploratory case library: ${count} independent injury cases</strong>` +
          `Results remain descriptive. Review evidence coverage and the named contributing cases before interpreting any pattern.`;
    }

    function renderOverview() {
      $("caseRows").innerHTML = app.data.events.map(event => {
        const contact = String(event.contact_mechanism || "unknown");
        const knownContact = mechanismIsResolved(contact);
        const injuredSide = readableValue(event.injured_side);
        const injuryParts = [
          injuredSide === "Not recorded" ? "Injured knee not recorded" : `${injuredSide} knee`,
          event.injury_date || null,
          event.match_minute ? `minute ${event.match_minute}` : null,
        ].filter(Boolean);
        const teamCompetition = [readableValue(event.team), readableValue(event.league), readableValue(event.competition)]
          .filter(value => value !== "Not recorded");
        const preferredSide = event.preferred_foot_knee_injured === true
          ? "preferred-foot knee injured"
          : event.preferred_foot_knee_injured === false
            ? "other knee injured"
            : "preferred-foot comparison unavailable";
        return `<tr>
          <td><strong>${escapeHtml(event.player_name)}</strong><br><span class="muted">Position: ${escapeHtml(readableValue(event.position_group))}</span></td>
          <td>${injuryParts.map(escapeHtml).join(" · ")}<br><span class="badge ${knownContact ? "good" : "caution"}">${escapeHtml(mechanismLabel(contact))}</span><br><span class="muted">${escapeHtml(categoryLabel(event.preferred_foot, ""))} foot · ${escapeHtml(preferredSide)}</span></td>
          <td>${teamCompetition.length ? teamCompetition.map(escapeHtml).join("<br>") : '<span class="muted">Not recorded</span>'}</td>
          <td>${event.analysed_view_count}</td>
          <td><strong>Geometry ${event.geometry_eligible_feature_count} / ${event.feature_count}</strong><br><span class="muted">Dynamics ${event.dynamic_eligible_feature_count} / ${event.feature_count}<br>Median frame coverage: geometry ${percent(event.median_geometry_completeness)} · dynamics ${percent(event.median_dynamic_completeness)}</span></td>
        </tr>`;
      }).join("") || '<tr><td colspan="5" class="empty">No completed case summaries are available.</td></tr>';

      $("featureRows").innerHTML = app.data.features.map(feature => {
        const limited = !valueAvailable(feature.median_geometry_completeness) || feature.median_geometry_completeness < 0.6;
        const unavailableLimited = feature.unavailable_case_count + feature.unsupported_case_count;
        const geometryWidth = valueAvailable(feature.median_geometry_completeness)
          ? Math.round(feature.median_geometry_completeness * 100) : 0;
        return `<tr>
          <td><strong>${escapeHtml(feature.label)}</strong><br><span class="muted">${escapeHtml(feature.description)}</span></td>
          <td>${escapeHtml(feature.body_region.replaceAll("_", " "))}</td>
          <td>${feature.supported_case_count} / ${feature.relevant_case_count}</td>
          <td>${feature.dynamic_supported_case_count} / ${feature.relevant_case_count}</td>
          <td>${unavailableLimited} / ${feature.relevant_case_count}<br><span class="muted">${feature.unavailable_case_count} unavailable · ${feature.unsupported_case_count} limited</span></td>
          <td class="coverage-cell">Geometry ${percent(feature.median_geometry_completeness)}<div class="coverage-track"><div class="coverage-fill ${limited ? "limited" : ""}" style="width:${geometryWidth}%"></div></div><span class="muted">Dynamics ${percent(feature.median_dynamic_completeness)}</span></td>
        </tr>`;
      }).join("") || '<tr><td colspan="6" class="empty">No measurement summaries are available.</td></tr>';

      const similarity = app.data.similarity;
      $("similarityStatus").innerHTML = `<h3><span class="badge ${similarity.available ? "good" : "caution"}">${escapeHtml(similarity.status.replaceAll("_", " "))}</span></h3>` +
        `<p>${escapeHtml(similarity.reason)} Found ${similarity.comparable_case_count || 0} comparable case${similarity.comparable_case_count === 1 ? "" : "s"} and ${similarity.pairwise_output_count} supported pairwise ranking${similarity.pairwise_output_count === 1 ? "" : "s"}.</p>` +
        `<p>${escapeHtml(similarity.scientific_note)}</p>`;
      const identifiers = [
        ...app.data.events.map(event => ["Case", event.player_name, event.case_id]),
        ...app.data.features.map(feature => ["Measurement", feature.label, feature.feature_name]),
      ];
      $("technicalIdentifierRows").innerHTML = identifiers.map(([type, label, identifier]) => `<tr>
        <td>${escapeHtml(type)}</td><td>${escapeHtml(label)}</td><td><code>${escapeHtml(identifier)}</code></td>
      </tr>`).join("");
    }

    function renderSources() {
      const query = $("sourceSearch").value.trim().toLowerCase();
      const mechanismFilter = $("sourceMechanismFilter").value;
      const evidenceFilter = $("sourceEvidenceFilter").value;
      const events = app.data.events
        .filter(event => {
          const sources = Array.isArray(event.mechanism_sources) ? event.mechanism_sources : [];
          const searchable = [
            event.player_name,
            event.team,
            event.league,
            event.competition,
            event.preferred_foot_source,
            event.mechanism_rationale,
            event.mechanism_investigation_note,
            ...sources.flatMap(source => [source.title, source.publisher, source.evidence]),
          ].join(" ").toLowerCase();
          const mechanism = mechanismIsResolved(event.contact_mechanism)
            ? event.contact_mechanism
            : "unclear";
          return (!query || searchable.includes(query)) &&
            (mechanismFilter === "all" || mechanism === mechanismFilter) &&
            (evidenceFilter === "all" || event.mechanism_verification_status === evidenceFilter);
        })
        .sort((left, right) => String(left.player_name).localeCompare(String(right.player_name)));

      $("sourceCount").textContent = `${events.length} of ${app.data.events.length} analysed injury events shown`;
      $("sourceCards").innerHTML = events.map(event => {
        const resolved = mechanismIsResolved(event.contact_mechanism);
        const sources = Array.isArray(event.mechanism_sources) ? event.mechanism_sources : [];
        const registeredIds = Array.isArray(event.registered_case_ids) ? event.registered_case_ids : [];
        const preferredFootSource = event.preferred_foot_source_url
          ? `<a href="${escapeHtml(event.preferred_foot_source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(event.preferred_foot_source || "preferred-foot source")}</a>`
          : escapeHtml(event.preferred_foot_source || "source not recorded");
        const sourceItems = sources.map(source => `<li class="report-item">
          <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title || "Open source")}</a>
          <span class="report-meta">${[source.publisher, source.published_date, readableValue(source.source_type, "")].filter(Boolean).map(escapeHtml).join(" · ")}</span>
          <p class="report-evidence"><strong>Evidence:</strong> ${escapeHtml(source.evidence || "The source confirms the injury but does not describe the mechanism.")}</p>
        </li>`).join("") || '<li class="report-item">No public injury-report source is recorded for this case.</li>';
        return `<article class="source-card">
          <div class="source-card-head">
            <div><h3>${escapeHtml(event.player_name)}</h3><p>${[event.injury_date, event.team, event.competition].filter(Boolean).map(escapeHtml).join(" · ")}</p></div>
            <div class="source-badges">
              <span class="badge ${resolved ? "good" : "caution"}">${escapeHtml(mechanismLabel(event.contact_mechanism))}</span>
              <span class="badge ${event.mechanism_verification_status === "verified_explicit" ? "good" : "caution"}">Source evidence: ${escapeHtml(readableValue(event.mechanism_confidence, "not reviewed"))}</span>
            </div>
          </div>
          <p class="source-assessment"><strong>${escapeHtml(changeStatusLabel(event))}.</strong> ${escapeHtml(event.mechanism_rationale || "No mechanism assessment is available.")}</p>
          <p class="source-provenance">${escapeHtml(evidenceStatusLabel(event.mechanism_verification_status))} · Evidence basis: ${escapeHtml(readableValue(event.mechanism_evidence_basis, "not recorded"))}${registeredIds.length > 1 ? ` · ${registeredIds.length} registered views of this event` : ""}</p>
          <p class="source-provenance"><strong>Preferred foot:</strong> ${escapeHtml(categoryLabel(event.preferred_foot, ""))} · ${preferredFootSource}${event.ea_fc_audit_status === "not_listed" ? " · EAFC record unavailable" : ""}</p>
          ${event.mechanism_investigation_status === "needs_further_investigation" ?
            `<p class="source-assessment"><span class="badge caution">Needs further investigation</span> ${escapeHtml(event.mechanism_investigation_note || "An additional independent review or clearer angle is still needed.")}</p>` : ""}
          <ul class="report-list">${sourceItems}</ul>
        </article>`;
      }).join("") || '<div class="empty">No injury reports match these filters.</div>';
    }

    function renderMechanismMethodology() {
      const method = app.data.mechanism_methodology || {};
      const review = app.data.mechanism_review || {};
      const definitions = method.definitions || {};
      const definitionText = ["non_contact", "indirect_contact", "direct_contact", "unclear"]
        .filter(key => definitions[key])
        .map(key => `<strong>${escapeHtml(mechanismLabel(key))}:</strong> ${escapeHtml(definitions[key])}`)
        .join(" ");
      const sourceLink = method.url
        ? `<a href="${escapeHtml(method.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(method.title || "classification method")}</a>`
        : "the recorded classification method";
      $("mechanismMethodology").innerHTML = `<strong>Classification method:</strong> ${definitionText}<br>` +
        `<span class="muted">Taxonomy source: ${sourceLink}. Review completed ${escapeHtml(review.reviewed_at || "date not recorded")}. ${escapeHtml(review.review_standard || "")}</span>`;
    }

    function populateControls() {
      const features = app.data.features.map(item => ({value: item.feature_name, label: item.label}));
      const statistics = app.data.statistics.map(value => ({value, label: statisticLabels[value] || value}));
      ["distributionFeature", "relationshipX", "relationshipY", "testFeature"].forEach(id =>
        setSelectOptions($(id), features, "value", "label")
      );
      ["distributionStatistic", "correlationStatistic", "testStatistic"].forEach(id =>
        setSelectOptions($(id), statistics, "value", "label")
      );
      const measurementGroups = [...new Set(app.data.features.map(item => item.body_region || "other"))]
        .sort((left, right) => String(left).localeCompare(String(right)));
      $("correlationGroup").innerHTML = '<option value="all">All measurement groups</option>' + measurementGroups.map(group =>
        `<option value="${escapeHtml(group)}">${escapeHtml(categoryLabel(group))}</option>`
      ).join("");
      if (features.length > 1) $("relationshipY").selectedIndex = 1;
      renderMeasurementHelp();
    }

    function breakdownCounts(variable) {
      const counts = new Map();
      app.data.events.forEach(event => {
        let value = event[variable];
        if (variable === "contact_mechanism" && !mechanismIsResolved(value)) value = "unclear";
        if (value === null || value === undefined || value === "") value = "unknown";
        const key = String(value);
        counts.set(key, (counts.get(key) || 0) + 1);
      });
      return [...counts.entries()]
        .map(([value, count]) => ({value, label: categoryLabel(value, variable), count}))
        .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
    }

    function renderBreakdowns() {
      const variable = $("breakdownVariable").value;
      const rows = breakdownCounts(variable);
      const total = rows.reduce((sum, row) => sum + row.count, 0);
      const barCanvas = $("breakdownBarCanvas");
      barCanvas.style.height = `${Math.max(300, rows.length * 48 + 90)}px`;
      const bar = canvasContext(barCanvas);
      bar.context.clearRect(0, 0, bar.width, bar.height);
      const barMargin = {left: Math.min(190, Math.max(118, bar.width * 0.3)), right: 28, top: 20, bottom: 44};
      const maxCount = Math.max(...rows.map(row => row.count), 1);
      const plotWidth = bar.width - barMargin.left - barMargin.right;
      const rowHeight = (bar.height - barMargin.top - barMargin.bottom) / Math.max(rows.length, 1);
      rows.forEach((row, index) => {
        const y = barMargin.top + index * rowHeight + rowHeight * 0.18;
        const height = rowHeight * 0.64;
        const width = row.count / maxCount * plotWidth;
        bar.context.fillStyle = chartColours[index % chartColours.length];
        bar.context.fillRect(barMargin.left, y, width, height);
        bar.context.fillStyle = "#1f2a33";
        bar.context.font = "12px system-ui";
        bar.context.textAlign = "right";
        bar.context.fillText(row.label, barMargin.left - 9, y + height / 2 + 4, barMargin.left - 18);
        bar.context.textAlign = "left";
        bar.context.font = "800 12px system-ui";
        bar.context.fillText(String(row.count), Math.min(barMargin.left + width + 7, bar.width - 18), y + height / 2 + 4);
      });
      bar.context.fillStyle = "#617080";
      bar.context.textAlign = "center";
      bar.context.font = "700 12px system-ui";
      bar.context.fillText("Independent injury cases", barMargin.left + plotWidth / 2, bar.height - 13);

      const pie = canvasContext($("breakdownPieCanvas"));
      pie.context.clearRect(0, 0, pie.width, pie.height);
      const radius = Math.min(pie.width, pie.height) * 0.31;
      const centreX = pie.width / 2;
      const centreY = pie.height / 2;
      let angle = -Math.PI / 2;
      rows.forEach((row, index) => {
        const sweep = total ? row.count / total * Math.PI * 2 : 0;
        pie.context.fillStyle = chartColours[index % chartColours.length];
        pie.context.beginPath();
        pie.context.moveTo(centreX, centreY);
        pie.context.arc(centreX, centreY, radius, angle, angle + sweep);
        pie.context.closePath();
        pie.context.fill();
        angle += sweep;
      });
      pie.context.fillStyle = "#fff";
      pie.context.beginPath();
      pie.context.arc(centreX, centreY, radius * 0.48, 0, Math.PI * 2);
      pie.context.fill();
      pie.context.fillStyle = "#1f2a33";
      pie.context.textAlign = "center";
      pie.context.font = "800 24px system-ui";
      pie.context.fillText(String(total), centreX, centreY + 2);
      pie.context.fillStyle = "#617080";
      pie.context.font = "11px system-ui";
      pie.context.fillText("cases", centreX, centreY + 20);

      $("breakdownSummary").textContent = `${total} independent injury cases grouped by ${breakdownLabels[variable]}. Counts describe this case library only; they do not estimate injury incidence or risk.`;
      $("breakdownLegend").innerHTML = rows.map((row, index) => `<li><span class="chart-key" style="background:${chartColours[index % chartColours.length]}">${row.count}</span><span>${escapeHtml(row.label)} · ${total ? Math.round(row.count / total * 100) : 0}%</span></li>`).join("");
    }

    function biometricStatusLabel(status) {
      return ({
        sourced: "Sourced",
        source_conflict: "Conflicting sources",
        not_found: "Not found",
      })[status] || readableValue(status, "Not reviewed");
    }

    function biometricFormat(value, unit) {
      return valueAvailable(value) ? `${Number(value).toFixed(0)} ${unit}` : "Unavailable";
    }

    function profileStatisticCard(label, value, detail = "") {
      return `<div class="stat-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}</div>`;
    }

    function renderProfileDistribution({field, statusField, label, unit, canvasId, statsId, summaryId, colour}) {
      const allEvents = [...app.data.events].sort((left, right) => String(left.player_name).localeCompare(String(right.player_name)));
      const records = allEvents
        .filter(event => valueAvailable(event[field]))
        .sort((left, right) => Number(left[field]) - Number(right[field]) || String(left.player_name).localeCompare(String(right.player_name)));
      const values = records.map(record => Number(record[field]));
      const canvas = $(canvasId);
      canvas.style.height = `${Math.max(420, records.length * 27 + 165)}px`;
      const {context, width, height} = canvasContext(canvas);
      if (!records.length) {
        drawNoData(context, width, height, `No sourced ${label.toLowerCase()} values.`);
        $(statsId).innerHTML = profileStatisticCard("Summary", "No sourced values");
        $(summaryId).textContent = `0 of ${allEvents.length} independent cases have a sourced ${label.toLowerCase()} value.`;
        return;
      }

      const average = mean(values);
      const centre = median(values);
      const standardDeviation = sampleStandardDeviation(values);
      const minimumRecord = records[0];
      const maximumRecord = records.at(-1);
      $(statsId).innerHTML = [
        profileStatisticCard("Mean", biometricFormat(average, unit)),
        profileStatisticCard("Median", biometricFormat(centre, unit)),
        profileStatisticCard("Sample SD", standardDeviation === null ? "Needs 2+ players" : biometricFormat(standardDeviation, unit)),
        profileStatisticCard("Minimum", biometricFormat(minimumRecord[field], unit), minimumRecord.player_name),
        profileStatisticCard("Maximum", biometricFormat(maximumRecord[field], unit), maximumRecord.player_name),
      ].join("");

      context.clearRect(0, 0, width, height);
      const margin = {left: Math.min(220, Math.max(155, width * 0.25)), right: 28, top: 92, bottom: 58};
      const [minimum, maximum] = numericDomain(values);
      drawLinearTicks(context, width, height, margin, [minimum, maximum], "x", unit);
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const scaleX = value => margin.left + (Number(value) - minimum) / (maximum - minimum) * plotWidth;
      const q1 = quantile(values, 0.25);
      const q3 = quantile(values, 0.75);
      const boxY = 48;
      const minimumX = scaleX(Math.min(...values));
      const maximumX = scaleX(Math.max(...values));
      const q1X = scaleX(q1);
      const q3X = scaleX(q3);
      const medianX = scaleX(centre);
      const meanX = scaleX(average);

      context.save();
      context.strokeStyle = colour;
      context.fillStyle = colour === "#18744a" ? "#e9f6ef" : "#e9f2fb";
      context.lineWidth = 2;
      context.beginPath(); context.moveTo(minimumX, boxY); context.lineTo(maximumX, boxY); context.stroke();
      context.beginPath(); context.moveTo(minimumX, boxY - 9); context.lineTo(minimumX, boxY + 9); context.moveTo(maximumX, boxY - 9); context.lineTo(maximumX, boxY + 9); context.stroke();
      context.fillRect(q1X, boxY - 15, Math.max(q3X - q1X, 1), 30);
      context.strokeRect(q1X, boxY - 15, Math.max(q3X - q1X, 1), 30);
      context.beginPath(); context.moveTo(medianX, boxY - 15); context.lineTo(medianX, boxY + 15); context.stroke();
      context.fillStyle = "#18744a";
      context.beginPath(); context.moveTo(meanX, boxY - 8); context.lineTo(meanX + 8, boxY); context.lineTo(meanX, boxY + 8); context.lineTo(meanX - 8, boxY); context.closePath(); context.fill();
      context.restore();

      context.save();
      context.strokeStyle = "#735b24";
      context.setLineDash([6, 5]);
      context.beginPath(); context.moveTo(medianX, margin.top); context.lineTo(medianX, height - margin.bottom); context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#735b24";
      context.font = "700 11px system-ui";
      context.textAlign = "center";
      context.fillText("Median", medianX, margin.top + 17);
      context.restore();

      records.forEach((record, index) => {
        const x = scaleX(record[field]);
        const y = records.length === 1
          ? margin.top + plotHeight / 2
          : margin.top + index / (records.length - 1) * plotHeight;
        context.strokeStyle = "#d7e1e8";
        context.beginPath(); context.moveTo(margin.left, y); context.lineTo(x, y); context.stroke();
        context.fillStyle = record[statusField] === "source_conflict" ? "#b26a1b" : colour;
        context.beginPath(); context.arc(x, y, 5.5, 0, Math.PI * 2); context.fill();
        context.fillStyle = "#1f2a33";
        context.font = "12px system-ui";
        context.textAlign = "right";
        context.fillText(record.player_name, margin.left - 10, y + 4, margin.left - 24);
      });
      context.fillStyle = "#1f2a33";
      context.font = "700 13px system-ui";
      context.textAlign = "center";
      context.fillText(`${label} (${unit})`, margin.left + plotWidth / 2, height - 16);
      canvas.setAttribute("aria-label", `${label} box plot and named player points across ${records.length} independent injury cases. The dashed line is the median and the diamond is the mean.`);
      const omitted = allEvents.length - records.length;
      $(summaryId).textContent = `${records.length} of ${allEvents.length} independent cases have a sourced ${label.toLowerCase()} value; ${omitted} ${omitted === 1 ? "profile is" : "profiles are"} unavailable or conflicting and not plotted.`;
    }

    function renderBiometricScatter() {
      const allEvents = [...app.data.events].sort((left, right) => String(left.player_name).localeCompare(String(right.player_name)));
      const points = allEvents.filter(event => valueAvailable(event.height_cm) && valueAvailable(event.weight_kg));
      const canvas = $("biometricScatterCanvas");
      const {context, width, height} = canvasContext(canvas);
      if (!points.length) {
        drawNoData(context, width, height, "No complete height and weight pairs.");
        $("biometricScatterLegend").innerHTML = "";
        $("biometricScatterSummary").textContent = `0 of ${allEvents.length} independent cases have both sourced fields.`;
        return;
      }
      context.clearRect(0, 0, width, height);
      const margin = drawAxes(context, width, height, "Height (cm)", "Weight (kg)");
      const xDomain = numericDomain(points.map(point => Number(point.height_cm)));
      const yDomain = numericDomain(points.map(point => Number(point.weight_kg)));
      drawLinearTicks(context, width, height, margin, xDomain, "x", "cm");
      drawLinearTicks(context, width, height, margin, yDomain, "y", "kg");
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      points.forEach((point, index) => {
        const x = margin.left + (Number(point.height_cm) - xDomain[0]) / (xDomain[1] - xDomain[0]) * plotWidth;
        const y = margin.top + (yDomain[1] - Number(point.weight_kg)) / (yDomain[1] - yDomain[0]) * plotHeight;
        context.fillStyle = "#18744a";
        context.beginPath(); context.arc(x, y, 10, 0, Math.PI * 2); context.fill();
        context.fillStyle = "#fff";
        context.font = "800 10px system-ui";
        context.textAlign = "center";
        context.fillText(String(index + 1), x, y + 3.5);
      });
      $("biometricScatterLegend").innerHTML = points.map((point, index) => (
        `<li><span class="chart-key">${index + 1}</span><span>${escapeHtml(point.player_name)} · ${biometricFormat(point.height_cm, "cm")} · ${biometricFormat(point.weight_kg, "kg")}</span></li>`
      )).join("");
      $("biometricScatterSummary").textContent = `${points.length} of ${allEvents.length} independent cases have a complete sourced pair. The chart describes the player profiles in this library and does not estimate injury risk.`;
    }

    function biometricTableCell(event, field, statusField, unit) {
      const status = String(event[statusField] || "not_found");
      const available = valueAvailable(event[field]);
      return `${available ? `<strong>${escapeHtml(biometricFormat(event[field], unit))}</strong><br>` : ""}<span class="badge ${available && status === "sourced" ? "good" : "caution"}">${escapeHtml(biometricStatusLabel(status))}</span>`;
    }

    function renderProfiles() {
      const events = [...app.data.events].sort((left, right) => String(left.player_name).localeCompare(String(right.player_name)));
      const heightCount = events.filter(event => valueAvailable(event.height_cm)).length;
      const weightCount = events.filter(event => valueAvailable(event.weight_kg)).length;
      const pairCount = events.filter(event => valueAvailable(event.height_cm) && valueAvailable(event.weight_kg)).length;
      const incompleteCount = events.length - pairCount;
      $("heightCoverage").textContent = `${heightCount} / ${events.length}`;
      $("weightCoverage").textContent = `${weightCount} / ${events.length}`;
      $("biometricPairCoverage").textContent = `${pairCount} / ${events.length}`;
      $("biometricMissingCount").textContent = incompleteCount;

      renderProfileDistribution({
        field: "height_cm", statusField: "height_verification_status", label: "Height", unit: "cm",
        canvasId: "heightCanvas", statsId: "heightStats", summaryId: "heightSummary", colour: "#215f9a",
      });
      renderProfileDistribution({
        field: "weight_kg", statusField: "weight_verification_status", label: "Weight", unit: "kg",
        canvasId: "weightCanvas", statsId: "weightStats", summaryId: "weightSummary", colour: "#18744a",
      });
      renderBiometricScatter();

      $("biometricRows").innerHTML = events.map(event => {
        const source = event.biometric_source_url
          ? `<a href="${escapeHtml(event.biometric_source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(event.biometric_source || "Open source")}</a>`
          : escapeHtml(event.biometric_source || "No source found");
        return `<tr>
          <td><strong>${escapeHtml(event.player_name)}</strong></td>
          <td>${biometricTableCell(event, "height_cm", "height_verification_status", "cm")}</td>
          <td>${biometricTableCell(event, "weight_kg", "weight_verification_status", "kg")}</td>
          <td>${source}</td>
          <td>${escapeHtml(event.biometric_note || "No source conflict noted in the selected-source audit.")}</td>
        </tr>`;
      }).join("") || '<tr><td colspan="5" class="empty">No player-profile metadata are available.</td></tr>';
    }

    function canvasContext(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(Math.round(rect.width * ratio), 600);
      canvas.height = Math.max(Math.round(rect.height * ratio), 280);
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {context, width: canvas.width / ratio, height: canvas.height / ratio};
    }

    function drawAxes(context, width, height, xLabel, yLabel) {
      const margin = {left: 66, right: 24, top: 24, bottom: 62};
      context.strokeStyle = "#aebbc7";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(margin.left, margin.top);
      context.lineTo(margin.left, height - margin.bottom);
      context.lineTo(width - margin.right, height - margin.bottom);
      context.stroke();
      context.fillStyle = "#1f2a33";
      context.font = "700 13px system-ui";
      context.textAlign = "center";
      context.fillText(xLabel, margin.left + (width - margin.left - margin.right) / 2, height - 18);
      context.save();
      context.translate(18, margin.top + (height - margin.top - margin.bottom) / 2);
      context.rotate(-Math.PI / 2);
      context.fillText(yLabel, 0, 0);
      context.restore();
      return margin;
    }

    function numericDomain(values) {
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || Math.max(Math.abs(max) * 0.2, 1);
      return [min - span * 0.12, max + span * 0.12];
    }

    function axisTickLabel(value, unit) {
      if (unit === "Coverage (%)") return `${Math.round(value * 100)}%`;
      if (unit === "Body-scale units") return value.toFixed(2);
      return value.toFixed(1);
    }

    function drawLinearTicks(context, width, height, margin, domain, axis, unit) {
      const [minimum, maximum] = domain;
      const count = 5;
      context.save();
      context.font = "11px system-ui";
      context.lineWidth = 1;
      for (let index = 0; index < count; index += 1) {
        const fraction = index / (count - 1);
        const value = minimum + fraction * (maximum - minimum);
        context.strokeStyle = "#e7edf2";
        context.beginPath();
        if (axis === "y") {
          const y = height - margin.bottom - fraction * (height - margin.top - margin.bottom);
          context.moveTo(margin.left, y);
          context.lineTo(width - margin.right, y);
          context.stroke();
          context.fillStyle = "#617080";
          context.textAlign = "right";
          context.fillText(axisTickLabel(value, unit), margin.left - 8, y + 4);
        } else {
          const x = margin.left + fraction * (width - margin.left - margin.right);
          context.moveTo(x, margin.top);
          context.lineTo(x, height - margin.bottom);
          context.stroke();
          context.fillStyle = "#617080";
          context.textAlign = "center";
          context.fillText(axisTickLabel(value, unit), x, height - margin.bottom + 17);
        }
      }
      context.restore();
    }

    function drawNoData(context, width, height, text) {
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#617080";
      context.font = "15px system-ui";
      context.textAlign = "center";
      context.fillText(text, width / 2, height / 2);
    }

    function renderDistribution() {
      if (!app.data.features.length) return;
      const feature = $("distributionFeature").value;
      const statistic = $("distributionStatistic").value;
      const allRecords = app.data.records.filter(record => record.feature_name === feature);
      const records = selectedRecords(feature, statistic).sort((a, b) => Number(a[statistic]) - Number(b[statistic]));
      const values = records.map(record => Number(record[statistic]));
      const centre = median(values);
      const average = mean(values);
      const standardDeviation = sampleStandardDeviation(values);
      const minimumRecord = records[0];
      const maximumRecord = records.at(-1);
      const statisticCard = (label, value, detail = "") => `<div class="stat-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}</div>`;
      $("distributionStats").innerHTML = records.length ? [
        statisticCard("Mean", formatValue(average, feature, statistic)),
        statisticCard("Median", formatValue(centre, feature, statistic)),
        statisticCard("Sample SD", standardDeviation === null ? "Needs 2+ cases" : formatValue(standardDeviation, feature, statistic)),
        statisticCard("Minimum", formatValue(minimumRecord[statistic], feature, statistic), minimumRecord.player_name),
        statisticCard("Maximum", formatValue(maximumRecord[statistic], feature, statistic), maximumRecord.player_name),
      ].join("") : statisticCard("Summary", "No supported values");
      const canvas = $("distributionCanvas");
      canvas.style.height = `${Math.max(420, records.length * 34 + 165)}px`;
      const {context, width, height} = canvasContext(canvas);
      if (!records.length) {
        drawNoData(context, width, height, "No supported case values for this selection.");
        $("distributionSummary").textContent = `0 of ${allRecords.length} independent cases have a supported value. Missing or limited measurements are listed below and are not plotted as zero.`;
      } else {
        context.clearRect(0, 0, width, height);
        const unit = unitLabel(feature, statistic);
        const margin = {left: Math.min(210, Math.max(145, width * 0.23)), right: 28, top: 92, bottom: 58};
        const [min, max] = numericDomain(values);
        drawLinearTicks(context, width, height, margin, [min, max], "x", unit);
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const medianX = margin.left + (centre - min) / (max - min) * plotWidth;
        const q1 = quantile(values, 0.25);
        const q3 = quantile(values, 0.75);
        const minimumX = margin.left + (Math.min(...values) - min) / (max - min) * plotWidth;
        const maximumX = margin.left + (Math.max(...values) - min) / (max - min) * plotWidth;
        const q1X = margin.left + (q1 - min) / (max - min) * plotWidth;
        const q3X = margin.left + (q3 - min) / (max - min) * plotWidth;
        const meanX = margin.left + (average - min) / (max - min) * plotWidth;
        const boxY = 48;
        context.save();
        context.strokeStyle = "#215f9a";
        context.fillStyle = "#e9f2fb";
        context.lineWidth = 2;
        context.beginPath(); context.moveTo(minimumX, boxY); context.lineTo(maximumX, boxY); context.stroke();
        context.beginPath(); context.moveTo(minimumX, boxY - 9); context.lineTo(minimumX, boxY + 9); context.moveTo(maximumX, boxY - 9); context.lineTo(maximumX, boxY + 9); context.stroke();
        context.fillRect(q1X, boxY - 15, Math.max(q3X - q1X, 1), 30);
        context.strokeRect(q1X, boxY - 15, Math.max(q3X - q1X, 1), 30);
        context.beginPath(); context.moveTo(medianX, boxY - 15); context.lineTo(medianX, boxY + 15); context.stroke();
        context.fillStyle = "#18744a";
        context.beginPath(); context.moveTo(meanX, boxY - 8); context.lineTo(meanX + 8, boxY); context.lineTo(meanX, boxY + 8); context.lineTo(meanX - 8, boxY); context.closePath(); context.fill();
        context.restore();
        context.save();
        context.strokeStyle = "#735b24";
        context.setLineDash([6, 5]);
        context.beginPath(); context.moveTo(medianX, margin.top); context.lineTo(medianX, height - margin.bottom); context.stroke();
        context.setLineDash([]);
        context.font = "700 11px system-ui";
        const medianLabel = "Median";
        const medianLabelWidth = context.measureText(medianLabel).width + 12;
        const medianLabelX = Math.min(Math.max(medianX - medianLabelWidth / 2, margin.left), width - margin.right - medianLabelWidth);
        context.fillStyle = "#fff8e1";
        context.fillRect(medianLabelX, margin.top + 3, medianLabelWidth, 20);
        context.fillStyle = "#735b24";
        context.textAlign = "center";
        context.fillText(medianLabel, medianLabelX + medianLabelWidth / 2, margin.top + 17);
        context.restore();
        records.forEach((record, index) => {
          const x = margin.left + (Number(record[statistic]) - min) / (max - min) * plotWidth;
          const y = records.length === 1
            ? margin.top + plotHeight / 2
            : margin.top + index / (records.length - 1) * plotHeight;
          context.strokeStyle = "#d7e1e8";
          context.beginPath(); context.moveTo(margin.left, y); context.lineTo(x, y); context.stroke();
          context.fillStyle = "#215f9a";
          context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.fill();
          context.fillStyle = "#1f2a33";
          context.font = "12px system-ui";
          context.textAlign = "right";
          context.fillText(record.player_name, margin.left - 10, y + 4, margin.left - 24);
        });
        context.fillStyle = "#1f2a33";
        context.font = "700 13px system-ui";
        context.textAlign = "center";
        context.fillText(unit, margin.left + plotWidth / 2, height - 16);
        canvas.setAttribute("aria-label", `${featureLabel(feature)} box plot and named case points across ${records.length} injury cases. The dashed line is the median and the diamond is the mean. Full values are listed below.`);
        const omitted = allRecords.length - records.length;
        $("distributionSummary").textContent = `${records.length} of ${allRecords.length} independent cases contribute supported values; ${omitted} ${omitted === 1 ? "case is" : "cases are"} unavailable or limited. The box covers the middle 50% of supported values. This is a named case comparison, not a population distribution.`;
      }
      const tableRows = [...allRecords].sort((left, right) => {
        const leftSupported = measurementStatus(left, statistic) === "Supported";
        const rightSupported = measurementStatus(right, statistic) === "Supported";
        if (leftSupported !== rightSupported) return leftSupported ? -1 : 1;
        if (leftSupported) return Number(left[statistic]) - Number(right[statistic]);
        return String(left.player_name).localeCompare(String(right.player_name));
      });
      $("distributionRows").innerHTML = tableRows.map(record => {
        const status = measurementStatus(record, statistic);
        const reason = evidenceReasonLabel(record.primary_rejection_reason || record.eligibility_reason);
        const coverage = statistic === "pre_late_change" || statistic === "dynamic_completeness"
          ? record.dynamic_completeness : record.geometry_completeness;
        return `<tr>
        <td><strong>${escapeHtml(record.player_name)}</strong></td>
        <td>${status === "Supported" ? formatValue(record[statistic], feature, statistic) : "Unavailable"}</td>
        <td><span class="badge ${status === "Supported" ? "good" : "caution"}">${escapeHtml(status)}</span>${status === "Supported" ? "" : `<br><span class="muted">${escapeHtml(reason)}</span>`}</td>
        <td>${escapeHtml(record.view_label)}${record.view_count > 1 ? `<br><span class="muted">selected from ${record.view_count} views</span>` : ""}</td>
        <td>${percent(coverage)}</td>
      </tr>`;
      }).join("") || '<tr><td colspan="5" class="empty">No case summaries are available for this measurement.</td></tr>';
    }

    function ranks(values) {
      const sorted = values.map((value, index) => ({value, index})).sort((a, b) => a.value - b.value);
      const result = new Array(values.length);
      let start = 0;
      while (start < sorted.length) {
        let end = start;
        while (end + 1 < sorted.length && sorted[end + 1].value === sorted[start].value) end += 1;
        const rank = (start + end + 2) / 2;
        for (let index = start; index <= end; index += 1) result[sorted[index].index] = rank;
        start = end + 1;
      }
      return result;
    }

    function correlation(left, right) {
      if (left.length < 2) return null;
      const meanLeft = left.reduce((sum, value) => sum + value, 0) / left.length;
      const meanRight = right.reduce((sum, value) => sum + value, 0) / right.length;
      const numerator = left.reduce((sum, value, index) => sum + (value - meanLeft) * (right[index] - meanRight), 0);
      const leftScale = Math.sqrt(left.reduce((sum, value) => sum + (value - meanLeft) ** 2, 0));
      const rightScale = Math.sqrt(right.reduce((sum, value) => sum + (value - meanRight) ** 2, 0));
      return leftScale && rightScale ? numerator / (leftScale * rightScale) : null;
    }

    const MINIMUM_CORRELATION_CASES = 5;

    function wrapSensitiveMean(featureName, statistic) {
      if (statistic !== "mean") return false;
      const name = String(featureName || "").toLowerCase();
      return name.includes("axis_angle_deg") || name.includes("line_angle_deg") ||
        (name.includes("orientation") && name.includes("deg"));
    }

    function supportedValuesByFeature(statistic) {
      const output = new Map(app.data.features.map(feature => [feature.feature_name, new Map()]));
      app.data.records.forEach(record => {
        if (!output.has(record.feature_name) || !valueAvailable(record[statistic]) || !statisticEligibility(record, statistic)) return;
        const caseId = record.statistical_unit_id || record.case_id;
        output.get(record.feature_name).set(caseId, Number(record[statistic]));
      });
      return output;
    }

    function measurementPairCorrelation(leftFeature, rightFeature, statistic, valuesByFeature) {
      const leftValues = valuesByFeature.get(leftFeature) || new Map();
      const rightValues = valuesByFeature.get(rightFeature) || new Map();
      const sharedCaseIds = [...leftValues.keys()].filter(caseId => rightValues.has(caseId));
      if (leftFeature === rightFeature) return {rho: 1, n: sharedCaseIds.length, reason: "same_measurement"};
      if (wrapSensitiveMean(leftFeature, statistic) || wrapSensitiveMean(rightFeature, statistic)) {
        return {rho: null, n: sharedCaseIds.length, reason: "circular_method_required"};
      }
      if (sharedCaseIds.length < MINIMUM_CORRELATION_CASES) {
        return {rho: null, n: sharedCaseIds.length, reason: "insufficient_paired_cases"};
      }
      const left = sharedCaseIds.map(caseId => leftValues.get(caseId));
      const right = sharedCaseIds.map(caseId => rightValues.get(caseId));
      const rho = correlation(ranks(left), ranks(right));
      return {rho, n: sharedCaseIds.length, reason: rho === null ? "constant_values" : "available"};
    }

    function correlationCellStyle(rho) {
      const strength = Math.abs(Number(rho));
      const target = rho < 0 ? [46, 111, 163] : [176, 74, 112];
      const blend = 0.12 + strength * 0.78;
      const colour = target.map(channel => Math.round(255 + (channel - 255) * blend));
      const text = strength >= 0.58 ? "#fff" : "#263746";
      return `background:rgb(${colour.join(",")});color:${text}`;
    }

    function correlationPairKey(leftFeature, rightFeature) {
      return [leftFeature, rightFeature].sort().join("||");
    }

    function associationOrderedFeatures(features, pairLookup) {
      if (features.length < 3) return features;
      const scoreFor = feature => features.reduce((sum, other) => {
        if (other.feature_name === feature.feature_name) return sum;
        const pair = pairLookup.get(correlationPairKey(feature.feature_name, other.feature_name));
        return sum + (pair?.rho === null || pair?.rho === undefined ? 0 : Math.abs(pair.rho));
      }, 0);
      const remaining = [...features];
      remaining.sort((left, right) => scoreFor(right) - scoreFor(left) || left.label.localeCompare(right.label));
      const ordered = [remaining.shift()];
      while (remaining.length) {
        const previous = ordered.at(-1);
        remaining.sort((left, right) => {
          const leftPair = pairLookup.get(correlationPairKey(previous.feature_name, left.feature_name));
          const rightPair = pairLookup.get(correlationPairKey(previous.feature_name, right.feature_name));
          const leftScore = leftPair?.rho === null || leftPair?.rho === undefined ? -1 : Math.abs(leftPair.rho);
          const rightScore = rightPair?.rho === null || rightPair?.rho === undefined ? -1 : Math.abs(rightPair.rho);
          return rightScore - leftScore || left.label.localeCompare(right.label);
        });
        ordered.push(remaining.shift());
      }
      return ordered;
    }

    function renderCorrelationMap() {
      if (!app.data?.features?.length) return;
      const statistic = $("correlationStatistic").value;
      const selectedGroup = $("correlationGroup").value;
      const valuesByFeature = supportedValuesByFeature(statistic);
      let features = app.data.features.filter(feature => selectedGroup === "all" || feature.body_region === selectedGroup);
      features = [...features].sort((left, right) =>
        String(left.body_region).localeCompare(String(right.body_region)) || left.label.localeCompare(right.label)
      );
      const pairLookup = new Map();
      features.forEach((left, leftIndex) => features.slice(leftIndex).forEach(right => {
        pairLookup.set(
          correlationPairKey(left.feature_name, right.feature_name),
          measurementPairCorrelation(left.feature_name, right.feature_name, statistic, valuesByFeature),
        );
      }));
      if ($("correlationOrder").value === "association") features = associationOrderedFeatures(features, pairLookup);
      const selectedX = $("relationshipX").value;
      const selectedY = $("relationshipY").value;
      const header = features.map(feature => `<th class="correlation-column ${[selectedX, selectedY].includes(feature.feature_name) ? "selected" : ""}" scope="col" title="${escapeHtml(feature.label)}"><span>${escapeHtml(feature.label)}</span></th>`).join("");
      let availablePairCount = 0;
      let circularPairCount = 0;
      const rows = features.map(left => {
        const cells = features.map(right => {
          const pair = pairLookup.get(correlationPairKey(left.feature_name, right.feature_name));
          if (left.feature_name === right.feature_name) return `<td><span class="correlation-diagonal" title="${escapeHtml(left.label)}; ${pair.n} supported independent cases">—</span></td>`;
          if (pair.reason === "circular_method_required") circularPairCount += 0.5;
          if (pair.rho === null) {
            const reason = pair.reason === "circular_method_required"
              ? `Withheld: ${left.label} and ${right.label} include a directional angle mean that requires a circular-aware method. Paired cases: ${pair.n}.`
              : pair.reason === "constant_values"
                ? `Unavailable: one measurement is constant across the ${pair.n} paired cases.`
                : `Insufficient evidence: ${pair.n} paired independent cases; at least ${MINIMUM_CORRELATION_CASES} are required.`;
            return `<td><span class="correlation-na" title="${escapeHtml(reason)}">–</span></td>`;
          }
          availablePairCount += 0.5;
          const selectedPair = [left.feature_name, right.feature_name].includes(selectedX) && [left.feature_name, right.feature_name].includes(selectedY) ? "selected-pair" : "";
          const title = `${left.label} and ${right.label}: Spearman rho ${pair.rho.toFixed(2)} across ${pair.n} independent injury cases. Select to inspect the named cases.`;
          return `<td><button class="correlation-cell ${selectedPair}" type="button" data-correlation-x="${escapeHtml(left.feature_name)}" data-correlation-y="${escapeHtml(right.feature_name)}" style="${correlationCellStyle(pair.rho)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">${pair.rho.toFixed(2)}</button></td>`;
        }).join("");
        const selected = [selectedX, selectedY].includes(left.feature_name) ? "selected" : "";
        return `<tr><th class="correlation-row-label ${selected}" scope="row" title="${escapeHtml(left.label)}">${escapeHtml(left.label)}</th>${cells}</tr>`;
      }).join("");
      $("correlationMap").innerHTML = features.length
        ? `<table class="correlation-map"><thead><tr><th class="correlation-corner">Measurement</th>${header}</tr></thead><tbody>${rows}</tbody></table>`
        : '<div class="empty">No measurements are available in this group.</div>';
      $("correlationMap").querySelectorAll("[data-correlation-x]").forEach(button => {
        button.addEventListener("click", () => {
          $("relationshipX").value = button.dataset.correlationX;
          $("relationshipY").value = button.dataset.correlationY;
          renderMeasurementHelp();
          renderCorrelationMap();
          renderRelationship();
          $("relationshipDrilldown").scrollIntoView({behavior: "smooth", block: "start"});
        });
      });
      const possiblePairCount = features.length * (features.length - 1) / 2;
      const groupText = selectedGroup === "all" ? "all measurement groups" : categoryLabel(selectedGroup).toLowerCase();
      $("correlationMapSummary").textContent = `${Math.round(availablePairCount)} of ${possiblePairCount} unique measurement pairs in ${groupText} meet the evidence rules for descriptive Spearman correlation. Each cell reports a coefficient from mutually supported independent injury cases; hover or focus it to see paired N. ${Math.round(circularPairCount)} directional-angle pairs are withheld pending a circular-aware method.`;
    }

    function renderRelationship() {
      if (!app.data.features.length) return;
      const xFeature = $("relationshipX").value;
      const yFeature = $("relationshipY").value;
      const statistic = $("correlationStatistic").value;
      const unitId = record => record.statistical_unit_id || record.case_id;
      const rawXRecords = new Map(app.data.records.filter(record => record.feature_name === xFeature).map(record => [unitId(record), record]));
      const rawYRecords = new Map(app.data.records.filter(record => record.feature_name === yFeature).map(record => [unitId(record), record]));
      const supportedXRecords = new Map(selectedRecords(xFeature, statistic).map(record => [unitId(record), record]));
      const points = selectedRecords(yFeature, statistic).filter(record => supportedXRecords.has(unitId(record))).map(record => ({
        caseId: unitId(record),
        playerName: record.player_name,
        x: Number(supportedXRecords.get(unitId(record))[statistic]),
        y: Number(record[statistic]),
      })).sort((left, right) => String(left.playerName).localeCompare(String(right.playerName)));
      const canvas = $("relationshipCanvas");
      const {context, width, height} = canvasContext(canvas);
      if (!points.length) {
        drawNoData(context, width, height, "No mutually supported case values.");
        $("relationshipSummary").textContent = `0 of ${app.data.events.length} independent cases support both selected measurements. Unavailable or limited values are not plotted as zero.`;
        $("relationshipTechnicalSummary").textContent = "A relationship coefficient cannot be calculated without mutually supported case values.";
        $("relationshipLegend").innerHTML = "";
      } else {
        context.clearRect(0, 0, width, height);
        const xUnit = unitLabel(xFeature, statistic);
        const yUnit = unitLabel(yFeature, statistic);
        const statisticLabel = statisticLabels[statistic] || statistic;
        const margin = drawAxes(
          context,
          width,
          height,
          `${featureLabel(xFeature)} · ${statisticLabel} (${xUnit})`,
          `${featureLabel(yFeature)} · ${statisticLabel} (${yUnit})`,
        );
        const [minX, maxX] = numericDomain(points.map(point => point.x));
        const [minY, maxY] = numericDomain(points.map(point => point.y));
        drawLinearTicks(context, width, height, margin, [minX, maxX], "x", xUnit);
        drawLinearTicks(context, width, height, margin, [minY, maxY], "y", yUnit);
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        points.forEach((point, index) => {
          const x = margin.left + (point.x - minX) / (maxX - minX) * plotWidth;
          const y = margin.top + (maxY - point.y) / (maxY - minY) * plotHeight;
          context.fillStyle = "#18744a";
          context.beginPath(); context.arc(x, y, 10, 0, Math.PI * 2); context.fill();
          context.fillStyle = "#fff";
          context.font = "800 10px system-ui";
          context.textAlign = "center";
          context.fillText(String(index + 1), x, y + 3.5);
        });
        $("relationshipLegend").innerHTML = points.map((point, index) => (
          `<li><span class="chart-key">${index + 1}</span><span>${escapeHtml(point.playerName)}</span></li>`
        )).join("");
        canvas.setAttribute("aria-label", `Two-measurement comparison with ${points.length} numbered case points. The numbered legend and table identify every player.`);
        $("relationshipSummary").textContent = `${points.length} of ${app.data.events.length} independent cases support both measurements. Inspect the named cases; this small case library does not establish a population relationship, causation, or ACL risk.`;
        const circularMethodRequired = wrapSensitiveMean(xFeature, statistic) || wrapSensitiveMean(yFeature, statistic);
        if (circularMethodRequired) {
          $("relationshipTechnicalSummary").textContent = `Paired independent cases: ${points.length}. An ordinary rank coefficient is withheld because at least one directional-angle mean requires a circular-aware method. The named points remain visible for evidence inspection.`;
        } else if (points.length < MINIMUM_CORRELATION_CASES) {
          $("relationshipTechnicalSummary").textContent = `Paired independent cases: ${points.length}. At least ${MINIMUM_CORRELATION_CASES} mutually supported injury cases are required before the descriptive Spearman coefficient is shown.`;
        } else {
          const rho = correlation(ranks(points.map(point => point.x)), ranks(points.map(point => point.y)));
          const rhoText = rho === null ? "undefined for constant values" : rho.toFixed(2);
          $("relationshipTechnicalSummary").textContent = `Descriptive Spearman rho: ${rhoText}; paired independent cases: ${points.length}. This is exploratory association, not causation, ACL risk, or confirmatory significance.`;
        }
      }
      $("relationshipXHeading").textContent = featureLabel(xFeature);
      $("relationshipYHeading").textContent = featureLabel(yFeature);
      $("relationshipRows").innerHTML = app.data.events.map(event => {
        const xRecord = rawXRecords.get(event.case_id);
        const yRecord = rawYRecords.get(event.case_id);
        const xStatus = measurementStatus(xRecord, statistic);
        const yStatus = measurementStatus(yRecord, statistic);
        const included = xStatus === "Supported" && yStatus === "Supported";
        const statusText = included ? "Included" : `First: ${xStatus}; second: ${yStatus}`;
        return `<tr>
          <td><strong>${escapeHtml(event.player_name)}</strong></td>
          <td>${xStatus === "Supported" ? formatValue(xRecord[statistic], xFeature, statistic) : "Unavailable"}</td>
          <td>${yStatus === "Supported" ? formatValue(yRecord[statistic], yFeature, statistic) : "Unavailable"}</td>
          <td><span class="badge ${included ? "good" : "caution"}">${escapeHtml(statusText)}</span></td>
        </tr>`;
      }).join("") || '<tr><td colspan="4" class="empty">No case summaries are available.</td></tr>';
    }

    function renderReadiness() {
      const readiness = app.data.readiness;
      const rows = [
        ["Describe the current cases", readiness.descriptive_eda],
        ["Compare two measurements", readiness.correlation],
        ["Compare contact groups", readiness.contact_group_comparison],
        ["Make confirmatory statistical claims", readiness.confirmatory_inference],
      ];
      $("readinessRows").innerHTML = rows.map(([label, item]) => `<tr>
        <td><strong>${escapeHtml(label)}</strong></td>
        <td><span class="badge ${item.eligible ? "good" : "caution"}">${escapeHtml(item.status.replaceAll("_", " "))}</span></td>
        <td>${escapeHtml(item.reason || (item.eligible ? "Available for inspection." : "Unavailable."))}</td>
      </tr>`).join("");
      $("testFamilyRows").innerHTML = app.data.test_families.map(item => `<tr>
        <td><strong>${escapeHtml(item.question)}</strong></td>
        <td>${item.candidate_tests.map(escapeHtml).join("; ")}</td>
        <td>${item.required_output.map(escapeHtml).join("; ")}</td>
      </tr>`).join("");
    }

    function checkTestEligibility() {
      const feature = $("testFeature").value;
      const statistic = $("testStatistic").value;
      const groupVariable = $("testGroup").value;
      const records = selectedRecords(feature, statistic).filter(record => {
        const value = record[groupVariable];
        return value !== null && value !== undefined &&
          !["unknown", "uncertain", "unclear", ""].includes(String(value));
      });
      const groups = {};
      records.forEach(record => {
        const group = String(record[groupVariable]);
        groups[group] = (groups[group] || 0) + 1;
      });
      const entries = Object.entries(groups);
      const result = $("eligibilityResult");
      if (entries.length < 2) {
        result.className = "eligibility-result caution";
        result.innerHTML = `<h3>These groups cannot be compared yet</h3><p>At least two groups with recorded cases are needed. Current eligible counts: ${entries.length ? entries.map(([name, count]) => `${escapeHtml(categoryLabel(name, groupVariable))} = ${count}`).join(", ") : "no classified cases"}. Add the missing case details or analyse more independent cases before comparing groups.</p>`;
        return;
      }
      const minimum = Math.min(...entries.map(([, count]) => count));
      if (minimum < 5) {
        result.className = "eligibility-result caution";
        result.innerHTML = `<h3>Not enough independent cases yet</h3><p>${entries.map(([name, count]) => `${escapeHtml(categoryLabel(name, groupVariable))} = ${count}`).join(", ")}. Each group needs at least five separate injury cases for this exploratory check. Extra frames and replay views do not increase the number of cases.</p>`;
        return;
      }
      const methods = entries.length === 2
        ? "Welch t-test, Mann-Whitney U, or a permutation test"
        : "Welch ANOVA, Kruskal-Wallis, or permutation ANOVA";
      const groupCounts = entries.map(([name, count]) => `${categoryLabel(name, groupVariable)} = ${count}`).join(", ");
      result.className = "eligibility-result good";
      result.innerHTML = `<h3>Exploratory comparison available</h3><p>${escapeHtml(groupCounts)}. Candidate methods: ${methods}. The final choice still requires distribution and outlier checks, effect sizes, confidence intervals, and multiple-testing correction.</p>`;
    }

    function redrawActiveCharts() {
      if (!app.data) return;
      if ($("breakdownsView").classList.contains("active")) renderBreakdowns();
      if ($("profilesView").classList.contains("active")) renderProfiles();
      if ($("distributionView").classList.contains("active")) renderDistribution();
      if ($("relationshipView").classList.contains("active")) {
        renderCorrelationMap();
        renderRelationship();
      }
    }

    function activateTab(button) {
      document.querySelectorAll(".tab").forEach(item => {
        const selected = item === button;
        item.classList.toggle("active", selected);
        item.setAttribute("aria-selected", String(selected));
        item.tabIndex = selected ? 0 : -1;
      });
      document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === button.dataset.view));
      requestAnimationFrame(redrawActiveCharts);
    }

    const tabs = Array.from(document.querySelectorAll(".tab"));
    tabs.forEach((button, index) => {
      button.addEventListener("click", () => activateTab(button));
      button.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const nextIndex = event.key === "Home"
          ? 0
          : event.key === "End"
            ? tabs.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        tabs[nextIndex].focus();
        activateTab(tabs[nextIndex]);
      });
    });

    ["distributionFeature", "distributionStatistic"].forEach(id => $(id).addEventListener("change", () => { renderMeasurementHelp(); renderDistribution(); }));
    $("breakdownVariable").addEventListener("change", renderBreakdowns);
    ["relationshipX", "relationshipY"].forEach(id => $(id).addEventListener("change", () => { renderMeasurementHelp(); renderCorrelationMap(); renderRelationship(); }));
    $("correlationStatistic").addEventListener("change", () => { renderCorrelationMap(); renderRelationship(); });
    ["correlationGroup", "correlationOrder"].forEach(id => $(id).addEventListener("change", renderCorrelationMap));
    $("testFeature").addEventListener("change", renderMeasurementHelp);
    $("sourceSearch").addEventListener("input", renderSources);
    ["sourceMechanismFilter", "sourceEvidenceFilter"].forEach(id => $(id).addEventListener("change", renderSources));
    $("checkEligibility").addEventListener("click", checkTestEligibility);
    window.addEventListener("resize", redrawActiveCharts);

    let exploreLoadController = null;
    async function loadExploreData() {
      if (exploreLoadController) exploreLoadController.abort();
      const controller = new AbortController();
      exploreLoadController = controller;
      let timedOut = false;
      const timeout = window.setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, 20000);
      $("analysisUnitNote").innerHTML = `<div class="app-football-loader" role="status" aria-live="polite"><span class="app-loader-pitch" aria-hidden="true"><span class="app-loader-ball">⚽</span></span><span class="app-loader-copy"><strong>Surveying the pitch…</strong><small>Loading case-level evidence, coverage, and comparison readiness.</small></span></div>`;
      $("inferenceStatus").textContent = "Loading";
      try {
        const response = await fetch("/api/explore", {signal: controller.signal});
        if (!response.ok) throw new Error("The exploration dataset could not be loaded.");
        const data = await response.json();
        app.data = data;
        renderSummary();
        renderOverview();
        renderMechanismMethodology();
        renderSources();
        populateControls();
        renderReadiness();
        renderBreakdowns();
        renderProfiles();
        redrawActiveCharts();
        const requestedView = document.body.dataset.initialView;
        if (requestedView === "correlations" || window.location.hash === "#correlations") activateTab($("relationshipTab"));
        else if (window.location.hash === "#sources") activateTab($("sourcesTab"));
      } catch (error) {
        if (controller !== exploreLoadController) return;
        const message = timedOut
          ? "The exploration dataset took longer than expected. You can safely try again."
          : error.name === "AbortError"
            ? "The previous loading attempt was stopped."
            : error.message;
        $("analysisUnitNote").innerHTML = `<strong>Data unavailable:</strong> ${escapeHtml(message)} <button type="button" id="retryExploreData">Try again</button>`;
        $("inferenceStatus").textContent = "Unavailable";
        $("retryExploreData").addEventListener("click", loadExploreData);
      } finally {
        window.clearTimeout(timeout);
      }
    }

    if (document.body.dataset.initialView === "correlations" || window.location.hash === "#correlations") activateTab($("relationshipTab"));
    else if (window.location.hash === "#sources") activateTab($("sourcesTab"));
    loadExploreData();
  </script>
</body>
</html>
""".replace("__APP_SHELL_CSS__", app_shell_css()).replace(
        "__APP_SITE_HEADER__", app_site_header(section_label)
    ).replace("__PAGE_TITLE__", page_title).replace(
        "__PAGE_HEADING__", page_heading
    ).replace("__PAGE_LEDE__", page_lede).replace(
        "__INITIAL_VIEW__", "correlations" if correlation_landing else "overview"
    )
    )


def render_feature_correlations_page() -> str:
    """Return the feature-correlation submenu landing directly on the heatmap."""

    return render_exploration_page(initial_view="correlations")
