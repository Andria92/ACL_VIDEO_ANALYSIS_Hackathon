"""Cross-case descriptive exploration page."""

from __future__ import annotations

from acl_motion.ui.app_shell import app_shell_css, app_site_header


def render_exploration_page() -> str:
    """Return the evidence-gated statistical exploration workspace."""

    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Explore Data - ACL Movement Analytics Lab</title>
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
    header {
      min-height: 60px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    header strong { font-size: 18px; }
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
      gap: 6px;
      margin: 18px 0 14px;
      padding: 8px 0;
      overflow-x: auto;
      background: var(--bg);
      box-shadow: 0 1px 0 var(--line);
    }
    .tab { white-space: nowrap; }
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
      header { align-items: flex-start; flex-direction: column; padding: 11px 14px; }
      main { width: min(100% - 20px, 640px); padding-top: 20px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .controls { grid-template-columns: 1fr; }
      .source-toolbar { grid-template-columns: 1fr; }
      .source-card-head { flex-direction: column; }
      .source-badges { justify-content: flex-start; }
      canvas { min-height: 240px; aspect-ratio: 1.5 / 1; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="app-skip-link" href="#mainContent">Skip to data exploration</a>
  __APP_SITE_HEADER__
  <header class="app-tool-header">
    <strong>Explore Data</strong>
    <a class="button" href="/">Main menu</a>
  </header>
  <main id="mainContent" class="app-page-main" tabindex="-1">
    <h1>Explore Data</h1>
    <p class="lede">Case-level descriptive analytics with visible evidence coverage and statistical safeguards. Case identity, injury status, and laterality come from supplied metadata, not video inference.</p>

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
      <button class="tab" id="sourcesTab" type="button" role="tab" aria-selected="false" aria-controls="sourcesView" tabindex="-1" data-view="sourcesView">Sources / Injury Reports</button>
      <button class="tab" id="distributionTab" type="button" role="tab" aria-selected="false" aria-controls="distributionView" tabindex="-1" data-view="distributionView">Compare cases</button>
      <button class="tab" id="relationshipTab" type="button" role="tab" aria-selected="false" aria-controls="relationshipView" tabindex="-1" data-view="relationshipView">Compare two measurements</button>
      <button class="tab" id="testBuilderTab" type="button" role="tab" aria-selected="false" aria-controls="testBuilderView" tabindex="-1" data-view="testBuilderView">Can these groups be compared?</button>
    </nav>

    <section class="view active" id="overviewView" role="tabpanel" aria-labelledby="overviewTab">
      <details class="band overview-disclosure" open>
        <summary><h2>Case Library</h2><span>Completed analyses grouped by registered injury event.</span></summary>
        <div class="table-wrap" role="region" aria-label="Case library table" tabindex="0"><table>
          <thead><tr><th>Case</th><th>Supplied case metadata</th><th>Team / competition</th><th>Position</th><th>Views</th><th>Geometry support</th><th>Dynamic support</th><th>Median frame coverage</th></tr></thead>
          <tbody id="caseRows"><tr><td colspan="8" class="empty">Loading cases</td></tr></tbody>
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

    <section class="view" id="distributionView" role="tabpanel" aria-labelledby="distributionTab">
      <div class="band">
        <h2>Compare Cases</h2>
        <p class="section-copy">Compare one supported measurement across registered injury cases. Every point is one injury case, never a video frame or replay.</p>
        <div class="controls">
          <label><span>Measurement</span><select id="distributionFeature"></select></label>
          <label><span>How to summarise each case</span><select id="distributionStatistic"></select></label>
        </div>
        <div class="chart-shell"><canvas id="distributionCanvas" class="tall-chart" role="img" aria-label="Named case comparison chart; values are also listed in the table below."></canvas></div>
        <p class="chart-summary" id="distributionSummary"></p>
        <div class="table-wrap"><table>
          <thead><tr><th>Case</th><th>Value</th><th>Support</th><th>Evidence view</th><th>Relevant coverage</th></tr></thead>
          <tbody id="distributionRows"></tbody>
        </table></div>
      </div>
    </section>

    <section class="view" id="relationshipView" role="tabpanel" aria-labelledby="relationshipTab">
      <div class="band">
        <h2>Compare Two Measurements</h2>
        <p class="section-copy">See how two supported measurements appear together in the same injury cases. With a small library, inspect the named cases rather than infer a population relationship.</p>
        <div class="controls">
          <label><span>First measurement</span><select id="relationshipX"></select></label>
          <label><span>Second measurement</span><select id="relationshipY"></select></label>
          <label><span>How to summarise each case</span><select id="relationshipStatistic"></select></label>
        </div>
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
            <option value="injured_side">Injured knee</option>
          </select></label>
        </div>
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
        const teamCompetition = [readableValue(event.team), readableValue(event.competition)]
          .filter(value => value !== "Not recorded");
        return `<tr>
          <td><strong>${escapeHtml(event.player_name)}</strong></td>
          <td>${injuryParts.map(escapeHtml).join(" · ")}<br><span class="badge ${knownContact ? "good" : "caution"}">${escapeHtml(mechanismLabel(contact))}</span></td>
          <td>${teamCompetition.length ? teamCompetition.map(escapeHtml).join("<br>") : '<span class="muted">Not recorded</span>'}</td>
          <td>${escapeHtml(readableValue(event.position_group))}</td>
          <td>${event.analysed_view_count}</td>
          <td>${event.geometry_eligible_feature_count} / ${event.feature_count}</td>
          <td>${event.dynamic_eligible_feature_count} / ${event.feature_count}</td>
          <td>Geometry ${percent(event.median_geometry_completeness)}<br><span class="muted">Dynamics ${percent(event.median_dynamic_completeness)}</span></td>
        </tr>`;
      }).join("") || '<tr><td colspan="8" class="empty">No completed case summaries are available.</td></tr>';

      $("featureRows").innerHTML = app.data.features.map(feature => {
        const limited = !valueAvailable(feature.median_geometry_completeness) || feature.median_geometry_completeness < 0.6;
        const unavailableLimited = feature.unavailable_case_count + feature.unsupported_case_count;
        const geometryWidth = valueAvailable(feature.median_geometry_completeness)
          ? Math.round(feature.median_geometry_completeness * 100) : 0;
        return `<tr>
          <td><strong>${escapeHtml(feature.label)}</strong></td>
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
            event.competition,
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
      ["distributionStatistic", "relationshipStatistic", "testStatistic"].forEach(id =>
        setSelectOptions($(id), statistics, "value", "label")
      );
      if (features.length > 1) $("relationshipY").selectedIndex = 1;
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
      const canvas = $("distributionCanvas");
      canvas.style.height = `${Math.max(360, records.length * 34 + 100)}px`;
      const {context, width, height} = canvasContext(canvas);
      if (!records.length) {
        drawNoData(context, width, height, "No supported case values for this selection.");
        $("distributionSummary").textContent = `0 of ${allRecords.length} independent cases have a supported value. Missing or limited measurements are listed below and are not plotted as zero.`;
      } else {
        context.clearRect(0, 0, width, height);
        const unit = unitLabel(feature, statistic);
        const margin = {left: Math.min(210, Math.max(145, width * 0.23)), right: 28, top: 24, bottom: 58};
        const values = records.map(record => Number(record[statistic]));
        const [min, max] = numericDomain(values);
        drawLinearTicks(context, width, height, margin, [min, max], "x", unit);
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const centre = median(values);
        const medianX = margin.left + (centre - min) / (max - min) * plotWidth;
        context.save();
        context.strokeStyle = "#735b24";
        context.setLineDash([6, 5]);
        context.beginPath(); context.moveTo(medianX, margin.top); context.lineTo(medianX, height - margin.bottom); context.stroke();
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
        canvas.setAttribute("aria-label", `${featureLabel(feature)} comparison across ${records.length} named injury cases. Full values are listed in the table below.`);
        const omitted = allRecords.length - records.length;
        $("distributionSummary").textContent = `${records.length} of ${allRecords.length} independent cases contribute supported values; ${omitted} ${omitted === 1 ? "case is" : "cases are"} unavailable or limited. Median: ${formatValue(centre, feature, statistic)}. Dashed line = median. This is a named case comparison, not a population distribution.`;
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

    function renderRelationship() {
      if (!app.data.features.length) return;
      const xFeature = $("relationshipX").value;
      const yFeature = $("relationshipY").value;
      const statistic = $("relationshipStatistic").value;
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
        const rho = correlation(ranks(points.map(point => point.x)), ranks(points.map(point => point.y)));
        const rhoText = rho === null ? "undefined for constant values" : rho.toFixed(2);
        $("relationshipTechnicalSummary").textContent = `Descriptive Spearman rho: ${rhoText}; paired independent cases: ${points.length}. This is exploratory association, not causation, ACL risk, or confirmatory significance.`;
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
      const records = selectedRecords(feature, statistic).filter(record =>
        !["unknown", "uncertain", "unclear", ""].includes(String(record[groupVariable] || "unknown"))
      );
      const groups = {};
      records.forEach(record => {
        const group = record[groupVariable];
        groups[group] = (groups[group] || 0) + 1;
      });
      const entries = Object.entries(groups);
      const result = $("eligibilityResult");
      if (entries.length < 2) {
        result.className = "eligibility-result caution";
        result.innerHTML = `<h3>These groups cannot be compared yet</h3><p>At least two groups with recorded cases are needed. Current eligible counts: ${entries.length ? entries.map(([name, count]) => `${escapeHtml(name)} = ${count}`).join(", ") : "no classified cases"}. Add the missing case details or analyse more independent cases before comparing groups.</p>`;
        return;
      }
      const minimum = Math.min(...entries.map(([, count]) => count));
      if (minimum < 5) {
        result.className = "eligibility-result caution";
        result.innerHTML = `<h3>Not enough independent cases yet</h3><p>${entries.map(([name, count]) => `${escapeHtml(name)} = ${count}`).join(", ")}. Each group needs at least five separate injury cases for this exploratory check. Extra frames and replay views do not increase the number of cases.</p>`;
        return;
      }
      const methods = entries.length === 2
        ? "Welch t-test, Mann-Whitney U, or a permutation test"
        : "Welch ANOVA, Kruskal-Wallis, or permutation ANOVA";
      result.className = "eligibility-result good";
      result.innerHTML = `<h3>Exploratory comparison available</h3><p>Candidate methods: ${methods}. The final choice still requires distribution and outlier checks, effect sizes, confidence intervals, and multiple-testing correction.</p>`;
    }

    function redrawActiveCharts() {
      if ($("distributionView").classList.contains("active")) renderDistribution();
      if ($("relationshipView").classList.contains("active")) renderRelationship();
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

    ["distributionFeature", "distributionStatistic"].forEach(id => $(id).addEventListener("change", renderDistribution));
    ["relationshipX", "relationshipY", "relationshipStatistic"].forEach(id => $(id).addEventListener("change", renderRelationship));
    $("sourceSearch").addEventListener("input", renderSources);
    ["sourceMechanismFilter", "sourceEvidenceFilter"].forEach(id => $(id).addEventListener("change", renderSources));
    $("checkEligibility").addEventListener("click", checkTestEligibility);
    window.addEventListener("resize", redrawActiveCharts);

    fetch("/api/explore")
      .then(response => {
        if (!response.ok) throw new Error("The exploration dataset could not be loaded.");
        return response.json();
      })
      .then(data => {
        app.data = data;
        renderSummary();
        renderOverview();
        renderMechanismMethodology();
        renderSources();
        populateControls();
        renderReadiness();
        if (window.location.hash === "#sources") activateTab($("sourcesTab"));
      })
      .catch(error => {
        $("analysisUnitNote").innerHTML = `<strong>Data unavailable:</strong> ${escapeHtml(error.message)}`;
        $("inferenceStatus").textContent = "Unavailable";
      });
  </script>
</body>
</html>
""".replace("__APP_SHELL_CSS__", app_shell_css()).replace(
        "__APP_SITE_HEADER__", app_site_header("Explore Data")
    )
