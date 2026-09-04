"""Blinded expert movement-similarity validation workspace."""

from __future__ import annotations

from acl_motion.ui.app_shell import app_shell_css, app_site_header, apply_app_brand


def render_similarity_validation_page() -> str:
    """Return the algorithm-blinded pairwise expert review page."""

    return apply_app_brand(
        VALIDATION_HTML.replace("__APP_SHELL_CSS__", app_shell_css())
        .replace("__APP_SITE_HEADER__", app_site_header("Similarity Validation"))
    )


VALIDATION_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Similarity Validation - ACL Movement Analytics Lab</title>
  <style>
    :root { --bg:#f5f8fa; --panel:#fff; --ink:#142334; --muted:#586879; --line:#d7e0e7; --accent:#0F62FE; --accent-soft:#eef5ff; --green:#08766d; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    main { width:min(1280px,calc(100% - 28px)); margin:0 auto; padding:28px 0 48px; }
    .panel { margin-bottom:14px; padding:17px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
    h1,h2,h3 { margin-top:0; }
    p { line-height:1.5; }
    .lede,.muted { color:var(--muted); }
    .controls { display:flex; flex-wrap:wrap; gap:9px; align-items:end; }
    label { display:grid; gap:5px; font-weight:750; }
    input,textarea,button,.button { min-height:44px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); font:inherit; }
    input,textarea { padding:8px 10px; }
    button,.button { display:inline-flex; align-items:center; justify-content:center; padding:8px 13px; font-weight:780; cursor:pointer; text-decoration:none; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button:disabled { cursor:not-allowed; opacity:.55; }
    .clips { display:grid; grid-template-columns:1fr 1fr; gap:13px; }
    .clip-card { padding:12px; border:1px solid var(--line); border-radius:9px; background:#fbfcfd; }
    .clip-card.query { grid-column:1 / -1; background:var(--accent-soft); }
    video { display:block; width:100%; max-height:330px; border-radius:7px; background:#111; }
    .choices { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; margin-top:13px; }
    .status { padding:10px 12px; border-left:4px solid var(--accent); background:var(--accent-soft); }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:9px; border-bottom:1px solid var(--line); text-align:left; }
    th { font-size:12px; color:var(--muted); text-transform:uppercase; }
    [hidden] { display:none !important; }
    @media(max-width:800px) { .clips,.choices { grid-template-columns:1fr; } .clip-card.query { grid-column:auto; } }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="app-skip-link" href="#mainContent">Skip to review workspace</a>
  __APP_SITE_HEADER__
  <main id="mainContent" class="app-page-main" tabindex="-1">
    <div class="controls" style="justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div><h1>Blinded expert similarity review</h1><p class="lede">Collect movement judgements without showing algorithm scores, rankings, or case metadata.</p></div>
      <div class="controls"><a class="button" href="/compare">Compare Movements</a><a class="button" href="/">Main menu</a></div>
    </div>

    <section class="panel">
      <h2>Before starting</h2>
      <p>Use a pseudonymous assessor ID that remains the same for one reviewer. Watch the query movement, then decide whether Option A or Option B is more similar overall. “About the same” and “Unable to judge” are valid answers.</p>
      <p class="muted">This hides the engine and recorded metadata, but footage itself may reveal a player or team. Reviewers should not open the normal comparison rankings during this task.</p>
      <div class="controls">
        <label>Assessor ID<input id="assessorId" autocomplete="off" placeholder="for example expert_02" /></label>
        <button id="startButton" class="primary" type="button">Start or continue review</button>
      </div>
    </section>

    <section class="panel" id="assignmentPanel" hidden>
      <div class="controls" style="justify-content:space-between;align-items:center;"><h2>Which option is closer to the query?</h2><strong id="progress"></strong></div>
      <div class="clips">
        <article class="clip-card query"><h3>Query movement</h3><video id="queryVideo" controls preload="metadata"></video></article>
        <article class="clip-card"><h3>Option A</h3><video id="optionAVideo" controls preload="metadata"></video></article>
        <article class="clip-card"><h3>Option B</h3><video id="optionBVideo" controls preload="metadata"></video></article>
      </div>
      <label style="margin-top:12px;">Optional observation<textarea id="notes" rows="2" placeholder="Record visibility problems or why the pair was difficult."></textarea></label>
      <div class="choices">
        <button type="button" data-choice="OPTION_A">Option A is closer</button>
        <button type="button" data-choice="OPTION_B">Option B is closer</button>
        <button type="button" data-choice="ABOUT_THE_SAME">About the same</button>
        <button type="button" data-choice="UNABLE_TO_JUDGE">Unable to judge</button>
      </div>
    </section>

    <section class="panel" id="completionPanel" hidden><h2>Review queue complete</h2><p>No unreviewed assignments remain for this assessor.</p></section>
    <p class="status" id="status">Enter an assessor ID to begin.</p>

    <section class="panel">
      <h2>Current validation evidence</h2>
      <p class="muted">These are current-case concordance results, not final held-out validation. Results appear only after usable A/B judgements have been saved.</p>
      <div id="report">No expert judgements have been evaluated yet.</div>
    </section>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    let assignment = null;
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[character]);
    async function loadAssignment() {
      const assessor = $('assessorId').value.trim();
      if (!assessor) { $('status').textContent = 'Please enter an assessor ID.'; return; }
      localStorage.setItem('similarityValidationAssessor', assessor);
      $('status').textContent = 'Loading a blinded assignment…';
      const response = await fetch(`/api/similarity-validation/assignment?assessor=${encodeURIComponent(assessor)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Assignment could not be loaded.');
      assignment = data.assignment;
      $('assignmentPanel').hidden = !assignment;
      $('completionPanel').hidden = Boolean(assignment);
      $('progress').textContent = `${data.completed_count} of ${data.assignment_count} completed`;
      if (assignment) {
        $('queryVideo').src = assignment.query_video_url;
        $('optionAVideo').src = assignment.option_a_video_url;
        $('optionBVideo').src = assignment.option_b_video_url;
        $('notes').value = '';
        $('status').textContent = 'The assignment is blinded to algorithm results and recorded case metadata.';
      } else {
        $('status').textContent = 'This assessor has completed the available review queue.';
      }
    }
    async function saveChoice(choice) {
      if (!assignment) return;
      document.querySelectorAll('[data-choice]').forEach(button => button.disabled = true);
      try {
        const response = await fetch('/api/similarity-validation/judgement', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({assessor_id:$('assessorId').value.trim(), assignment_id:assignment.assignment_id, choice, notes:$('notes').value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Judgement could not be saved.');
        await Promise.all([loadAssignment(), loadReport()]);
      } catch (error) { $('status').textContent = error.message; }
      finally { document.querySelectorAll('[data-choice]').forEach(button => button.disabled = false); }
    }
    async function loadReport() {
      const response = await fetch('/api/similarity-validation/report');
      const data = await response.json();
      if (!response.ok) return;
      const rows = Object.values(data.expert_concordance?.lenses || {});
      const summary = data.internal_audit?.summary || {};
      const expert = data.expert_concordance || {};
      $('report').innerHTML = `<p><strong>${expert.judgement_count || 0}</strong> saved judgements from <strong>${expert.assessor_count || 0}</strong> assessors. Internal audit median primary top-rank retention: <strong>${summary.median_primary_top_retention ?? 'unavailable'}</strong> across <strong>${summary.primary_jackknife_evaluable_query_count || 0}</strong> evaluable queries.</p>` +
        (rows.length ? `<table><thead><tr><th>Lens</th><th>Evaluated choices</th><th>Expert concordance</th><th>95% interval</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escapeHtml(row.label)}</td><td>${row.evaluated}</td><td>${row.concordance ?? 'unavailable'}</td><td>${row.wilson_95_interval ? row.wilson_95_interval.join('–') : 'unavailable'}</td></tr>`).join('')}</tbody></table>` : '');
    }
    $('startButton').onclick = () => loadAssignment().catch(error => $('status').textContent = error.message);
    document.querySelectorAll('[data-choice]').forEach(button => button.onclick = () => saveChoice(button.dataset.choice));
    $('assessorId').value = localStorage.getItem('similarityValidationAssessor') || '';
    loadReport();
  </script>
</body>
</html>
"""
