"""Application home for the ACL Movement Analytics Lab workflow."""

from __future__ import annotations

from acl_motion.ui.app_shell import app_shell_css


def render_home_page() -> str:
    """Return the connected ACL Movement Analytics Lab workflow home."""

    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Analytics Lab</title>
  <style>
    :root {
      color-scheme: light;
      --navy-950: #071a2d;
      --navy-900: #0b2239;
      --navy-800: #123a5a;
      --blue-700: #155f91;
      --blue-600: #1677ac;
      --blue-050: #f1f8fc;
      --teal-700: #08766d;
      --teal-100: #d9f2ed;
      --amber-700: #7a5700;
      --amber-100: #fff1c2;
      --red-700: #a12d43;
      --red-100: #fde8ed;
      --ink: #142334;
      --muted: #586879;
      --subtle: #7a8997;
      --line: #d7e0e7;
      --line-strong: #bdcbd6;
      --panel: #ffffff;
      --surface: #f5f8fa;
      --focus: #e6a500;
      --shadow-sm: 0 1px 2px rgba(7,26,45,.05), 0 8px 24px rgba(7,26,45,.05);
      --shadow-md: 0 14px 38px rgba(7,26,45,.10);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      min-height: 100vh;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    button, input, select { font: inherit; }
    button, a { -webkit-tap-highlight-color: transparent; }
    :focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }
    .visually-hidden {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0,0,0,0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
    .skip-link {
      position: fixed;
      z-index: 20;
      top: 10px;
      left: 10px;
      transform: translateY(-150%);
      padding: 10px 14px;
      border-radius: 8px;
      background: #fff;
      color: var(--navy-950);
      font-weight: 850;
      box-shadow: var(--shadow-md);
    }
    .skip-link:focus { transform: translateY(0); }
    .site-header {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 12px max(24px, calc((100% - 1320px) / 2));
      border-bottom: 1px solid rgba(255,255,255,.10);
      background: var(--navy-950);
      color: #fff;
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .brand-mark {
      position: relative;
      width: 38px;
      height: 38px;
      flex: 0 0 auto;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,.34);
      border-radius: 10px;
      background:
        linear-gradient(90deg, transparent 48%, rgba(255,255,255,.28) 48% 52%, transparent 52%),
        linear-gradient(0deg, transparent 48%, rgba(255,255,255,.28) 48% 52%, transparent 52%),
        var(--teal-700);
    }
    .brand-mark::after {
      content: "";
      position: absolute;
      width: 14px;
      height: 14px;
      top: 11px;
      left: 11px;
      border: 1px solid rgba(255,255,255,.52);
      border-radius: 50%;
    }
    .brand h1 { margin: 0; font-size: 18px; line-height: 1.15; letter-spacing: -.01em; }
    .brand span { display: block; margin-top: 3px; color: #b9c9d5; font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
    .event-label { color: #c9d6df; font-size: 12px; font-weight: 750; white-space: nowrap; }
    main { width: min(1320px, calc(100% - 48px)); margin: 0 auto; padding: 24px 0 56px; }
    .hero {
      position: relative;
      min-height: 190px;
      display: grid;
      align-items: center;
      overflow: hidden;
      padding: 30px clamp(24px, 4vw, 52px) 0;
      border-radius: 18px;
      background:
        radial-gradient(circle at 84% 10%, rgba(34,160,151,.20), transparent 27%),
        linear-gradient(118deg, var(--navy-900), #0e3653 72%, #0b4761);
      color: #fff;
      box-shadow: var(--shadow-md);
    }
    .hero-copy { position: relative; z-index: 2; max-width: 790px; padding-bottom: 76px; }
    .eyebrow { margin: 0 0 8px; color: #7fd8cf; font-size: 11px; font-weight: 900; letter-spacing: .105em; text-transform: uppercase; }
    .hero h2 { margin: 0; max-width: 720px; font-size: clamp(27px, 3vw, 42px); line-height: 1.08; letter-spacing: -.032em; }
    .hero-lede { max-width: 760px; margin: 10px 0 0; color: #d6e1e8; font-size: 14px; line-height: 1.5; }
    .scope-strip {
      position: absolute;
      z-index: 3;
      inset: auto 0 0;
      min-height: 54px;
      display: flex;
      align-items: stretch;
      background: rgba(4,21,36,.77);
      border-top: 1px solid rgba(255,255,255,.12);
    }
    .scope-item { min-width: 0; display: flex; align-items: center; gap: 9px; padding: 11px 20px; color: #d8e3e9; font-size: 12px; font-weight: 700; }
    .scope-item + .scope-item { border-left: 1px solid rgba(255,255,255,.12); }
    .scope-icon { color: #75d4cb; font-size: 14px; font-weight: 900; }
    .scope-item.limit .scope-icon { color: #f5ce66; }
    .movement-graphic { position: absolute; z-index: 1; right: 20px; top: 2px; width: 350px; max-width: 30%; height: 178px; opacity: .54; pointer-events: none; }
    .section { margin-top: 28px; scroll-margin-top: 16px; }
    .section-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 13px; }
    .section-kicker { margin: 0 0 4px; color: var(--blue-700); font-size: 11px; font-weight: 900; letter-spacing: .09em; text-transform: uppercase; }
    .section-header h2 { margin: 0; font-size: clamp(22px, 2vw, 29px); line-height: 1.15; letter-spacing: -.025em; }
    .section-intro { max-width: 720px; margin: 6px 0 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .live-count {
      flex: 0 0 auto;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 7px 11px;
      border: 1px solid #a8d5cc;
      border-radius: 999px;
      background: var(--teal-100);
      color: #075c55;
      font-size: 12px;
      font-weight: 850;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 4px rgba(8,118,109,.10); }
    .review-shell {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .75fr);
      min-height: 394px;
      overflow: hidden;
      border: 1px solid var(--line-strong);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow-sm);
    }
    .case-browser { min-width: 0; padding: 20px; border-right: 1px solid var(--line); }
    .browser-label { display: block; margin-bottom: 10px; font-size: 13px; font-weight: 850; }
    .case-tools { display: grid; grid-template-columns: minmax(0, 1fr) 170px 190px; gap: 10px; margin-bottom: 12px; }
    .input-wrap { position: relative; }
    .input-wrap svg { position: absolute; top: 14px; left: 13px; color: var(--subtle); pointer-events: none; }
    input[type="search"], select { width: 100%; min-height: 46px; border: 1px solid var(--line-strong); border-radius: 9px; background: #fff; color: var(--ink); }
    input[type="search"] { padding: 10px 12px 10px 40px; }
    select { padding: 9px 34px 9px 11px; }
    input[type="search"]:hover, select:hover { border-color: var(--blue-600); }
    .case-list { height: 430px; overflow: auto; padding: 2px; scrollbar-color: var(--line-strong) transparent; }
    .case-option {
      width: 100%;
      min-height: 70px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      padding: 12px 13px;
      border: 1px solid transparent;
      border-radius: 10px;
      background: transparent;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }
    .case-option + .case-option { margin-top: 4px; }
    .case-option:hover { background: var(--blue-050); }
    .case-option[aria-pressed="true"] { border-color: #8dc1dd; background: var(--blue-050); box-shadow: inset 4px 0 0 var(--blue-600); }
    .case-option-copy { min-width: 0; }
    .case-option strong { display: block; overflow-wrap: anywhere; font-size: 14px; line-height: 1.3; }
    .case-option small { display: block; margin-top: 4px; overflow-wrap: anywhere; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .case-option-state { display: inline-flex; align-items: center; gap: 5px; color: var(--amber-700); font-size: 11px; font-weight: 850; white-space: nowrap; }
    .case-option-state.ready { color: var(--teal-700); }
    .case-option-state::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .list-message { min-height: 220px; display: grid; place-items: center; padding: 30px; color: var(--muted); text-align: center; }
    .loading-lines { width: min(100%, 520px); }
    .loading-line { height: 54px; margin: 8px 0; border-radius: 9px; background: linear-gradient(90deg, #edf2f5 25%, #f7f9fa 50%, #edf2f5 75%); background-size: 200% 100%; animation: shimmer 1.4s infinite linear; }
    .selected-case { position: relative; min-width: 0; display: flex; flex-direction: column; padding: 25px; background: linear-gradient(180deg, rgba(220,238,250,.72), rgba(255,255,255,0) 54%), #fff; }
    .selected-case::before { content: ""; position: absolute; inset: 0 0 auto; height: 4px; background: linear-gradient(90deg, var(--blue-600), var(--teal-700)); }
    .selected-label { margin: 0 0 12px; color: var(--blue-700); font-size: 11px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; }
    .selected-case h3 { margin: 0; overflow-wrap: anywhere; font-size: clamp(22px, 2vw, 28px); line-height: 1.12; letter-spacing: -.025em; }
    .selected-view { margin: 7px 0 0; color: var(--muted); font-size: 14px; line-height: 1.4; }
    .case-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 16px 0; }
    .case-fact { min-width: 0; padding: 9px 10px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.82); }
    .case-fact dt { color: var(--subtle); font-size: 10px; font-weight: 850; letter-spacing: .035em; text-transform: uppercase; }
    .case-fact dd { margin: 3px 0 0; overflow-wrap: anywhere; color: var(--ink); font-size: 12px; font-weight: 750; }
    .case-views-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 3px 0 8px; }
    .case-views-heading h4 { margin: 0; font-size: 14px; }
    .case-views-heading span { color: var(--muted); font-size: 11px; }
    .case-view-list { display: grid; gap: 8px; max-height: 250px; overflow: auto; padding-right: 2px; }
    .case-view-row { padding: 10px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
    .case-view-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .case-view-copy { min-width: 0; }
    .case-view-copy strong { display: block; overflow-wrap: anywhere; font-size: 12px; }
    .case-view-copy small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; line-height: 1.35; }
    .view-status { flex: 0 0 auto; color: var(--amber-700); font-size: 10px; font-weight: 850; }
    .view-status.ready { color: var(--teal-700); }
    .case-view-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .case-view-actions .button { min-height: 32px; padding: 6px 9px; font-size: 10px; }
    .meta-chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 18px 0; }
    .chip { display: inline-flex; align-items: center; min-height: 28px; padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--muted); font-size: 11px; font-weight: 800; }
    .chip.ready { border-color: #a8d5cc; background: var(--teal-100); color: #075c55; }
    .case-progress { display: grid; gap: 7px; margin: -7px 0 15px; }
    .case-progress-copy { align-items: baseline; display: flex; gap: 8px; justify-content: space-between; }
    .case-progress-copy strong { font-size: 12px; }
    .case-progress-copy span { color: var(--muted); font-size: 10px; }
    .case-progress-track { height: 7px; overflow: hidden; border-radius: 999px; background: #dfe8ee; }
    .case-progress-track span { display: block; height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, var(--blue-600), var(--teal-700)); transition: width .18s ease; }
    .selected-technical { margin-top: 2px; color: var(--muted); font-size: 12px; }
    .selected-technical summary { cursor: pointer; font-weight: 800; }
    .selected-technical code { display: block; margin-top: 7px; overflow-wrap: anywhere; color: var(--subtle); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
    .selected-action { margin-top: auto; padding-top: 16px; }
    .selected-action-note { margin: 0 0 10px; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .button {
      min-height: 46px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 10px 15px;
      border: 1px solid var(--line-strong);
      border-radius: 9px;
      background: #fff;
      color: var(--ink);
      font-weight: 820;
      text-decoration: none;
      cursor: pointer;
      transition: border-color .15s ease, background .15s ease, color .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    .button:hover { border-color: var(--blue-600); color: var(--blue-700); }
    .button.primary { border-color: var(--blue-700); background: var(--blue-700); color: #fff; box-shadow: 0 7px 18px rgba(21,95,145,.20); }
    .button.primary:hover { background: #0f527f; color: #fff; transform: translateY(-1px); }
    .button.danger { border-color: #d58c9b; background: #fff; color: var(--red-700); }
    .button.danger:hover { border-color: var(--red-700); background: var(--red-100); color: #7f1930; }
    .button.tertiary { min-height: 40px; padding: 6px 0; border-color: transparent; background: transparent; color: var(--blue-700); justify-content: flex-start; }
    .button[disabled] { cursor: wait; opacity: .62; transform: none; }
    .selected-action { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .selected-action[hidden] { display: none; }
    .selected-action-note { grid-column: 1 / -1; }
    .selected-action .button { width: 100%; }
    #continueCaseButton { grid-column: 1 / -1; }
    #deleteCaseButton, #deletionStatus { grid-column: 1 / -1; }
    #deletionStatus { min-height: 18px; margin: 0; color: var(--muted); font-size: 11px; line-height: 1.4; }
    .error-panel { padding: 15px; border: 1px solid #e4a5b2; border-radius: 10px; background: var(--red-100); color: var(--red-700); }
    .error-panel p { margin: 0 0 10px; font-size: 13px; line-height: 1.45; }
    .workflow-rail { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); overflow: hidden; border: 1px solid var(--line-strong); border-radius: 14px; background: var(--panel); box-shadow: var(--shadow-sm); }
    .workflow-step { position: relative; min-width: 0; display: flex; flex-direction: column; min-height: 218px; padding: 19px; }
    .workflow-step + .workflow-step { border-left: 1px solid var(--line); }
    .workflow-step + .workflow-step::before { content: "›"; position: absolute; z-index: 2; left: -11px; top: 24px; width: 22px; height: 22px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 50%; background: #fff; color: var(--subtle); font-weight: 900; }
    .step-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
    .step-number { color: var(--blue-700); font-size: 11px; font-weight: 900; letter-spacing: .08em; }
    .step-state { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 10px; font-weight: 850; text-transform: uppercase; }
    .step-state::before { content: ""; width: 7px; height: 7px; border: 2px solid currentColor; border-radius: 50%; }
    .step-state.available { color: var(--teal-700); }
    .step-state.available::before { border: 0; background: currentColor; }
    .step-state.embedded { color: var(--amber-700); }
    .workflow-step h3 { margin: 0; font-size: 17px; letter-spacing: -.01em; }
    .workflow-step p { margin: 7px 0 15px; color: var(--muted); font-size: 12px; line-height: 1.5; }
    .step-sequence { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0 0 15px; padding: 0; color: var(--blue-700); list-style: none; font-size: 10px; font-weight: 900; letter-spacing: .035em; text-transform: uppercase; }
    .step-sequence li { display: inline-flex; align-items: center; gap: 6px; }
    .step-sequence li + li::before { content: "→"; color: var(--subtle); }
    .workflow-step .button { width: fit-content; margin-top: auto; }
    .advanced-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .research-card { min-width: 0; min-height: 264px; display: flex; flex-direction: column; padding: 21px; border: 1px solid var(--line-strong); border-radius: 14px; background: var(--panel); box-shadow: var(--shadow-sm); }
    .research-card.gated { background: linear-gradient(140deg, #fffaf0, #fff 55%); }
    .card-status { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 13px; }
    .status-badge { display: inline-flex; align-items: center; gap: 6px; min-height: 26px; padding: 5px 8px; border-radius: 999px; background: var(--amber-100); color: var(--amber-700); font-size: 10px; font-weight: 900; letter-spacing: .04em; text-transform: uppercase; }
    .status-badge.available { background: var(--teal-100); color: #075c55; }
    .research-card h3 { margin: 0; font-size: 20px; letter-spacing: -.018em; }
    .research-card > p { margin: 7px 0 13px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .evidence-meter { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 2px 0 12px; }
    .evidence-stat { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.82); }
    .evidence-stat strong { display: block; font-size: 18px; }
    .evidence-stat span { display: block; margin-top: 2px; color: var(--muted); font-size: 10px; line-height: 1.25; }
    .gate-requirement { margin: 0 0 8px; padding: 10px 12px; border-left: 3px solid var(--amber-700); background: rgba(255,241,194,.42); color: #594813; font-size: 11px; line-height: 1.45; }
    .research-card details { color: var(--muted); font-size: 11px; line-height: 1.45; }
    .research-card summary { cursor: pointer; font-weight: 800; }
    .research-card details p { margin: 7px 0 0; }
    .research-card .button { margin-top: auto; width: fit-content; }
    .locked-action {
      width: fit-content;
      min-height: 40px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: auto;
      padding: 8px 11px;
      border: 1px solid #dec984;
      border-radius: 9px;
      background: #fffaf0;
      color: var(--amber-700);
      font-size: 12px;
      font-weight: 850;
    }
    .dataset-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 4px 0 16px; }
    .dataset-stat { padding: 12px; border-radius: 10px; background: var(--blue-050); }
    .dataset-stat strong { display: block; color: var(--navy-800); font-size: 22px; }
    .dataset-stat span { color: var(--muted); font-size: 10px; line-height: 1.25; }
    .page-note { margin: 25px 0 0; color: var(--subtle); font-size: 11px; line-height: 1.45; text-align: center; }
    @keyframes shimmer { to { background-position: -200% 0; } }
    @media (max-width: 1000px) {
      main { width: min(100% - 32px, 860px); }
      .movement-graphic { opacity: .34; max-width: 38%; }
      .review-shell { grid-template-columns: minmax(0, 1.1fr) minmax(300px, .9fr); }
      .case-tools { grid-template-columns: 1fr; }
      .case-list { height: 318px; }
      .workflow-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .workflow-step { min-height: 202px; }
    }
    @media (max-width: 720px) {
      html { scroll-behavior: auto; }
      .site-header { align-items: flex-start; padding: 12px 16px; }
      .brand span, .event-label { display: none; }
      main { width: min(100% - 24px, 560px); padding-top: 14px; }
      .hero { min-height: 276px; padding: 24px 20px 0; border-radius: 14px; }
      .hero-copy { padding-bottom: 126px; }
      .hero h2 { font-size: clamp(27px, 8vw, 35px); }
      .movement-graphic { right: -80px; opacity: .23; max-width: 80%; }
      .scope-strip { display: grid; grid-template-columns: 1fr; }
      .scope-item { padding: 7px 18px; }
      .scope-item + .scope-item { border-left: 0; border-top: 1px solid rgba(255,255,255,.10); }
      .section { margin-top: 24px; }
      .section-header { align-items: flex-start; flex-direction: column; gap: 10px; }
      .review-shell, .advanced-grid { grid-template-columns: 1fr; }
      .case-browser { padding: 15px; border-right: 0; border-bottom: 1px solid var(--line); }
      .case-list { height: 286px; }
      .selected-case { min-height: 342px; padding: 21px; }
      .case-facts { grid-template-columns: 1fr 1fr; }
      .workflow-rail { grid-template-columns: 1fr; }
      .workflow-step { min-height: 0; padding: 18px; }
      .workflow-step + .workflow-step { border-left: 0; border-top: 1px solid var(--line); }
      .workflow-step + .workflow-step::before { content: none; }
      .workflow-step .button, .research-card .button, .locked-action { width: 100%; }
      .research-card { min-height: 0; }
    }
    @media (max-width: 390px) {
      .brand-mark { width: 34px; height: 34px; }
      .brand-mark::after { top: 9px; left: 9px; }
      .brand h1 { font-size: 16px; }
      .hero { padding-inline: 17px; }
      .case-option { grid-template-columns: 1fr; gap: 7px; }
      .case-facts, .selected-action { grid-template-columns: 1fr; }
      .selected-action-note { grid-column: auto; }
      .evidence-meter, .dataset-summary { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="skip-link" href="#review">Skip to case library</a>
  <header class="site-header">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true"></span>
      <div><h1>ACL Movement Analytics Lab</h1><span>Women’s football movement research</span></div>
    </div>
    <span class="event-label">Hack for Humanity · Summer 2026</span>
  </header>
  <main id="mainContent">
    <section class="hero" aria-labelledby="heroTitle">
      <div class="hero-copy">
        <p class="eyebrow">Human-guided · Evidence-led · 2D video analysis</p>
        <h2 id="heroTitle">Observe the movement. Follow the evidence.</h2>
        <p class="hero-lede">Explore traceable movement stories from documented women’s-football ACL injury clips, with measurement support and evidence gaps kept in view.</p>
      </div>
      <svg class="movement-graphic" viewBox="0 0 350 178" aria-hidden="true">
        <path d="M18 140 C78 104,112 153,174 95 S281 56,332 27" fill="none" stroke="#7fd8cf" stroke-width="2" stroke-dasharray="7 8" />
        <path d="M265 23 L287 54 L270 91 M287 54 L316 72 M287 54 L303 28 M270 91 L245 127 M270 91 L302 126" fill="none" stroke="#d7f3ef" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        <g fill="#7fd8cf"><circle cx="265" cy="23" r="6"/><circle cx="287" cy="54" r="5"/><circle cx="270" cy="91" r="5"/><circle cx="316" cy="72" r="4"/><circle cx="303" cy="28" r="4"/><circle cx="245" cy="127" r="4"/><circle cx="302" cy="126" r="4"/></g>
        <path d="M14 151 H334 M50 151 V163 M126 151 V163 M202 151 V163 M278 151 V163" fill="none" stroke="#d7f3ef" stroke-width="1" opacity=".6" />
      </svg>
      <div class="scope-strip" aria-label="Research scope">
        <div class="scope-item"><span class="scope-icon" aria-hidden="true">✓</span><span>Human-verified athlete and movement window</span></div>
        <div class="scope-item"><span class="scope-icon" aria-hidden="true">◎</span><span>Projected 2D observations with visible support</span></div>
        <div class="scope-item limit"><span class="scope-icon" aria-hidden="true">—</span><span>Not diagnosis, injury-risk calculation, or causation</span></div>
      </div>
    </section>

    <section class="section" id="review" aria-labelledby="reviewTitle">
      <div class="section-header">
        <div>
          <p class="section-kicker">Primary entry point</p>
          <h2 id="reviewTitle">Injury Case Library</h2>
          <p class="section-intro">Choose one injury event to review its player details, attached video views, annotation status, and available analyses.</p>
        </div>
        <span class="live-count" id="analysisCount" role="status" aria-live="polite"><span class="status-dot" aria-hidden="true"></span>Loading analyses</span>
      </div>
      <div class="review-shell">
        <div class="case-browser">
          <label class="browser-label" for="caseSearch">Find an injury case</label>
          <div class="case-tools">
            <div class="input-wrap">
              <svg width="17" height="17" viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M13 13l4 4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
              <input id="caseSearch" type="search" placeholder="Search player, date, team, or competition" autocomplete="off" disabled />
            </div>
            <label><span class="visually-hidden">Filter injury cases</span><select id="caseFilter" disabled>
              <option value="all">All injury cases</option>
              <option value="ready">Has analysis</option>
              <option value="pending">Needs annotation or analysis</option>
              <option value="multiview">Multiple video views</option>
            </select></label>
            <label><span class="visually-hidden">Sort injury cases</span><select id="caseSort" disabled>
              <option value="analysis_desc">Recently analysed (unfinished first)</option>
              <option value="player_asc">Player name (A–Z)</option>
            </select></label>
          </div>
          <div class="case-list" id="caseList" role="group" aria-label="Injury cases" aria-busy="true">
            <div class="app-football-loader" role="status" aria-live="polite">
              <span class="app-loader-pitch" aria-hidden="true"><span class="app-loader-ball">⚽</span></span>
              <span class="app-loader-copy"><strong>Reviewing the case line-up…</strong><small>Loading players, clips, and the latest completed analyses.</small></span>
              <span class="visually-hidden">Loading injury cases</span>
            </div>
          </div>
          <div class="error-panel" id="analysisError" role="alert" hidden>
            <p id="analysisErrorMessage">The analysis list could not be loaded.</p>
            <button class="button" id="retryAnalyses" type="button">Try again</button>
          </div>
        </div>
        <aside class="selected-case" id="selectedCase" aria-labelledby="selectedCaseName">
          <p class="selected-label">Selected injury case</p>
          <h3 id="selectedCaseName">Loading case</h3>
          <p class="selected-view" id="selectedCaseView">Checking shared case metadata and video views.</p>
          <div class="meta-chips" id="selectedMeta"><span class="chip">Loading</span></div>
          <div class="case-progress" id="caseProgress" role="progressbar" aria-label="Selected case analysis progress" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0">
            <div class="case-progress-copy"><strong id="caseProgressLabel">Analysis progress</strong><span id="caseProgressDetail">Loading</span></div>
            <div class="case-progress-track" aria-hidden="true"><span id="caseProgressBar"></span></div>
          </div>
          <dl class="case-facts" id="caseFacts"></dl>
          <div class="case-views-heading"><h4>Video views</h4><span id="caseViewCount"></span></div>
          <div class="case-view-list" id="caseViews"></div>
          <details class="selected-technical" id="technicalDetails" hidden>
            <summary>Technical case metadata</summary><code id="technicalCaseId"></code>
          </details>
          <div class="selected-action" id="selectedAction">
            <p class="selected-action-note" id="selectedActionNote">Loading the next step for this case.</p>
            <a class="button primary" id="continueCaseButton" href="/annotate">Continue case <span aria-hidden="true">→</span></a>
            <a class="button" id="addViewButton" href="/video-cutter">Add video view <span aria-hidden="true">＋</span></a>
            <a class="button" id="editCaseButton" href="/annotate">Edit case details</a>
            <button class="button danger" id="deleteCaseButton" type="button">Delete case</button>
            <p id="deletionStatus" role="status" aria-live="polite"></p>
          </div>
        </aside>
      </div>
    </section>

    <section class="section" aria-labelledby="workflowTitle">
      <div class="section-header"><div>
        <p class="section-kicker">For new material</p>
        <h2 id="workflowTitle">Create and analyse a case</h2>
        <p class="section-intro">Use this sequence when adding a new injury event. Existing cases and all their attached views remain available in the library above.</p>
      </div></div>
      <div class="workflow-rail" aria-label="Case creation workflow">
        <article class="workflow-step">
          <div class="step-top"><span class="step-number">STEP 01</span><span class="step-state available">Available</span></div>
          <h3>Create or choose the injury case</h3><p>Record the player and injury-event information before opening source video. One injury case can hold any number of video views.</p>
          <a class="button" href="/video-cutter">Start case setup <span aria-hidden="true">→</span></a>
        </article>
        <article class="workflow-step">
          <div class="step-top"><span class="step-number">STEP 02</span><span class="step-state available">One workspace</span></div>
          <h3>Cut and attach video views</h3><p>Cut one or several intervals. Every cut remains attached to the active player injury case instead of becoming a new case.</p>
          <a class="button" href="/video-cutter">Open Video Cutter <span aria-hidden="true">→</span></a>
        </article>
        <article class="workflow-step">
          <div class="step-top"><span class="step-number">STEP 03</span><span class="step-state available">One workspace</span></div>
          <h3>Annotate, verify and generate</h3><p>Identify the athlete, confirm the movement window and case details, validate the annotation, then generate the human-guided analysis.</p>
          <ol class="step-sequence" aria-label="Inside the annotation workspace"><li>Annotate</li><li>Verify</li><li>Generate</li></ol>
          <a class="button" href="/annotate">Open Annotation Workspace <span aria-hidden="true">→</span></a>
        </article>
        <article class="workflow-step">
          <div class="step-top"><span class="step-number">STEP 04</span><span class="step-state available" id="reviewStepState">Check status</span></div>
          <h3>Review the Movement Story</h3><p>Inspect the clip, observations, measurements, limitations, and research detail.</p>
          <a class="button tertiary" href="#review">Choose a completed analysis <span aria-hidden="true">↑</span></a>
        </article>
      </div>
    </section>

    <section class="section" aria-labelledby="researchTitle">
      <div class="section-header"><div>
        <p class="section-kicker">Separate research tools</p>
        <h2 id="researchTitle">Compare and explore the case library</h2>
        <p class="section-intro">Cross-case tools operate on independent injury events and supported measurements. They are not required to create or review an individual case.</p>
      </div></div>
      <div class="advanced-grid">
        <article class="research-card gated" aria-labelledby="compareTitle">
          <div class="card-status"><span class="status-badge" id="similarityBadge"><span aria-hidden="true">▣</span> Evidence-gated</span></div>
          <h3 id="compareTitle">Compare movements</h3>
          <p id="similaritySummary">Checking the current evidence and deterministic comparison outputs.</p>
          <div class="evidence-meter" aria-label="Movement comparison readiness">
            <div class="evidence-stat"><strong id="independentCaseCount">–</strong><span>independent analysed cases</span></div>
            <div class="evidence-stat"><strong id="signatureCount">–</strong><span>comparable cases</span></div>
            <div class="evidence-stat"><strong id="pairwiseCount">–</strong><span>computed pairings</span></div>
          </div>
          <p class="gate-requirement"><strong>Comparison rule:</strong> only mutually supported case-level measurements contribute to a ranking.</p>
          <details>
            <summary>Scientific scope of a future comparison</summary>
            <p id="similarityScientificNote">A comparison could describe only the supported projected movement representation—not injury mechanism, tissue loading, biological cause, or clinical condition.</p>
          </details>
          <a class="button primary" href="/compare">Open Compare Movements <span aria-hidden="true">→</span></a>
        </article>
        <article class="research-card" aria-labelledby="exploreTitle">
          <div class="card-status"><span class="status-badge available">Available now</span></div>
          <h3 id="exploreTitle">Explore the dataset</h3>
          <p>Inspect case-level distributions, measurement relationships, evidence coverage, and test readiness.</p>
          <div class="dataset-summary" aria-label="Current dataset summary">
            <div class="dataset-stat"><strong id="datasetCaseCount">–</strong><span>independent cases</span></div>
            <div class="dataset-stat"><strong id="datasetViewCount">–</strong><span>analysed views</span></div>
            <div class="dataset-stat"><strong id="datasetFeatureCount">–</strong><span>measurement types</span></div>
          </div>
          <a class="button primary" href="/explore">Open Statistical Explorer <span aria-hidden="true">→</span></a>
          <a class="button" href="/explore#sources">Sources / Injury Reports <span aria-hidden="true">→</span></a>
        </article>
      </div>
    </section>
    <p class="page-note">Research interface for observable movement evidence. Outputs require contextual interpretation and are not clinical assessments.</p>
  </main>

  <script>
    const app = {views: [], cases: [], filtered: [], selectedCaseId: null};
    const $ = id => document.getElementById(id);
    const uuidPattern = /[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/ig;

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
      }[char]));
    }
    function clipWindowLabel(item) {
      const source = String(item.video_source_label || item.video_path || item.slug || "");
      const match = source.match(/(\d{2})m(\d{2})s(\d{3})[_-]+(\d{2})m(\d{2})s(\d{3})/i);
      if (!match) return "";
      return match[1] + ":" + match[2] + "." + match[3] + "–" + match[4] + ":" + match[5] + "." + match[6];
    }
    function displayPlayerName(item) {
      const sourceName = String(item.player_name || "");
      const raw = sourceName.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
      uuidPattern.lastIndex = 0;
      if (!raw) return "Player not recorded";
      if (uuidPattern.test(sourceName)) {
        uuidPattern.lastIndex = 0;
        return "Imported movement clip";
      }
      uuidPattern.lastIndex = 0;
      if (/^screen recording\b/i.test(raw)) return "Imported screen recording";
      const casePlayer = String(item.case_id || "").match(/^([a-z][a-z'-]+)_([a-z][a-z'-]+)_acl$/i);
      if (/\binjury\b/i.test(raw) && casePlayer) {
        return [casePlayer[1], casePlayer[2]].map(part => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
      }
      return raw.replace(/\s+\S{8,}\s+\d{2}m\d{2}s\d{3}.*$/i, "").replace(/\s+\d{2}m\d{2}s\d{3}.*$/i, "").trim() || "Imported movement clip";
    }
    function viewLabel(item) {
      const label = String(item.view_label || "").trim();
      if (label && !/^imported local video$/i.test(label)) return label;
      if (item.slow_motion) return "Imported slow-motion view";
      return item.primary_view ? "Imported primary view" : "Imported video view";
    }
    function formatDate(value) {
      const text = String(value || "").trim();
      if (!text) return "Not recorded";
      const parts = text.split("-");
      if (parts.length !== 3) return text;
      return `${parts[2]} ${["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][Number(parts[1]) - 1] || parts[1]} ${parts[0]}`;
    }
    function recorded(value, fallback = "Not recorded") {
      const text = String(value || "").trim();
      return text && text !== "unknown" ? text : fallback;
    }
    function groupCases(views) {
      const grouped = new Map();
      views.forEach(view => {
        const caseId = String(view.case_id || view.slug);
        if (!grouped.has(caseId)) grouped.set(caseId, []);
        grouped.get(caseId).push(view);
      });
      return Array.from(grouped.entries()).map(([caseId, caseViews]) => {
        caseViews.sort((left, right) => Number(!left.primary_view) - Number(!right.primary_view) || viewLabel(left).localeCompare(viewLabel(right)));
        const primary = caseViews[0];
        const details = primary.case_details || {};
        const analysisDates = caseViews
          .map(view => String(view.analysis_generated_at || ""))
          .filter(Boolean)
          .sort();
        return {
          case_id: caseId,
          player_name: details.player_name || primary.player_name,
          details,
          injured_side: primary.injured_side || "unknown",
          views: caseViews,
          ready_count: caseViews.filter(view => view.results_available).length,
          annotated_count: caseViews.filter(view => view.annotation_saved).length,
          latest_analysis_at: analysisDates.length ? analysisDates[analysisDates.length - 1] : "",
        };
      }).sort((left, right) => displayPlayerName(left).localeCompare(displayPlayerName(right)) || String(left.details.injury_date || "").localeCompare(String(right.details.injury_date || "")));
    }
    function caseSubtitle(item) {
      const details = item.details || {};
      const parts = [];
      if (details.injury_date) parts.push(formatDate(details.injury_date));
      if (details.team) parts.push(details.team);
      if (details.position_group && details.position_group !== "unknown") parts.push(details.position_group);
      parts.push(`${item.views.length} video ${item.views.length === 1 ? "view" : "views"}`);
      if (item.latest_analysis_at) parts.push(`Analysed ${formatDate(item.latest_analysis_at.slice(0, 10))}`);
      return parts.join(" · ");
    }
    function caseState(item) {
      if (item.ready_count === item.views.length) return "Complete";
      if (item.ready_count) return `${item.ready_count}/${item.views.length} analysed`;
      if (item.annotated_count) return "Ready to generate";
      return "Needs annotation";
    }
    function matchesFilter(item, filter) {
      if (filter === "ready") return item.ready_count > 0;
      if (filter === "pending") return item.ready_count < item.views.length;
      if (filter === "multiview") return item.views.length > 1;
      return true;
    }
    function compareCases(left, right, sort) {
      const byPlayer = displayPlayerName(left).localeCompare(displayPlayerName(right)) || left.case_id.localeCompare(right.case_id);
      if (sort === "player_asc") return byPlayer;
      const leftNeedsWork = left.ready_count < left.views.length;
      const rightNeedsWork = right.ready_count < right.views.length;
      if (leftNeedsWork !== rightNeedsWork) return Number(rightNeedsWork) - Number(leftNeedsWork);
      const leftTime = Date.parse(left.latest_analysis_at || "") || 0;
      const rightTime = Date.parse(right.latest_analysis_at || "") || 0;
      return rightTime - leftTime || byPlayer;
    }
    function syncCaseUrl(caseId, historyMode = "replace") {
      const url = new URL(window.location.href);
      if (caseId) url.searchParams.set("case", caseId);
      else url.searchParams.delete("case");
      const method = historyMode === "push" ? "pushState" : "replaceState";
      window.history[method]({caseId: caseId || ""}, "", url);
    }
    function showNoFilteredSelection() {
      app.selectedCaseId = null;
      $("selectedCaseName").textContent = "No visible injury case selected";
      $("selectedCaseView").textContent = "Adjust the search or filter to select a case.";
      $("selectedMeta").innerHTML = '<span class="chip">No matching case</span>';
      $("caseFacts").innerHTML = "";
      $("caseViews").innerHTML = "";
      $("caseViewCount").textContent = "0 visible";
      $("caseProgressLabel").textContent = "No matching case";
      $("caseProgressDetail").textContent = "Adjust the search or filter";
      $("caseProgressBar").style.width = "0%";
      $("caseProgress").setAttribute("aria-valuemax", "1");
      $("caseProgress").setAttribute("aria-valuenow", "0");
      $("technicalDetails").hidden = true;
      $("selectedAction").hidden = true;
      $("deleteCaseButton").disabled = true;
      $("addViewButton").href = "/video-cutter";
      $("continueCaseButton").href = "/";
      $("editCaseButton").href = "/annotate";
      syncCaseUrl(null, "replace");
    }
    function renderCaseList({reconcileSelection = true} = {}) {
      const query = $("caseSearch").value.trim().toLocaleLowerCase();
      const filter = $("caseFilter").value;
      const sort = $("caseSort").value;
      app.filtered = app.cases.filter(item => {
        const details = item.details || {};
        const viewTerms = item.views.flatMap(view => [viewLabel(view), clipWindowLabel(view), view.video_source_label]);
        const haystack = [displayPlayerName(item), item.case_id, details.injury_date, details.date_of_birth, details.team, details.opponent, details.competition, details.position_group, ...viewTerms].join(" ").toLocaleLowerCase();
        return matchesFilter(item, filter) && (!query || haystack.includes(query));
      }).sort((left, right) => compareCases(left, right, sort));
      const list = $("caseList");
      list.setAttribute("aria-busy", "false");
      if (!app.filtered.length) {
        list.innerHTML = '<div class="list-message"><div><strong>No matching injury cases</strong><br><span>Adjust the search or case filter.</span></div></div>';
        if (reconcileSelection) showNoFilteredSelection();
        return;
      }
      if (reconcileSelection && !app.filtered.some(item => item.case_id === app.selectedCaseId)) {
        selectCase(app.filtered[0].case_id, {historyMode: "replace", renderList: false});
      }
      list.innerHTML = app.filtered.map(item => {
        const selected = item.case_id === app.selectedCaseId;
        const readyClass = item.ready_count === item.views.length ? " ready" : "";
        return '<button class="case-option" type="button" aria-pressed="' + selected + '" data-case-id="' + escapeHtml(item.case_id) + '">' +
          '<span class="case-option-copy"><strong>' + escapeHtml(displayPlayerName(item)) + '</strong><small>' + escapeHtml(caseSubtitle(item)) + '</small></span>' +
          '<span class="case-option-state' + readyClass + '">' + escapeHtml(caseState(item)) + '</span></button>';
      }).join("");
      list.querySelectorAll(".case-option").forEach(option => option.addEventListener("click", () => selectCase(option.dataset.caseId)));
    }
    function fact(label, value) {
      return '<div class="case-fact"><dt>' + escapeHtml(label) + '</dt><dd>' + escapeHtml(recorded(value)) + '</dd></div>';
    }
    function viewStatus(view) {
      if (view.results_available) return {label: "Analysis ready", className: "ready"};
      if (view.annotation_saved) return {label: "Annotated", className: ""};
      return {label: "Needs annotation", className: ""};
    }
    function renderView(view, allowDelete) {
      const status = viewStatus(view);
      const description = [clipWindowLabel(view) ? "Clip " + clipWindowLabel(view) : "", view.perspective && view.perspective !== "unknown" ? String(view.perspective).replaceAll("-", " ") : "", view.slow_motion ? "slow motion" : ""].filter(Boolean).join(" · ") || "Registered video view";
      const primaryAction = view.results_available
        ? '<a class="button primary" href="/results?case=' + encodeURIComponent(view.slug) + '">View analysis</a>'
        : '<a class="button primary" href="/annotate?case=' + encodeURIComponent(view.slug) + '">' + (view.annotation_saved ? "Generate analysis" : "Annotate view") + '</a>';
      const editAction = view.results_available
        ? '<a class="button" href="/annotate?case=' + encodeURIComponent(view.slug) + '">Edit annotation</a>'
        : '';
      const deleteAction = allowDelete
        ? '<button class="button danger delete-view-button" type="button" data-view-slug="' + escapeHtml(view.slug) + '">Delete view</button>'
        : '';
      return '<article class="case-view-row"><div class="case-view-top"><div class="case-view-copy"><strong>' + escapeHtml(viewLabel(view)) + '</strong><small>' + escapeHtml(description) + '</small></div><span class="view-status ' + status.className + '">' + escapeHtml(status.label) + '</span></div><div class="case-view-actions">' + primaryAction + editAction + deleteAction + '</div></article>';
    }
    function nextCaseAction(item) {
      const awaitingAnalysis = item.views.find(view => view.annotation_saved && !view.results_available);
      if (awaitingAnalysis) return {
        label: "Generate next analysis",
        href: "/annotate?case=" + encodeURIComponent(awaitingAnalysis.slug),
        note: "A saved annotation is ready for validation or analysis generation."
      };
      const awaitingAnnotation = item.views.find(view => !view.annotation_saved);
      if (awaitingAnnotation) return {
        label: "Annotate next clip",
        href: "/annotate?case=" + encodeURIComponent(awaitingAnnotation.slug),
        note: `${item.views.length - item.annotated_count} of ${item.views.length} clips still ${item.views.length - item.annotated_count === 1 ? "needs" : "need"} annotation.`
      };
      const firstResult = item.views.find(view => view.results_available);
      if (firstResult) return {
        label: "Review analysis",
        href: "/results?case=" + encodeURIComponent(firstResult.slug),
        note: "Every attached clip has an analysis ready for review."
      };
      return {
        label: "Add the first video view",
        href: "/video-cutter?case=" + encodeURIComponent(item.case_id),
        note: "Attach a video clip to begin this case."
      };
    }
    function selectCase(caseId, {historyMode = "push", renderList = true} = {}) {
      const item = app.cases.find(candidate => candidate.case_id === caseId);
      if (!item) return false;
      app.selectedCaseId = item.case_id;
      $("selectedAction").hidden = false;
      const details = item.details || {};
      const primaryView = item.views[0];
      $("selectedCaseName").textContent = displayPlayerName(item);
      $("selectedCaseView").textContent = [details.team, details.opponent ? "vs " + details.opponent : "", details.injury_date ? "Injury " + formatDate(details.injury_date) : ""].filter(Boolean).join(" · ") || "Shared case information";
      const chips = [
        '<span class="chip">' + item.views.length + ' ' + (item.views.length === 1 ? "clip" : "clips") + '</span>',
        '<span class="chip ready">' + item.ready_count + '/' + item.views.length + ' analysed</span>',
        '<span class="chip">' + escapeHtml(recorded(item.injured_side, "Injured side unknown")) + '</span>',
      ];
      $("selectedMeta").innerHTML = chips.join("");
      const completion = item.views.length ? item.ready_count / item.views.length : 0;
      $("caseProgress").setAttribute("aria-valuemax", String(item.views.length || 1));
      $("caseProgress").setAttribute("aria-valuenow", String(item.ready_count));
      $("caseProgressLabel").textContent = item.ready_count === item.views.length ? "Case analysis complete" : "Case analysis progress";
      $("caseProgressDetail").textContent = `${item.ready_count} of ${item.views.length} clips analysed`;
      $("caseProgressBar").style.width = `${Math.round(completion * 100)}%`;
      $("caseFacts").innerHTML = [
        fact("Date of birth", formatDate(details.date_of_birth)),
        fact("Injury date", formatDate(details.injury_date)),
        fact("Team", details.team),
        fact("Opponent", details.opponent),
        fact("Competition", details.competition),
        fact("Position", details.position_group),
      ].join("");
      $("caseViewCount").textContent = `${item.views.length} attached`;
      $("caseViews").innerHTML = item.views.map(view => renderView(view, item.views.length > 1)).join("");
      $("caseViews").querySelectorAll(".delete-view-button").forEach(button => button.addEventListener("click", () => deleteView(item, button.dataset.viewSlug, button)));
      $("technicalDetails").hidden = false;
      $("technicalCaseId").textContent = "Case ID: " + (item.case_id || "not recorded");
      const nextAction = nextCaseAction(item);
      $("selectedActionNote").textContent = nextAction.note;
      $("continueCaseButton").textContent = nextAction.label + " →";
      $("continueCaseButton").href = nextAction.href;
      const caseReturnPath = "/?case=" + encodeURIComponent(item.case_id);
      $("addViewButton").href = "/video-cutter?case=" + encodeURIComponent(item.case_id)
        + "&return=" + encodeURIComponent(caseReturnPath);
      $("editCaseButton").href = "/annotate?case=" + encodeURIComponent(primaryView.slug);
      $("deleteCaseButton").disabled = false;
      $("deletionStatus").textContent = "";
      if (historyMode) syncCaseUrl(item.case_id, historyMode);
      if (renderList) renderCaseList({reconcileSelection: false});
      return true;
    }
    function renderCases(views) {
      app.views = Array.isArray(views) ? views : [];
      app.cases = groupCases(app.views);
      const count = app.cases.length;
      const analysisCount = app.views.filter(view => view.results_available).length;
      $("caseSearch").disabled = !count;
      $("caseFilter").disabled = !count;
      $("caseSort").disabled = !count;
      $("analysisError").hidden = true;
      $("caseList").hidden = false;
      $("analysisCount").innerHTML = '<span class="status-dot" aria-hidden="true"></span>' +
        (count ? count + " injury " + (count === 1 ? "case" : "cases") : "No injury cases");
      $("reviewStepState").textContent = analysisCount ? analysisCount + " ready" : "Not ready";
      if (!count) {
        app.selectedCaseId = null;
        syncCaseUrl(null, "replace");
        $("caseList").setAttribute("aria-busy", "false");
        $("caseList").innerHTML = '<div class="list-message"><div><strong>No injury cases yet</strong><br><span>Use the workflow below to create the first case.</span></div></div>';
        $("selectedCaseName").textContent = "No injury case selected";
        $("selectedCaseView").textContent = "Cases and their video views will appear here.";
        $("selectedMeta").innerHTML = '<span class="chip">Awaiting case</span>';
        $("caseProgressLabel").textContent = "No case progress yet";
        $("caseProgressDetail").textContent = "0 clips";
        $("caseProgressBar").style.width = "0%";
        $("caseProgress").setAttribute("aria-valuenow", "0");
        $("caseFacts").innerHTML = "";
        $("caseViews").innerHTML = "";
        $("caseViewCount").textContent = "0 attached";
        $("technicalDetails").hidden = true;
        $("selectedAction").hidden = false;
        $("selectedActionNote").textContent = "Create a case and attach its first video view to begin.";
        $("addViewButton").href = "/video-cutter";
        $("continueCaseButton").textContent = "Create the first case →";
        $("continueCaseButton").href = "/video-cutter";
        $("editCaseButton").href = "/annotate";
        $("deleteCaseButton").disabled = true;
        return;
      }
      const requested = new URLSearchParams(window.location.search).get("case");
      const orderedCases = [...app.cases].sort(
        (left, right) => compareCases(left, right, $("caseSort").value)
      );
      const initial = app.cases.find(item => item.case_id === requested || item.views.some(view => view.slug === requested)) || orderedCases[0];
      selectCase(initial.case_id, {historyMode: "replace"});
    }
    function showAnalysisError(message) {
      $("analysisCount").innerHTML = '<span class="status-dot" aria-hidden="true"></span>Case library unavailable';
      $("caseList").hidden = true;
      $("analysisErrorMessage").textContent = message || "The analysis list could not be loaded.";
      $("analysisError").hidden = false;
      $("selectedCaseName").textContent = "Case library unavailable";
      $("selectedCaseView").textContent = "Retry the connection without losing any case data.";
      $("selectedMeta").innerHTML = '<span class="chip">Connection error</span>';
      $("caseProgressLabel").textContent = "Progress unavailable";
      $("caseProgressDetail").textContent = "Retry loading cases";
      $("caseProgressBar").style.width = "0%";
      $("caseProgress").setAttribute("aria-valuemax", "1");
      $("caseProgress").setAttribute("aria-valuenow", "0");
      $("caseFacts").innerHTML = "";
      $("caseViews").innerHTML = "";
      $("caseViewCount").textContent = "Unavailable";
      $("technicalDetails").hidden = true;
      $("selectedAction").hidden = true;
    }
    async function loadCases() {
      $("retryAnalyses").disabled = true;
      try {
        const response = await fetch("/api/cases?include_video_metadata=0");
        if (!response.ok) throw new Error("Could not load the injury case library.");
        const data = await response.json();
        renderCases(data.cases || []);
      } catch (error) {
        showAnalysisError(error.message);
      } finally {
        $("retryAnalyses").disabled = false;
      }
    }
    async function deleteEntry(payload, button, successMessage) {
      button.disabled = true;
      $("deletionStatus").textContent = "Removing from the case library…";
      try {
        const response = await fetch("/api/cases/delete", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "The case could not be deleted.");
        await loadCases();
        $("deletionStatus").textContent = successMessage + " Its source video and generated files were moved to Trash.";
      } catch (error) {
        button.disabled = false;
        $("deletionStatus").textContent = error.message;
      }
    }
    function deleteView(item, slug, button) {
      const view = item.views.find(candidate => candidate.slug === slug);
      if (!view) return;
      const confirmed = window.confirm('Delete the video view "' + viewLabel(view) + '" from ' + displayPlayerName(item) + '? Its source video and generated files will move to Trash. The other views in this case will remain.');
      if (!confirmed) return;
      deleteEntry(
        {scope: "view", case_id: item.case_id, slug},
        button,
        "Video view deleted from the library.",
      );
    }
    function deleteSelectedCase() {
      const item = app.cases.find(candidate => candidate.case_id === app.selectedCaseId);
      if (!item) return;
      const confirmed = window.confirm('Delete the entire injury case for ' + displayPlayerName(item) + ' and all ' + item.views.length + ' attached video ' + (item.views.length === 1 ? 'view' : 'views') + '? The source videos and generated files will move to Trash.');
      if (!confirmed) return;
      deleteEntry(
        {scope: "case", case_id: item.case_id},
        $("deleteCaseButton"),
        "Injury case deleted from the library.",
      );
    }
    function renderEvidenceSummary(data) {
      const summary = data && data.summary ? data.summary : {};
      const similarity = data && data.similarity ? data.similarity : {};
      const cases = Number(summary.analysed_case_count || 0);
      const views = Number(summary.analysed_view_count || 0);
      const features = Number(summary.feature_count || 0);
      const signatures = Number(similarity.comparable_case_count || 0);
      const pairwise = Number(similarity.pairwise_output_count || 0);
      $("independentCaseCount").textContent = cases;
      $("signatureCount").textContent = signatures;
      $("pairwiseCount").textContent = pairwise;
      $("datasetCaseCount").textContent = cases;
      $("datasetViewCount").textContent = views;
      $("datasetFeatureCount").textContent = features;
      if (similarity.available) {
        $("similarityBadge").innerHTML = '<span aria-hidden="true">✓</span> Comparison ready';
        $("similarityBadge").classList.add("available");
        $("similaritySummary").textContent = similarity.reason || "Supported player-to-player movement rankings are available.";
      } else {
        $("similaritySummary").textContent = similarity.reason || "At least two comparable analysed cases are required.";
      }
      if (similarity.scientific_note) $("similarityScientificNote").textContent = similarity.scientific_note;
    }
    function showEvidenceError() {
      ["independentCaseCount", "signatureCount", "pairwiseCount", "datasetCaseCount", "datasetViewCount", "datasetFeatureCount"]
        .forEach(id => $(id).textContent = "–");
      $("similaritySummary").textContent = "Readiness details are temporarily unavailable. Movement comparison remains evidence-gated.";
    }
    async function loadEvidenceSummary() {
      try {
        const response = await fetch("/api/explore/summary");
        if (!response.ok) throw new Error("Could not load evidence readiness.");
        renderEvidenceSummary(await response.json());
      } catch (error) {
        showEvidenceError();
      }
    }
    $("caseSearch").addEventListener("input", renderCaseList);
    $("caseFilter").addEventListener("change", renderCaseList);
    $("caseSort").addEventListener("change", renderCaseList);
    $("retryAnalyses").addEventListener("click", loadCases);
    $("deleteCaseButton").addEventListener("click", deleteSelectedCase);
    window.addEventListener("popstate", () => {
      const requested = new URLSearchParams(window.location.search).get("case");
      const item = app.cases.find(candidate => (
        candidate.case_id === requested || candidate.views.some(view => view.slug === requested)
      ));
      if (item) {
        $("caseSearch").value = "";
        $("caseFilter").value = "all";
        selectCase(item.case_id, {historyMode: null});
      } else if (app.cases.length) {
        const fallback = app.cases.find(candidate => candidate.case_id === app.selectedCaseId) || app.cases[0];
        selectCase(fallback.case_id, {historyMode: "replace"});
      }
    });
    loadCases().finally(loadEvidenceSummary);
  </script>
</body>
</html>
""".replace("__APP_SHELL_CSS__", app_shell_css())
