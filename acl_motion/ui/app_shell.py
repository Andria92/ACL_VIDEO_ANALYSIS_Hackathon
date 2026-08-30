"""Shared application shell for ACL Movement Analytics Lab tool pages."""

from __future__ import annotations

from html import escape


def app_shell_css() -> str:
    """Return the home-page header and page-frame styles used by submenus."""

    return r"""
    :focus-visible {
      outline: 3px solid #e6a500;
      outline-offset: 3px;
    }
    .app-skip-link {
      position: fixed;
      z-index: 1000;
      top: 10px;
      left: 10px;
      transform: translateY(-160%);
      padding: 10px 14px;
      border-radius: 8px;
      background: #fff;
      color: #071a2d;
      font-weight: 850;
      text-decoration: none;
      box-shadow: 0 14px 38px rgba(7,26,45,.16);
    }
    .app-skip-link:focus { transform: translateY(0); }
    .app-site-header {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 12px max(24px, calc((100% - 1320px) / 2));
      border: 0;
      border-bottom: 1px solid rgba(255,255,255,.10);
      border-radius: 0;
      background: #071a2d;
      color: #fff;
      box-shadow: none;
    }
    .app-site-header .app-brand {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 12px;
      color: inherit;
      text-decoration: none;
    }
    .app-site-header .app-brand-mark {
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
        #08766d;
    }
    .app-site-header .app-brand-mark::after {
      content: "";
      position: absolute;
      width: 14px;
      height: 14px;
      top: 11px;
      left: 11px;
      border: 1px solid rgba(255,255,255,.52);
      border-radius: 50%;
    }
    .app-site-header .app-brand-copy { min-width: 0; }
    .app-site-header .app-brand-title {
      display: block;
      margin: 0;
      color: #fff;
      font-size: 18px;
      font-weight: 800;
      line-height: 1.15;
      letter-spacing: -.01em;
    }
    .app-site-header .app-brand-subtitle {
      display: block;
      margin-top: 3px;
      color: #b9c9d5;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .app-site-header .app-section-label {
      color: #c9d6df;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .app-tool-header {
      width: min(1320px, calc(100% - 48px));
      min-height: 64px;
      margin: 18px auto 0;
      padding: 12px 16px;
      border: 1px solid #d7e0e7;
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(7,26,45,.05), 0 8px 24px rgba(7,26,45,.05);
    }
    .app-page-main {
      width: min(1320px, calc(100% - 48px));
      max-width: none;
      margin: 0 auto;
      padding: 24px 0 56px;
    }
    .app-football-loader {
      width: min(100%, 560px);
      min-height: 82px;
      display: grid;
      grid-template-columns: 82px minmax(0, 1fr);
      align-items: center;
      gap: 14px;
      margin: 0 auto;
      padding: 13px 15px;
      border: 1px solid #c9ddd8;
      border-radius: 12px;
      background: linear-gradient(135deg, #f5fbf9, #eef7fb);
      color: #142334;
      text-align: left;
    }
    .app-football-loader.compact {
      grid-template-columns: 66px minmax(0, 1fr);
      min-height: 70px;
      padding: 10px 11px;
    }
    .app-loader-pitch {
      position: relative;
      width: 72px;
      height: 46px;
      overflow: hidden;
      border: 2px solid #5caa9c;
      border-radius: 7px;
      background: #dff2ed;
    }
    .compact .app-loader-pitch { width: 58px; height: 38px; }
    .app-loader-pitch::before {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      border-left: 1px solid rgba(8,118,109,.42);
    }
    .app-loader-pitch::after {
      content: "";
      position: absolute;
      width: 15px;
      height: 15px;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      border: 1px solid rgba(8,118,109,.42);
      border-radius: 50%;
    }
    .app-loader-ball {
      position: absolute;
      z-index: 1;
      top: 50%;
      left: 7px;
      transform: translateY(-50%);
      font-size: 15px;
      line-height: 1;
      animation: app-loader-dribble 2.2s ease-in-out infinite alternate;
    }
    .app-loader-copy strong { display: block; color: #0b4f55; font-size: 14px; line-height: 1.3; }
    .app-loader-copy small { display: block; margin-top: 4px; color: #586879; font-size: 12px; line-height: 1.4; }
    @keyframes app-loader-dribble {
      from { left: 7px; transform: translateY(-50%) rotate(0deg); }
      to { left: calc(100% - 22px); transform: translateY(-50%) rotate(300deg); }
    }
    @media (prefers-reduced-motion: reduce) {
      .app-loader-ball { animation: none; left: calc(50% - 7px); }
    }
    @media (max-width: 720px) {
      .app-site-header {
        min-height: 62px;
        align-items: flex-start;
        padding: 12px 16px;
      }
      .app-site-header .app-brand-subtitle,
      .app-site-header .app-section-label { display: none; }
      .app-tool-header {
        width: min(100% - 24px, 560px);
        margin-top: 12px;
        border-radius: 12px;
      }
      .app-page-main {
        width: min(100% - 24px, 560px);
        padding: 14px 0 40px;
      }
    }
    """


def app_site_header(section_label: str, *, home_url: str = "/") -> str:
    """Return the home-matching application header for a submenu."""

    return f"""
  <header class="site-header app-site-header">
    <a class="app-brand" href="{escape(home_url, quote=True)}" aria-label="ACL Movement Analytics Lab home">
      <span class="app-brand-mark" aria-hidden="true"></span>
      <span class="app-brand-copy">
        <span class="app-brand-title">ACL Movement Analytics Lab</span>
        <span class="app-brand-subtitle">Women’s football movement research</span>
      </span>
    </a>
    <span class="app-section-label">{escape(section_label)}</span>
  </header>"""
