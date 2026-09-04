"""Shared application shell for ACL Movement Analytics Lab tool pages."""

from __future__ import annotations

from html import escape
from pathlib import Path

BRAND_ASSET_URL_PREFIX = "/assets/brand/"
BRAND_ASSET_DIR = Path(__file__).resolve().parent / "static" / "brand"


def brand_asset_path(request_path: str) -> Path | None:
    """Resolve a public brand asset without exposing arbitrary local files."""

    if request_path == "/favicon.ico":
        filename = "acl_favicon_runner_32.png"
    elif request_path.startswith(BRAND_ASSET_URL_PREFIX):
        filename = request_path.removeprefix(BRAND_ASSET_URL_PREFIX)
    else:
        return None
    if not filename or "/" in filename or "\\" in filename:
        return None
    candidate = BRAND_ASSET_DIR / filename
    return candidate if candidate.is_file() else None


def app_brand_head() -> str:
    """Return shared browser identity metadata for every application page."""

    return f"""
  <meta name="theme-color" content="#0A2540" />
  <link rel="icon" type="image/png" sizes="32x32" href="{BRAND_ASSET_URL_PREFIX}acl_favicon_runner_32.png" />
  <link rel="icon" type="image/png" sizes="64x64" href="{BRAND_ASSET_URL_PREFIX}acl_favicon_runner_64.png" />
  <link rel="apple-touch-icon" sizes="180x180" href="{BRAND_ASSET_URL_PREFIX}acl_favicon_runner_180.png" />"""


def apply_app_brand(html: str) -> str:
    """Inject shared browser identity metadata into one rendered page."""

    return html.replace("</head>", f"{app_brand_head()}\n</head>", 1)


def app_shell_css() -> str:
    """Return the home-page header and page-frame styles used by submenus."""

    return r"""
    :root {
      --brand-navy: #0A2540;
      --brand-blue: #0F62FE;
      --brand-teal: #00D4A6;
      --brand-mint: #7CF1BB;
      --brand-ice: #E6F6FF;
    }
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
    .app-visually-hidden {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
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
      background: var(--brand-navy);
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
    .app-site-header .app-brand-badge {
      width: 66px;
      height: 47px;
      flex: 0 0 auto;
      display: block;
      object-fit: contain;
      filter: drop-shadow(0 5px 10px rgba(0,0,0,.18));
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
      border: 1px solid #b9ded8;
      border-radius: 12px;
      background: linear-gradient(135deg, #f8fffd, var(--brand-ice));
      color: #142334;
      text-align: left;
    }
    .app-football-loader.compact {
      grid-template-columns: 66px minmax(0, 1fr);
      min-height: 70px;
      padding: 10px 11px;
    }
    .app-loader-pitch {
      width: 72px;
      height: 51px;
      border: 0;
      background: transparent url("/assets/brand/acl_badge_pitch_runner_analytics.png") center / contain no-repeat;
      filter: drop-shadow(0 5px 9px rgba(10,37,64,.17));
      animation: app-loader-breathe 1.8s ease-in-out infinite alternate;
    }
    .compact .app-loader-pitch { width: 58px; height: 41px; }
    .app-loader-ball {
      display: none;
    }
    .app-loader-copy strong { display: block; color: #0b4f55; font-size: 14px; line-height: 1.3; }
    .app-loader-copy small { display: block; margin-top: 4px; color: #586879; font-size: 12px; line-height: 1.4; }
    @keyframes app-loader-dribble {
      from { left: 7px; transform: translateY(-50%) rotate(0deg); }
      to { left: calc(100% - 22px); transform: translateY(-50%) rotate(300deg); }
    }
    @keyframes app-loader-breathe {
      from { opacity: .72; transform: translateY(1px); }
      to { opacity: 1; transform: translateY(-1px); }
    }
    @media (prefers-reduced-motion: reduce) {
      .app-loader-pitch { animation: none; }
    }
    @media (max-width: 720px) {
      .app-site-header {
        min-height: 62px;
        align-items: flex-start;
        padding: 12px 16px;
      }
      .app-site-header .app-brand-subtitle,
      .app-site-header .app-section-label { display: none; }
      .app-site-header .app-brand-badge { width: 54px; height: 38px; }
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
      <img class="app-brand-badge" src="{BRAND_ASSET_URL_PREFIX}acl_badge_pitch_runner_analytics.png" alt="" width="66" height="47" />
      <span class="app-brand-copy">
        <span class="app-brand-title">ACL Movement Analytics Lab</span>
        <span class="app-brand-subtitle">Women’s football movement research</span>
      </span>
    </a>
    <span class="app-section-label">{escape(section_label)}</span>
  </header>"""
