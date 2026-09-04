from __future__ import annotations

from acl_motion.annotations.registry import default_annotation_cases
from acl_motion.cases.models import InjurySide
from acl_motion.ui.annotation import _case_payload, render_annotation_page
from acl_motion.ui.app_shell import BRAND_ASSET_DIR, brand_asset_path
from acl_motion.ui.comparison import render_comparison_page
from acl_motion.ui.exploration import (
    render_exploration_page,
    render_feature_correlations_page,
)
from acl_motion.ui.home import render_home_page
from acl_motion.ui.results import render_results_page
from acl_motion.ui.similarity_validation import render_similarity_validation_page
from acl_motion.ui.video_cutter import render_video_cutter_page


def test_home_page_connects_current_workflow_tools() -> None:
    html = render_home_page()

    assert "ACL Movement Analytics Lab" in html
    assert "Women’s football movement research" in html
    assert "Observe the movement. Follow the evidence." in html
    assert "Not diagnosis, injury-risk calculation, or causation" in html
    assert 'href="#workflow">Skip to analysis workflow</a>' in html
    assert '<div class="hero-actions"' not in html
    assert 'class="button hero-primary"' not in html
    assert 'class="button hero-secondary"' not in html
    assert "Core analysis workflow" in html
    assert "Create, annotate, and review a case" in html
    assert "Follow the evidence trail" not in html
    assert 'aria-label="Four-stage evidence workflow"' not in html
    assert "Stage 01 · Observe" not in html
    assert "guided demo" not in html.lower()
    assert "4-minute demo" not in html.lower()
    assert "four-minute rule" not in html.lower()
    assert "opening context" not in html.lower()
    assert "unless a judge asks" not in html.lower()
    assert "Responsible AI in practice" in html
    assert "Knowing when not to give an absolute answer" in html
    assert "evidence-gated restraint built into the workflow" in html
    assert 'aria-label="Responsible AI safeguards"' in html
    assert "Humans retain contextual judgement" in html
    assert "Uncertainty changes the output" in html
    assert "Comparison must be earned" in html
    assert 'aria-label="Real case examples of evidence-gated outcomes"' in html
    assert "Andi Sullivan" in html
    assert "Charlotte Newsham" in html
    assert "Jordyn Huitema" in html
    assert "Caroline Weir" in html
    assert 'href="/results?case=imported_andi_sullivan_2024_10_06_view_02"' in html
    assert 'href="/results?case=imported_charlotte_newsham_2026_05_02_view_01"' in html
    assert 'href="/results?case=imported_jordyn_huitema_2026_07_18_view_01"' in html
    assert 'href="/results?case=imported_caroline_wier_2023_09_26_view_02"' in html
    assert "No transition” is a valid result, not a processing failure" in html
    assert "phase analysis is withheld instead of manufacturing a confident story" in html
    assert "Injury Case Library" in html
    assert 'id="caseSearch"' in html
    assert 'id="caseFilter"' in html
    assert 'id="caseSort"' in html
    assert '<option value="analysis_desc">Recently analysed (unfinished first)</option>' in html
    assert '<option value="player_asc">Player name (A–Z)</option>' in html
    assert 'role="group" aria-label="Injury cases"' in html
    assert "Reviewing the case line-up…" in html
    assert 'class="app-loader-pitch"' in html
    assert "@keyframes app-loader-dribble" in html
    assert 'aria-pressed="' in html
    assert 'id="selectedCaseName"' in html
    assert 'id="caseFacts"' in html
    assert 'id="caseViews"' in html
    assert 'id="caseProgress" role="progressbar"' in html
    assert 'id="continueCaseButton"' in html
    assert "function nextCaseAction(item)" in html
    assert "Generate next analysis" in html
    assert "Annotate next clip" in html
    assert "clips analysed" in html
    assert "Add video view" in html
    assert "Edit case details" in html
    assert 'id="deleteCaseButton"' in html
    assert "Delete case" in html
    assert "Delete view" in html
    assert "View analysis" in html
    assert "Annotate view" in html
    assert "Create, annotate, and review a case" in html
    assert 'aria-label="Case analysis workflow"' in html
    assert "STEP 01" in html
    assert "Create or choose the injury case" in html
    assert 'href="/video-cutter"' in html
    assert "127.0.0.1:8770" not in html
    assert "STEP 02" in html
    assert "Cut and attach video views" in html
    assert "STEP 03" in html
    assert "Annotate, verify and generate" in html
    assert 'aria-label="Inside the annotation workspace"' in html
    assert "<li>Annotate</li><li>Verify</li><li>Generate</li>" in html
    assert 'href="/annotate"' in html
    assert "STEP 04" in html
    assert "Review the Movement Story" in html
    assert "Continue in annotation" not in html
    assert "Compare movements" in html
    assert "Comparison rule:" in html
    assert "mutually supported case-level measurements" in html
    assert 'href="/compare"' in html
    assert html.count('href="/compare"') == 1
    assert "Open Compare Movements" in html
    assert "Explore the dataset" in html
    assert 'href="/explore"' in html
    assert html.count('href="/explore"') == 1
    assert "Open Statistical Explorer" in html
    assert 'href="/explore#sources"' in html
    assert "Sources / Injury Reports" in html
    assert "Feature correlations" not in html
    assert 'href="/correlations"' not in html
    assert "Open Feature Correlations" not in html
    assert html.index("Create, annotate, and review a case") < html.index("Injury Case Library")
    assert html.index("Injury Case Library") < html.index("Responsible AI in practice")
    assert html.index("Responsible AI in practice") < html.index("Continue from one story to the wider library")
    assert 'role="status" aria-live="polite"' in html
    assert ":focus-visible" in html
    assert "prefers-reduced-motion: reduce" in html


def test_approved_brand_identity_is_shared_without_repeating_the_hero() -> None:
    home = render_home_page()
    tool_pages = (
        render_annotation_page(),
        render_results_page(),
        render_comparison_page(),
        render_exploration_page(),
        render_video_cutter_page(main_menu_url="/"),
    )

    assert "/assets/brand/acl_movement_analytics_lab_hero_banner.png" in home
    assert "/assets/brand/acl_brand_tagline.png" in home
    assert "#0A2540" in home
    assert "#0F62FE" in home
    assert "#00D4A6" in home
    assert "#7CF1BB" in home
    assert "#E6F6FF" in home
    for html in (home, *tool_pages):
        assert "/assets/brand/acl_favicon_runner_32.png" in html
        assert "/assets/brand/acl_badge_pitch_runner_analytics.png" in html
    for html in tool_pages:
        assert "acl_movement_analytics_lab_hero_banner.png" not in html


def test_brand_assets_are_local_and_static_resolution_is_constrained() -> None:
    expected = {
        "acl_badge_acl_trajectory.png",
        "acl_badge_kinematic_angle.png",
        "acl_badge_pitch_runner_analytics.png",
        "acl_brand_colour_palette.png",
        "acl_brand_tagline.png",
        "acl_favicon_acl_trajectory.png",
        "acl_favicon_kinematic_angle.png",
        "acl_favicon_runner.png",
        "acl_favicon_runner_32.png",
        "acl_favicon_runner_64.png",
        "acl_favicon_runner_180.png",
        "acl_movement_analytics_lab_hero_banner.png",
    }

    assert expected <= {path.name for path in BRAND_ASSET_DIR.iterdir()}
    assert brand_asset_path("/favicon.ico") == BRAND_ASSET_DIR / "acl_favicon_runner_32.png"
    assert brand_asset_path("/assets/brand/acl_brand_tagline.png") == (
        BRAND_ASSET_DIR / "acl_brand_tagline.png"
    )
    assert brand_asset_path("/assets/brand/../app_shell.py") is None


def test_results_page_preserves_analysis_when_cutting_another_subclip() -> None:
    html = render_results_page()

    assert 'id="addCaseViewButton" href="/video-cutter">Cut another subclip</a>' in html
    assert "const currentAnalysisUrl = window.location.pathname + window.location.search + window.location.hash;" in html
    assert "'&return=' + encodeURIComponent(currentAnalysisUrl)" in html


def test_home_page_exposes_accessible_case_states_without_primary_uuid_labels() -> None:
    html = render_home_page()

    assert 'aria-busy="true"' in html
    assert "Loading injury cases" in html
    assert "No injury cases yet" in html
    assert "No matching injury cases" in html
    assert 'id="analysisError" role="alert" hidden' in html
    assert "Try again" in html
    assert "Player not recorded" in html
    assert 'return "Imported movement clip"' in html
    assert "Technical case metadata" in html
    assert "Case ID:" in html
    assert "overflow-wrap: anywhere" in html


def test_home_page_uses_existing_api_routes_and_links_gated_comparison() -> None:
    html = render_home_page()

    assert 'fetch("/api/cases?include_video_metadata=0")' in html
    assert 'fetch("/api/explore/summary")' in html
    assert "loadCases().finally(loadEvidenceSummary);" in html
    assert 'href="/results?case=' in html
    assert "results_available" in html
    assert "annotation_saved" in html
    assert "function groupCases(views)" in html
    assert "function compareCases(left, right, sort)" in html
    assert "leftNeedsWork !== rightNeedsWork" in html
    assert "const orderedCases = [...app.cases].sort(" in html
    assert '$("caseSort").addEventListener("change", renderCaseList);' in html
    assert "latest_analysis_at" in html
    assert '"/video-cutter?case="' in html
    assert '"&return=" + encodeURIComponent(caseReturnPath)' in html
    assert 'fetch("/api/cases/delete"' in html
    assert "comparable_case_count" in html
    assert "pairwise_output_count" in html
    assert 'id="similarityBadge"' in html
    assert 'id="pairwiseCount"' in html
    assert "Inspect evidence readiness" not in html
    assert "Similarity Analysis" not in html
    assert 'aria-disabled="true"' not in html


def test_home_page_exposes_completed_views_inside_selected_case_only() -> None:
    html = render_home_page()

    assert 'id="caseAnalysisShortcuts"' in html
    assert "Open any available angle" in html
    assert "function renderSelectedAnalysisShortcuts(item)" in html
    assert "item.views.filter(view => view.results_available)" in html
    assert 'class="case-analysis-shortcut" href="/results?case=' in html
    assert "View analysis →" in html
    assert "Browse completed analyses" not in html
    assert 'id="analysisDirectory"' not in html
    assert "function renderAnalysisDirectory()" not in html


def test_case_payload_exposes_completed_analysis_state() -> None:
    payload = _case_payload(default_annotation_cases()[0])

    assert payload["slug"] == "christen_press"
    assert payload["results_available"] is True
    assert payload["analysis_generated_at"] is not None
    assert payload["annotation_saved"] is False
    assert payload["video_path"] == "02_YvnMYc6OdT8_160s-166s.mp4"
    assert "/Users/" not in payload["video_path"]


def test_builtin_cases_preserve_supplied_injury_laterality() -> None:
    cases = {case.case_id: case for case in default_annotation_cases()}

    assert cases["case_01_acl_candidate"].injured_side is InjurySide.LEFT
    assert cases["christen_press_acl"].injured_side is InjurySide.RIGHT
    assert "human_operator" in cases["case_01_acl_candidate"].injury_laterality_source
    assert "human_operator" in cases["christen_press_acl"].injury_laterality_source


def test_case_payload_can_skip_video_metadata_for_the_case_library() -> None:
    payload = _case_payload(
        default_annotation_cases()[0],
        include_video_metadata=False,
    )

    assert payload["metadata"] is None
    assert payload["video_available"] is True


def test_case_payload_prefers_saved_case_identity_over_imported_filename() -> None:
    payload = _case_payload(
        default_annotation_cases()[0],
        display_details={
            "player_name": "Beth Mead",
            "injury_date": "2022-11-19",
            "team": "Arsenal",
            "opponent": "Manchester United",
            "competition": "WSL",
            "position_group": "forward",
            "match_minute": "90",
        },
    )

    assert payload["player_name"] == "Beth Mead"
    assert payload["team"] == "Arsenal"
    assert payload["injury_date"] == "2022-11-19"
    assert payload["case_details"]["opponent"] == "Manchester United"


def test_tool_pages_return_to_main_menu() -> None:
    assert 'class="app-brand" href="/"' in render_annotation_page()
    assert 'class="app-brand" href="/"' in render_results_page()
    assert "http://127.0.0.1:8765/" in render_video_cutter_page()
    assert 'class="app-brand" href="/"' in render_comparison_page()
    assert 'class="app-brand" href="/"' in render_exploration_page()
    assert 'class="app-brand" href="/"' in render_feature_correlations_page()
    assert '<a class="button" href="/">Main menu</a>' in render_similarity_validation_page()


def test_primary_tools_use_football_loading_messages() -> None:
    assert "Reviewing the replay…" in render_results_page()
    assert "Checking the match-ups…" in render_comparison_page()
    assert "Surveying the pitch…" in render_exploration_page()
    assert "Warming up the video player…" in render_video_cutter_page()
    assert "Reviewing the replay and building the movement analysis…" in render_annotation_page()


def test_submenus_share_the_home_application_shell() -> None:
    pages = {
        "Human Annotation": render_annotation_page(),
        "Movement Analysis": render_results_page(),
        "Compare Movements": render_comparison_page(),
        "Explore Data": render_exploration_page(),
        "Feature Correlations": render_feature_correlations_page(),
        "Similarity Validation": render_similarity_validation_page(),
        "Video Cutter": render_video_cutter_page(main_menu_url="/"),
    }

    for section_label, html in pages.items():
        assert 'class="site-header app-site-header"' in html
        assert 'class="app-brand" href="/"' in html
        assert "Women’s football movement research" in html
        assert f'class="app-section-label">{section_label}</span>' in html
        assert 'class="app-page-main"' in html or "app-page-main" in html
        assert "__APP_SHELL_CSS__" not in html
        assert "__APP_SITE_HEADER__" not in html


def test_similarity_validation_page_hides_algorithm_output_during_review() -> None:
    html = render_similarity_validation_page()

    assert "Blinded expert similarity review" in html
    assert 'id="queryVideo"' in html
    assert 'data-choice="OPTION_A"' in html
    assert 'data-choice="UNABLE_TO_JUDGE"' in html
    assert "/api/similarity-validation/assignment" in html
    assert "/api/similarity-validation/judgement" in html
    assert "algorithm scores, rankings, or case metadata" in html
    assert html.index(".app-site-header") < html.index("</style>")
    assert html.index("</style>") < html.index("<body>")
    assert 'href="#mainContent">Skip to review workspace</a>' in html


def test_integrated_video_cutter_uses_main_app_routes() -> None:
    html = render_video_cutter_page(
        main_menu_url="/",
        api_base="/video-cutter/api",
    )

    assert 'class="app-brand" href="/"' in html
    assert 'const apiBase = "/video-cutter/api";' in html
    assert 'fetch(`${apiBase}/videos`)' in html
    assert 'fetch(`${apiBase}/context-cases`)' in html
    assert 'fetch(`${apiBase}/open-path`' in html
    assert 'fetch(`${apiBase}/cut`' in html
    assert "const requestedParams = new URLSearchParams(window.location.search);" in html
    assert 'const requestedCaseId = requestedParams.get("case");' in html
    assert 'const requestedVideoRef = requestedParams.get("video");' in html
    assert '$("analysisCaseSelect").value = requestedCase.case_id;' in html
    assert "127.0.0.1:8770" not in html


def test_exploration_page_exposes_evidence_gated_views() -> None:
    html = render_exploration_page()

    assert "Explore Data" in html
    assert "Overview" in html
    assert "Compare cases" in html
    assert "Measurement correlations" in html
    assert "Measurement Correlation Map" in html
    assert 'id="correlationMap"' in html
    assert 'id="correlationStatistic"' in html
    assert "MINIMUM_CORRELATION_CASES = 5" in html
    assert "Every paired value represents one independent injury event" in html
    assert "directional-angle pairs are withheld" in html
    assert "Can these groups be compared?" in html
    assert "Sources / Injury Reports" in html
    assert 'id="sourceSearch"' in html
    assert 'id="mechanismMethodology"' in html
    assert "Explicit source wording" in html
    assert 'id="smallSampleBanner"' in html
    assert "Every point is one injury case, never a video frame or replay." in html
    assert "Extra frames and replay views do not increase the number of cases." in html
    assert "Movement Similarity" in html
    assert 'id="similarityStatus"' in html
    assert "Missing or limited measurements are listed below and are not plotted as zero." in html
    assert "function valueAvailable(value)" in html
    assert "statistic === \"pre_late_change\"" in html
    assert "drawLinearTicks" in html
    assert "Dashed line = median; box = middle 50%; diamond = mean" in html
    assert "Case breakdowns" in html
    assert 'id="breakdownBarCanvas"' in html
    assert 'id="breakdownPieCanvas"' in html
    assert '<option value="preferred_foot_knee_injured">Preferred-foot knee injured</option>' in html
    assert 'preferred_foot_knee_injured: "whether the preferred-foot knee was injured"' in html
    assert 'id="profilesTab"' in html
    assert 'id="heightCanvas" class="tall-chart"' in html
    assert 'id="weightCanvas" class="tall-chart"' in html
    assert 'id="biometricScatterCanvas"' in html
    assert "missing or conflicting values are never estimated or plotted as zero" in html
    assert "function renderProfiles()" in html
    assert 'value="preferred_foot_knee_injured"' in html
    assert "Sample SD" in html
    assert "if (!app.data) return;" in html
    assert "renderBreakdowns();\n        renderProfiles();\n        redrawActiveCharts();" in html
    assert 'id="distributionFeatureHelp"' in html
    assert 'id="relationshipFeatureHelp"' in html
    assert "renderMeasurementHelp" in html
    assert 'role="tablist"' in html
    assert html.count('role="tabpanel"') == 7
    assert 'aria-controls="sourcesView"' in html
    assert 'aria-labelledby="sourcesTab"' in html
    assert 'aria-controls="distributionView"' in html
    assert 'aria-labelledby="distributionTab"' in html
    assert 'item.tabIndex = selected ? 0 : -1;' in html
    assert 'fetch("/api/explore", {signal: controller.signal})' in html
    assert 'id="retryExploreData"' in html


def test_feature_correlation_submenu_is_separate_from_case_similarity() -> None:
    html = render_feature_correlations_page()

    assert "<title>Feature Correlations - ACL Movement Analytics Lab</title>" in html
    assert "<h1>Feature Correlations</h1>" in html
    assert 'data-initial-view="correlations"' in html
    assert "feature-by-feature analysis, independent of movement-similarity rankings" in html
    assert "Negative correlation" in html
    assert "Positive correlation" in html
    assert 'id="correlationMap"' in html


def test_compare_movements_page_owns_similarity_experience() -> None:
    html = render_comparison_page()

    assert "Compare Movements" in html
    assert "Choose an injury case" in html
    assert 'id="comparisonLens"' in html
    assert 'id="simpleMode"' in html
    assert 'id="scientificMode"' in html
    assert 'id="engineNote" hidden' in html
    assert 'mode !== "scientific"' in html
    assert 'mode === "scientific" ? `<div class="score-box">' in html
    assert "Weighted robust movement difference" in html
    assert "robustly scaled L1 distance" in html
    assert "Weighted cosine sensitivity" in html
    assert "soft cosine" in html
    assert "Euclidean" in html
    assert "How each comparison lens works" in html
    assert "How evidence support is judged" in html
    assert "Validation work" in html
    assert 'href="/validate-similarity"' in html
    assert "not diagnostic confidence" in html
    assert "Similarity score / 100" not in html
    assert 'id="playerSearch"' in html
    assert 'id="playerList"' in html
    assert 'id="rankingList"' in html
    assert 'id="similaritySpectrum"' in html
    assert 'id="similarityMatrix"' in html
    assert 'id="similarityMatrixDisclosure"' in html
    assert "function renderSimilaritySpectrum()" in html
    assert "function renderSimilarityMatrix()" in html
    assert "Unavailable cells are not zero" in html
    assert 'id="neighbourhoodMapButton"' not in html
    assert 'id="fullNetworkMapButton"' not in html
    assert 'id="measurementFilterOptions"' in html
    assert "Keep at least ${minimum} movement areas selected." in html
    assert 'groups=${encodeURIComponent(app.measurementGroups.join(","))}' in html
    assert "Closest injury-event movement profiles" in html
    assert "Views are never averaged." in html
    assert "Best supported-view match" in html
    assert "Query-only · comparison depends on shared evidence" in html
    assert "Query-only · more supported evidence needed" in html
    assert "eligible_view_pair_count" in html
    assert "selected_value" in html
    assert "candidate_value" in html
    assert "Most similar players" not in html
    assert 'id="poseProfilePool"' not in html
    assert "selectedPosePoolId" not in html
    assert "/api/movement-comparison?case=" in html


def test_compare_movements_page_ignores_stale_case_responses() -> None:
    html = render_comparison_page()

    assert "comparisonRequestId: 0" in html
    assert "const requestId = ++app.comparisonRequestId;" in html
    assert (
        "if (requestId !== app.comparisonRequestId || caseId !== app.selectedCaseId) return;"
        in html
    )
    assert "loading comparison…" in html


def test_results_page_does_not_contain_comparison_lens_workspace() -> None:
    html = render_results_page()

    assert 'id="comparisonLens"' not in html
    assert 'id="similarityLensSelect"' not in html
    assert "How each comparison lens works" not in html
    assert "How comparison reliability is judged" not in html


def test_results_annotation_link_uses_annotation_route() -> None:
    html = render_results_page()

    assert "'/annotate?case='" in html
    assert "'/?case='" in html
    assert 'id="annotateNextClipButton"' in html
    assert 'id="caseClipsButton"' in html


def test_annotation_page_returns_to_the_selected_case_clip_list() -> None:
    html = render_annotation_page()

    assert 'id="caseClipsLink"' in html
    assert '`/?case=${encodeURIComponent(app.currentCase.case_id)}`' in html
