import pytest

from ia_visao_web.renderer.playwright_renderer import PlaywrightRenderer, RendererUnavailableError


def test_renderer_reports_missing_playwright_with_actionable_error(monkeypatch):
    monkeypatch.setattr("ia_visao_web.renderer.playwright_renderer.sync_playwright", None)

    with pytest.raises(RendererUnavailableError, match="playwright"):
        PlaywrightRenderer()
