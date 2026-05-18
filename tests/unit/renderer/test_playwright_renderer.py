import pytest

from ia_visao_web.renderer.playwright_renderer import PlaywrightRenderer, RendererUnavailableError


def test_renderer_reports_missing_playwright_with_actionable_error(monkeypatch):
    monkeypatch.setattr("ia_visao_web.renderer.playwright_renderer.sync_playwright", None)

    with pytest.raises(RendererUnavailableError, match="playwright"):
        PlaywrightRenderer()


def test_renderer_closes_browser_when_page_use_fails(monkeypatch):
    browser = FakeBrowser()
    playwright = FakePlaywright(browser)
    monkeypatch.setattr(
        "ia_visao_web.renderer.playwright_renderer.sync_playwright",
        lambda: FakeSyncPlaywright(playwright),
    )

    renderer = PlaywrightRenderer()

    with pytest.raises(RendererUnavailableError, match="falha ao renderizar"):
        with renderer.open_page("<html></html>", (100, 100)):
            raise RuntimeError("boom")

    assert browser.closed
    assert playwright.stopped


class FakeSyncPlaywright:
    def __init__(self, playwright):
        self.playwright = playwright

    def start(self):
        return self.playwright


class FakePlaywright:
    def __init__(self, browser):
        self.chromium = FakeChromium(browser)
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeChromium:
    def __init__(self, browser):
        self.browser = browser

    def launch(self, headless):
        assert headless is True
        return self.browser


class FakeBrowser:
    def __init__(self):
        self.closed = False

    def new_page(self, viewport):
        return FakePage(viewport)

    def close(self):
        self.closed = True


class FakePage:
    def __init__(self, viewport):
        self.viewport = viewport

    def set_content(self, html, wait_until):
        assert html
        assert wait_until == "networkidle"

    def evaluate(self, expression):
        assert expression

    def wait_for_timeout(self, milliseconds):
        assert milliseconds == 200
