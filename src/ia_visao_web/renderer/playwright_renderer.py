from dataclasses import dataclass

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - tested by monkeypatching the module global
    sync_playwright = None


class RendererUnavailableError(RuntimeError):
    """Raised when Playwright or its browser runtime is unavailable."""


@dataclass(frozen=True)
class RenderedPage:
    png: bytes
    viewport: tuple[int, int]


class PlaywrightRenderer:
    def __init__(self) -> None:
        if sync_playwright is None:
            raise RendererUnavailableError(
                "playwright não está instalado. Instale com "
                "`uv add playwright` ou `python -m pip install playwright`, "
                "depois rode `python -m playwright install chromium`."
            )

    def render_html(self, html: str, viewport: tuple[int, int]) -> RenderedPage:
        width, height = viewport
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": width, "height": height})
                page.set_content(html, wait_until="networkidle")
                page.evaluate("document.fonts && document.fonts.ready")
                page.wait_for_timeout(200)
                png = page.screenshot(full_page=True)
                browser.close()
        except Exception as exc:  # pragma: no cover - requires browser runtime
            raise RendererUnavailableError(f"falha ao renderizar com playwright: {exc}") from exc
        return RenderedPage(png=png, viewport=viewport)
