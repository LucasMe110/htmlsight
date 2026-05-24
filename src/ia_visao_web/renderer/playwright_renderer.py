import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sync_playwright: Any
try:
    from playwright.sync_api import sync_playwright  # type: ignore[no-redef]
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
        _set_default_browser_path()
        if sync_playwright is None:
            raise RendererUnavailableError(
                "playwright não está instalado. Instale com "
                "`uv add playwright` ou `python -m pip install playwright`, "
                "depois rode `python -m playwright install chromium`."
            )

    def render_html(self, html: str, viewport: tuple[int, int]) -> RenderedPage:
        with self.open_page(html, viewport) as page:
            png = page.screenshot(full_page=True)
        return RenderedPage(png=png, viewport=viewport)

    @contextmanager
    def open_page(self, html: str, viewport: tuple[int, int]) -> Iterator[Any]:
        if sync_playwright is None:
            raise RendererUnavailableError(
                "playwright não está instalado. Instale com "
                "`uv add playwright` ou `python -m pip install playwright`, "
                "depois rode `python -m playwright install chromium`."
            )

        width, height = viewport
        playwright = None
        browser = None
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="networkidle")
            page.evaluate("document.fonts && document.fonts.ready")
            page.wait_for_timeout(200)
            yield page
        except Exception as exc:  # pragma: no cover - requires browser runtime
            raise RendererUnavailableError(f"falha ao renderizar com playwright: {exc}") from exc
        finally:
            if browser is not None:
                browser.close()
            if playwright is not None:
                playwright.stop()


def _set_default_browser_path() -> None:
    local_browser_path = Path(sys.prefix) / "ms-playwright"
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and local_browser_path.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(local_browser_path)
