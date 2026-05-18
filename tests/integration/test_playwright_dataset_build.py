from io import BytesIO

import pytest
from PIL import Image

from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.dom_walker import DomWalker
from ia_visao_web.labeler.walker import filter_matches
from ia_visao_web.renderer.playwright_renderer import PlaywrightRenderer, RendererUnavailableError


def test_playwright_render_dom_walk_and_write_dataset_sample(tmp_path):
    try:
        renderer = PlaywrightRenderer()
        with renderer.open_page(_HTML, (800, 600)) as page:
            raw_matches = DomWalker().collect(page)
            png = page.screenshot(full_page=True)
    except RendererUnavailableError as exc:
        pytest.skip(f"Playwright/Chromium indisponivel: {exc}")

    image = Image.open(BytesIO(png)).convert("RGB")
    detections = filter_matches(
        raw_matches,
        viewport_width=image.width,
        viewport_height=image.height,
    )
    assert detections

    DatasetWriter(tmp_path).write_sample("playwright-sample", image, detections, split="train")

    label_lines = (tmp_path / "labels/train/playwright-sample.txt").read_text().splitlines()
    attrs_text = (tmp_path / "attrs/train/playwright-sample.json").read_text()

    assert label_lines
    assert attrs_text.count('"tag"') == len(label_lines)


_HTML = """<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <style>
      .navbar { display: flex; width: 760px; height: 56px; background: #0d6efd; }
      .btn { display: inline-block; width: 100px; height: 34px; }
      .card { display: block; width: 240px; height: 160px; margin-top: 24px; }
      .form-control { display: block; width: 300px; height: 40px; }
    </style>
  </head>
  <body>
    <nav class="navbar" role="navigation">
      <button class="btn" type="button">Entrar</button>
    </nav>
    <main class="container">
      <div class="card">
        <p>Texto do card</p>
        <input class="form-control" type="email">
      </div>
    </main>
  </body>
</html>
"""
