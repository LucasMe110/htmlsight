# IA Visao Web MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that generates Bootstrap-like web screenshots, auto-labels visible components, writes YOLO-compatible datasets with HTML attribute sidecars, validates the dataset, exposes model/eval/predict scaffolding, and verifies the behavior with TDD.

**Architecture:** The project is a `src/` Python package with small modules for sources, renderer, labeler, dataset, model, eval, and CLI. Dataset artifacts are exchanged through disk using YOLO text labels plus JSON sidecars, while model code exposes stable interfaces that can be backed by real Ultralytics training when heavy dependencies are installed.

**Tech Stack:** Python 3.11+, pytest, typer, jinja2, faker, pillow, pyyaml, playwright optional, torch/ultralytics optional, pycocotools optional. Target dependency manager is `uv`; this machine currently does not have `uv` installed.

**Execution note:** The user explicitly requested no `git init` and no commits for this run. Commit steps from the Superpowers workflow are intentionally disabled.

---

## File Structure

- `pyproject.toml`: package metadata, CLI entry point, dependencies, pytest/ruff/mypy config.
- `README.md`: concise project usage.
- `CLAUDE.md`: commands, folder structure, architectural decisions, and environment gotchas.
- `.gitignore`: generated data, caches, model runs, virtualenvs.
- `src/ia_visao_web/cli.py`: Typer CLI and command wiring.
- `src/ia_visao_web/sources/generator.py`: deterministic Bootstrap HTML generator.
- `src/ia_visao_web/renderer/playwright_renderer.py`: Playwright screenshot renderer with deterministic waiting.
- `src/ia_visao_web/labeler/selectors.py`: taxonomy and selector priority map.
- `src/ia_visao_web/labeler/geometry.py`: bbox clipping, normalization, IoU, duplicate filtering.
- `src/ia_visao_web/labeler/walker.py`: DOM walker interface and filtering of raw detections.
- `src/ia_visao_web/dataset/splits.py`: deterministic train/val/test split by SHA1.
- `src/ia_visao_web/dataset/writer.py`: writes images, YOLO labels, sidecar attrs, and `data.yaml`.
- `src/ia_visao_web/dataset/validator.py`: validates counts, sidecar alignment, split coverage, and QA overlays.
- `src/ia_visao_web/model/vocab.py`: fixed vocab building and unknown-to-`other` mapping.
- `src/ia_visao_web/model/heads.py`: custom multitask head interface with torch implementation when available.
- `src/ia_visao_web/model/loss.py`: combined loss interface with torch implementation when available.
- `src/ia_visao_web/model/train.py`: training command boundary and dependency checks.
- `src/ia_visao_web/eval/metrics.py`: detection/attribute metric helpers for fixtures and optional pycocotools path.
- `src/ia_visao_web/eval/predict.py`: prediction JSON schema and optional model loading boundary.
- `tests/unit/...`: TDD unit tests per module.
- `tests/integration/...`: small end-to-end dataset pipeline test using fixtures.

---

### Task 1: Project Baseline and CLI Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `src/ia_visao_web/__init__.py`
- Create: `src/ia_visao_web/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from typer.testing import CliRunner

from ia_visao_web.cli import app


def test_cli_shows_top_level_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "dataset" in result.stdout
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "predict" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_cli.py -v`

Expected: FAIL because package or CLI does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml`, package skeleton, and Typer commands. Dataset/model commands may raise clear dependency/runtime errors until later tasks fill them in.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Update docs**

Add `CLAUDE.md` with commands known so far and note that `uv` is missing locally.

---

### Task 2: Taxonomy, Selectors, and Geometry

**Files:**
- Create: `src/ia_visao_web/labeler/selectors.py`
- Create: `src/ia_visao_web/labeler/geometry.py`
- Test: `tests/unit/labeler/test_selectors.py`
- Test: `tests/unit/labeler/test_geometry.py`

- [ ] **Step 1: Write failing selector tests**

```python
from ia_visao_web.labeler.selectors import TAXONOMY, SELECTORS, class_id


def test_taxonomy_matches_spec_order():
    assert TAXONOMY == [
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ]


def test_button_selector_has_priority_over_link():
    classes = [rule.class_name for rule in SELECTORS]
    assert classes.index("button") < classes.index("link")
    assert class_id("button") == 0
```

- [ ] **Step 2: Write failing geometry tests**

```python
from ia_visao_web.labeler.geometry import BBox, dedupe_by_iou, normalize_bbox


def test_normalize_bbox_converts_pixels_to_yolo_center_format():
    assert normalize_bbox(BBox(10, 20, 30, 40), image_width=100, image_height=200) == (
        0.25, 0.2, 0.3, 0.2,
    )


def test_dedupe_by_iou_keeps_first_high_overlap_box():
    boxes = [BBox(0, 0, 100, 100), BBox(1, 1, 100, 100), BBox(200, 200, 10, 10)]
    assert dedupe_by_iou(boxes, threshold=0.95) == [boxes[0], boxes[2]]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/labeler/test_selectors.py tests/unit/labeler/test_geometry.py -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 4: Implement selectors and geometry**

Define immutable selector rules, taxonomy helpers, bbox dataclass, area, visibility clipping, IoU, normalization, and de-duplication.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/labeler/test_selectors.py tests/unit/labeler/test_geometry.py -v`

Expected: PASS.

---

### Task 3: Deterministic Source Generator

**Files:**
- Create: `src/ia_visao_web/sources/generator.py`
- Test: `tests/unit/sources/test_generator.py`

- [ ] **Step 1: Write failing tests**

```python
from ia_visao_web.sources.generator import BootstrapPageGenerator


def test_generator_is_deterministic_for_seed():
    first = BootstrapPageGenerator(seed=123).generate_page(page_id="sample")
    second = BootstrapPageGenerator(seed=123).generate_page(page_id="sample")

    assert first.html == second.html
    assert first.viewport == second.viewport


def test_generator_includes_core_bootstrap_components():
    page = BootstrapPageGenerator(seed=1).generate_page(page_id="components")

    assert "navbar" in page.html
    assert "card" in page.html
    assert "form-control" in page.html
    assert "alert" in page.html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/sources/test_generator.py -v`

Expected: FAIL because generator does not exist.

- [ ] **Step 3: Implement minimal generator**

Return a dataclass containing `page_id`, `html`, and `viewport`. Use seeded random choices and Jinja2 templates; fall back to deterministic text if Faker is unavailable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/sources/test_generator.py -v`

Expected: PASS.

---

### Task 4: DOM Walker Filtering

**Files:**
- Create: `src/ia_visao_web/labeler/walker.py`
- Test: `tests/unit/labeler/test_walker.py`

- [ ] **Step 1: Write failing tests**

```python
from ia_visao_web.labeler.walker import RawDomMatch, filter_matches


def test_filter_matches_drops_hidden_small_and_outside_elements():
    matches = [
        RawDomMatch("button", 0, 0, 20, 20, "button", "inline-block", None, False, 0, True),
        RawDomMatch("button", 0, 0, 2, 2, "button", "inline-block", None, False, 0, True),
        RawDomMatch("button", 10, 10, 20, 20, "button", "none", None, False, 0, True),
        RawDomMatch("button", 500, 500, 20, 20, "button", "inline-block", None, False, 0, False),
    ]

    filtered = filter_matches(matches, viewport_width=100, viewport_height=100)

    assert len(filtered) == 1
    assert filtered[0].class_name == "button"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/labeler/test_walker.py -v`

Expected: FAIL because walker does not exist.

- [ ] **Step 3: Implement walker filtering**

Create raw and labeled detection dataclasses. Filter by area, display/visibility, viewport intersection, and same-class IoU duplicate threshold.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/labeler/test_walker.py -v`

Expected: PASS.

---

### Task 5: Dataset Writer and Splitter

**Files:**
- Create: `src/ia_visao_web/dataset/splits.py`
- Create: `src/ia_visao_web/dataset/writer.py`
- Test: `tests/unit/dataset/test_splits.py`
- Test: `tests/unit/dataset/test_writer.py`

- [ ] **Step 1: Write failing tests**

```python
import json

from PIL import Image

from ia_visao_web.dataset.splits import split_for_id
from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection


def test_split_for_id_is_deterministic():
    assert split_for_id("abc") == split_for_id("abc")
    assert split_for_id("abc") in {"train", "val", "test"}


def test_writer_keeps_yolo_labels_and_attrs_aligned(tmp_path):
    image = Image.new("RGB", (100, 200), "white")
    detections = [
        LabeledDetection("button", 0, BBox(10, 20, 30, 40), {
            "tag": "button", "display": "inline-block", "role": None,
            "has_children": False, "n_descendants": 0,
        })
    ]

    writer = DatasetWriter(tmp_path)
    writer.write_sample("abc", image, detections, split="train")

    label_text = (tmp_path / "labels/train/abc.txt").read_text()
    attrs = json.loads((tmp_path / "attrs/train/abc.json").read_text())

    assert label_text.strip() == "0 0.250000 0.200000 0.300000 0.200000"
    assert attrs[0]["tag"] == "button"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/dataset/test_splits.py tests/unit/dataset/test_writer.py -v`

Expected: FAIL because dataset modules do not exist.

- [ ] **Step 3: Implement splitter and writer**

Write PNGs, YOLO label lines, JSON sidecars, and `data.yaml` with taxonomy names.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/dataset/test_splits.py tests/unit/dataset/test_writer.py -v`

Expected: PASS.

---

### Task 6: Dataset Validator and QA Overlays

**Files:**
- Create: `src/ia_visao_web/dataset/validator.py`
- Test: `tests/unit/dataset/test_validator.py`

- [ ] **Step 1: Write failing tests**

```python
from PIL import Image

from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.dataset.validator import DatasetValidator
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection


def test_validator_detects_label_attr_mismatch(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample("one", image, [
        LabeledDetection("button", 0, BBox(10, 10, 20, 20), {
            "tag": "button", "display": "inline-block", "role": None,
            "has_children": False, "n_descendants": 0,
        })
    ], split="train")
    (tmp_path / "attrs/train/one.json").write_text("[]")

    result = DatasetValidator(tmp_path, min_train_instances=1).validate()

    assert not result.ok
    assert any("sidecar" in error for error in result.errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/dataset/test_validator.py -v`

Expected: FAIL because validator does not exist.

- [ ] **Step 3: Implement validator**

Check label/attr counts, class coverage, minimum train instances, and optional QA overlay creation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/dataset/test_validator.py -v`

Expected: PASS.

---

### Task 7: Renderer Boundary

**Files:**
- Create: `src/ia_visao_web/renderer/playwright_renderer.py`
- Test: `tests/unit/renderer/test_playwright_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from ia_visao_web.renderer.playwright_renderer import PlaywrightRenderer, RendererUnavailableError


def test_renderer_reports_missing_playwright_with_actionable_error(monkeypatch):
    monkeypatch.setattr("ia_visao_web.renderer.playwright_renderer.sync_playwright", None)

    with pytest.raises(RendererUnavailableError, match="playwright"):
        PlaywrightRenderer()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/renderer/test_playwright_renderer.py -v`

Expected: FAIL because renderer does not exist.

- [ ] **Step 3: Implement renderer boundary**

Import Playwright optionally, expose `render_html(html, viewport)` returning image bytes and raw DOM matches, and provide clear install/browser error messages.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/renderer/test_playwright_renderer.py -v`

Expected: PASS.

---

### Task 8: Dataset Build CLI

**Files:**
- Modify: `src/ia_visao_web/cli.py`
- Test: `tests/integration/test_dataset_build_cli.py`

- [ ] **Step 1: Write failing integration test**

```python
from typer.testing import CliRunner

from ia_visao_web.cli import app


def test_dataset_build_synthetic_fixture_writes_expected_files(tmp_path):
    result = CliRunner().invoke(app, [
        "dataset", "build",
        "--output", str(tmp_path),
        "--count", "2",
        "--synthetic-only",
    ])

    assert result.exit_code == 0
    assert (tmp_path / "data.yaml").exists()
    assert len(list((tmp_path / "images").glob("*/*.png"))) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/integration/test_dataset_build_cli.py -v`

Expected: FAIL because command is not implemented.

- [ ] **Step 3: Implement minimal synthetic build path**

Use generator plus a deterministic fixture labeler path for CI-friendly synthetic images. Real Playwright rendering remains available via renderer module when dependencies are installed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/integration/test_dataset_build_cli.py -v`

Expected: PASS.

---

### Task 9: Vocab, Heads, and Loss Interfaces

**Files:**
- Create: `src/ia_visao_web/model/vocab.py`
- Create: `src/ia_visao_web/model/heads.py`
- Create: `src/ia_visao_web/model/loss.py`
- Test: `tests/unit/model/test_vocab.py`
- Test: `tests/unit/model/test_heads.py`
- Test: `tests/unit/model/test_loss.py`

- [ ] **Step 1: Write failing tests**

```python
from ia_visao_web.model.vocab import AttributeVocab


def test_vocab_maps_unknown_values_to_other():
    vocab = AttributeVocab.from_observations(tags=["button"], displays=["block"], roles=[None])

    assert vocab.encode_tag("missing") == vocab.tag_to_id["other"]
    assert vocab.encode_role(None) == vocab.role_to_id["none"]
```

```python
import pytest

from ia_visao_web.model.heads import MultiTaskHead, TorchUnavailableError


def test_multitask_head_requires_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr("ia_visao_web.model.heads.torch", None)

    with pytest.raises(TorchUnavailableError):
        MultiTaskHead(num_classes=17, tag_classes=4, display_classes=7, role_classes=8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/model/test_vocab.py tests/unit/model/test_heads.py tests/unit/model/test_loss.py -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement model interfaces**

Provide vocab persistence, optional torch multitask head, and combined loss with clear missing-dependency errors.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/model/test_vocab.py tests/unit/model/test_heads.py tests/unit/model/test_loss.py -v`

Expected: PASS.

---

### Task 10: Eval Metrics and Predict JSON

**Files:**
- Create: `src/ia_visao_web/eval/metrics.py`
- Create: `src/ia_visao_web/eval/predict.py`
- Modify: `src/ia_visao_web/cli.py`
- Test: `tests/unit/eval/test_metrics.py`
- Test: `tests/unit/eval/test_predict.py`

- [ ] **Step 1: Write failing tests**

```python
from ia_visao_web.eval.metrics import attribute_accuracy


def test_attribute_accuracy_counts_matching_values():
    result = attribute_accuracy(
        predicted=[{"tag": "button"}, {"tag": "div"}],
        expected=[{"tag": "button"}, {"tag": "span"}],
        key="tag",
    )

    assert result == 0.5
```

```python
from ia_visao_web.eval.predict import detection_to_json
from ia_visao_web.labeler.geometry import BBox


def test_detection_to_json_matches_cli_schema():
    payload = detection_to_json("button", 0.92, BBox(1, 2, 3, 4), {"tag": "button"})

    assert payload == {
        "class": "button",
        "score": 0.92,
        "bbox": [1, 2, 3, 4],
        "attrs": {"tag": "button"},
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/eval/test_metrics.py tests/unit/eval/test_predict.py -v`

Expected: FAIL because eval modules do not exist.

- [ ] **Step 3: Implement eval helpers and CLI JSON output**

Implement lightweight fixture-safe metrics and prediction serialization. Real model inference can fail with an actionable missing-artifact message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/eval/test_metrics.py tests/unit/eval/test_predict.py -v`

Expected: PASS.

---

### Task 11: Full Verification and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Run all tests**

Run: `python3 -m pytest -v`

Expected: PASS.

- [ ] **Step 2: Run lint**

Run: `python3 -m ruff check .`

Expected: PASS, if `ruff` is installed. If it is not installed, document `não testei lint: ruff não instalado`.

- [ ] **Step 3: Run typecheck**

Run: `python3 -m mypy src`

Expected: PASS, if `mypy` is installed. If it is not installed, document `não testei typecheck: mypy não instalado`.

- [ ] **Step 4: Update docs**

Ensure `CLAUDE.md` contains current commands, folder structure, architectural decisions, and gotchas discovered during execution.

---

## Self-Review

- Spec coverage: The plan covers generator, renderer boundary, labeler selectors/filtering, dataset writer/validator, model multitask interfaces, eval helpers, CLI, TDD tests, and docs. Heavy training quality targets are represented as architecture and command boundaries; real 3000-image generation and 100-epoch training require dependencies/hardware and are not a fast unit-test task.
- Placeholder scan: No task relies on "TBD" implementation language. Heavy external integrations are explicitly scoped as optional dependency boundaries with actionable errors.
- Type consistency: Detection dataclasses use `BBox`, `RawDomMatch`, and `LabeledDetection` consistently across labeler, dataset, and eval tasks.
