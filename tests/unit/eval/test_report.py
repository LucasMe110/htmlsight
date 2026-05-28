import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ia_visao_web.eval.report import ModelReport, generate_report


def _make_val_results(names: dict[int, str]) -> MagicMock:
    val = MagicMock()
    val.box.map50 = 0.812
    val.box.map = 0.634
    val.box.mp = 0.801  # mean precision
    val.box.mr = 0.776  # mean recall
    val.box.maps = [0.72, 0.65, 0.58, 0.81, 0.77, 0.90, 0.55, 0.68, 0.83, 0.71,
                    0.79, 0.60, 0.85, 0.73, 0.66, 0.80, 0.74]
    val.box.ap50 = [0.85, 0.78, 0.70, 0.92, 0.88, 0.96, 0.67, 0.80, 0.94, 0.83,
                    0.91, 0.73, 0.95, 0.86, 0.79, 0.90, 0.87]
    val.names = names
    return val


def _make_mock_ultralytics(names: dict[int, str]) -> types.ModuleType:
    mod = types.ModuleType("ultralytics")
    mock_model = MagicMock()
    mock_model.val.return_value = _make_val_results(names)
    mock_model.info.return_value = (11_000_000, 28_500_000)  # params, flops
    mod.YOLO = MagicMock(return_value=mock_model)
    return mod


def _make_dataset(tmp_path: Path, names: dict[int, str]) -> Path:
    ds = tmp_path / "dataset"
    (ds / "images" / "test").mkdir(parents=True)
    (ds / "labels" / "test").mkdir(parents=True)
    (ds / "attrs" / "test").mkdir(parents=True)
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "val").mkdir(parents=True)

    # write 3 test images with labels
    from PIL import Image as PILImage
    for i in range(3):
        PILImage.new("RGB", (640, 480), "white").save(ds / "images" / "test" / f"img{i:03d}.png")
        (ds / "labels" / "test" / f"img{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (ds / "attrs" / "test" / f"img{i:03d}.json").write_text(
            '[{"tag":"button","display":"inline-block","role":null,"has_children":false}]'
        )

    for split in ("train", "val"):
        for i in range(2):
            PILImage.new("RGB", (640, 480), "white").save(ds / "images" / split / f"s{i}.png")

    nc = len(names)
    class_names = [names[i] for i in range(nc)]
    yaml_content = f"nc: {nc}\nnames: {class_names!r}\n"
    (ds / "data.yaml").write_text(yaml_content)
    return ds


def test_generate_report_returns_model_report(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")

    mock_ul = _make_mock_ultralytics(names)
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")

    assert isinstance(report, ModelReport)
    assert report.map50 == pytest.approx(0.812)
    assert report.map50_95 == pytest.approx(0.634)
    assert report.precision == pytest.approx(0.801)
    assert report.recall == pytest.approx(0.776)
    assert len(report.per_class) == 17
    assert report.dataset_stats["train_images"] == 2
    assert report.dataset_stats["val_images"] == 2
    assert report.dataset_stats["test_images"] == 3


def test_generate_report_per_class_sorted_by_map(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")

    monkeypatch.setitem(sys.modules, "ultralytics", _make_mock_ultralytics(names))

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")

    maps = [c["map50_95"] for c in report.per_class]
    assert maps == sorted(maps, reverse=True)


def test_generate_report_per_class_includes_map50(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")

    monkeypatch.setitem(sys.modules, "ultralytics", _make_mock_ultralytics(names))

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")

    # Every per-class entry must have map50 populated (not always 0.0)
    for cls_entry in report.per_class:
        assert "map50" in cls_entry
        assert cls_entry["map50"] > 0.0, f"map50 is 0 for {cls_entry['class']}"

    # button is index 0, ap50[0] = 0.85
    button_entry = next(c for c in report.per_class if c["class"] == "button")
    assert button_entry["map50"] == pytest.approx(0.85)


def test_report_to_markdown_contains_key_sections(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")

    monkeypatch.setitem(sys.modules, "ultralytics", _make_mock_ultralytics(names))

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")
    md = report.to_markdown()

    assert "mAP@50" in md
    assert "mAP@50-95" in md
    assert "button" in md
    assert "Precision" in md
    assert "Recall" in md
    assert "Dataset" in md
    assert "train" in md.lower()


def test_report_to_json_is_valid_and_complete(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")

    monkeypatch.setitem(sys.modules, "ultralytics", _make_mock_ultralytics(names))

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")
    data = json.loads(report.to_json())

    assert data["map50"] == pytest.approx(0.812)
    assert data["map50_95"] == pytest.approx(0.634)
    assert "per_class" in data
    assert "dataset_stats" in data
    assert "model_info" in data


def test_report_saves_files_to_output_dir(tmp_path, monkeypatch):
    names = {i: c for i, c in enumerate([
        "button", "input", "textarea", "checkbox", "radio", "select", "link",
        "card", "navbar", "tabs", "modal", "table", "alert", "accordion",
        "image", "text", "container",
    ])}
    ds = _make_dataset(tmp_path, names)
    weights = tmp_path / "best.pt"
    weights.write_text("fake")
    out = tmp_path / "report_out"

    monkeypatch.setitem(sys.modules, "ultralytics", _make_mock_ultralytics(names))

    report = generate_report(dataset_path=ds, weights_path=weights, split="test")
    report.save(out)

    assert (out / "report.md").exists()
    assert (out / "report.json").exists()
    md_text = (out / "report.md").read_text()
    assert "mAP@50" in md_text
