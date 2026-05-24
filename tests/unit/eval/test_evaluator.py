import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_val_results(
    map50: float = 0.85,
    map50_95: float = 0.65,
    maps: list[float] | None = None,
    names: dict[int, str] | None = None,
) -> MagicMock:
    if maps is None:
        maps = [0.65]
    if names is None:
        names = {0: "button"}

    mock_box = MagicMock()
    mock_box.map50 = map50
    mock_box.map = map50_95
    mock_box.maps = maps

    mock_results = MagicMock()
    mock_results.box = mock_box
    mock_results.names = names

    return mock_results


def _make_mock_ultralytics(val_results: MagicMock) -> types.ModuleType:
    ultralytics = types.ModuleType("ultralytics")
    mock_model = MagicMock()
    mock_model.val.return_value = val_results
    ultralytics.YOLO = MagicMock(return_value=mock_model)
    return ultralytics


def _write_fake_dataset(tmp_path: Path, split: str = "test") -> Path:
    dataset = tmp_path / "dataset"
    attrs_dir = dataset / "attrs" / split
    images_dir = dataset / "images" / split
    attrs_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    sidecar = attrs_dir / "sample-00000.json"
    sidecar.write_text(
        json.dumps(
            [{"tag": "button", "display": "inline-block", "role": None, "has_children": False}]
        )
    )
    (images_dir / "sample-00000.png").write_bytes(b"fake-png")

    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [button]\n"
    )

    return dataset


def test_evaluate_model_returns_correct_report_shape(tmp_path: Path, monkeypatch: Any) -> None:
    dataset = _write_fake_dataset(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    val_results = _make_val_results(map50=0.85, map50_95=0.65, maps=[0.65], names={0: "button"})
    mock_ul = _make_mock_ultralytics(val_results)
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)
    monkeypatch.setattr("ia_visao_web.eval.evaluator.predict_image", lambda img, wts: [])

    from ia_visao_web.eval.evaluator import evaluate_model

    report = evaluate_model(dataset, weights, split="test")

    assert abs(report.map50 - 0.85) < 1e-6
    assert abs(report.map50_95 - 0.65) < 1e-6
    assert len(report.per_class) == 1
    assert report.per_class[0]["class"] == "button"
    assert "map50_95" in report.per_class[0]
    assert isinstance(report.attr_accuracy, dict)
    for key in ("tag", "display", "role", "has_children"):
        assert key in report.attr_accuracy


def test_evaluate_model_raises_when_ultralytics_missing(tmp_path: Path, monkeypatch: Any) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    monkeypatch.setitem(sys.modules, "ultralytics", None)  # type: ignore[arg-type]

    from ia_visao_web.eval.evaluator import evaluate_model
    from ia_visao_web.eval.predict import UltralyticsUnavailableError

    with pytest.raises(UltralyticsUnavailableError):
        evaluate_model(dataset, weights)


def test_evaluate_model_calls_val_with_correct_params(tmp_path: Path, monkeypatch: Any) -> None:
    dataset = _write_fake_dataset(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    val_results = _make_val_results()
    mock_ul = _make_mock_ultralytics(val_results)
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)
    monkeypatch.setattr("ia_visao_web.eval.evaluator.predict_image", lambda img, wts: [])

    from ia_visao_web.eval.evaluator import evaluate_model

    evaluate_model(dataset, weights, split="val")

    data_yaml = str(dataset / "data.yaml")
    mock_ul.YOLO.assert_called_once_with(str(weights))
    mock_model = mock_ul.YOLO.return_value
    mock_model.val.assert_called_once_with(data=data_yaml, split="val")


def test_evaluate_model_attr_accuracy_computed_from_sidecars(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = _write_fake_dataset(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    val_results = _make_val_results()
    mock_ul = _make_mock_ultralytics(val_results)
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)

    def fake_predict(img: Path, wts: Path) -> list[dict[str, Any]]:
        return [
            {
                "class": "button",
                "score": 0.9,
                "bbox": [0, 0, 100, 50],
                "attrs": {
                    "tag": "button",
                    "display": "block",
                    "role": None,
                    "has_children": False,
                },
            }
        ]

    monkeypatch.setattr("ia_visao_web.eval.evaluator.predict_image", fake_predict)

    from ia_visao_web.eval.evaluator import evaluate_model

    report = evaluate_model(dataset, weights, split="test")

    assert report.attr_accuracy["tag"] == 1.0
    assert report.attr_accuracy["display"] == 0.0  # "block" != "inline-block"
    assert report.attr_accuracy["role"] == 1.0
    assert report.attr_accuracy["has_children"] == 1.0


def test_evaluate_model_attr_accuracy_zero_when_no_sidecars(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "attrs" / "test").mkdir(parents=True)
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "data.yaml").write_text("path: .\nnc: 0\nnames: []\n")

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    val_results = _make_val_results(maps=[], names={})
    mock_ul = _make_mock_ultralytics(val_results)
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)

    from ia_visao_web.eval.evaluator import evaluate_model

    report = evaluate_model(dataset, weights, split="test")

    for key in ("tag", "display", "role", "has_children"):
        assert report.attr_accuracy[key] == 0.0
