import sys
import types
from unittest.mock import MagicMock

import pytest

from ia_visao_web.eval.predict import detection_to_json, predict_image
from ia_visao_web.labeler.geometry import BBox


def test_detection_to_json_matches_cli_schema():
    payload = detection_to_json("button", 0.92, BBox(1, 2, 3, 4), {"tag": "button"})

    assert payload == {
        "class": "button",
        "score": 0.92,
        "bbox": [1, 2, 3, 4],
        "attrs": {"tag": "button"},
    }


def _make_mock_ultralytics(
    xyxy: list[list[float]],
    conf: list[float],
    cls: list[float],
    names: dict[int, str],
) -> types.ModuleType:
    ultralytics = types.ModuleType("ultralytics")

    mock_boxes = MagicMock()
    mock_boxes.xyxy.tolist.return_value = xyxy
    mock_boxes.conf.tolist.return_value = conf
    mock_boxes.cls.tolist.return_value = cls

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = names

    mock_model = MagicMock(return_value=[mock_result])
    ultralytics.YOLO = MagicMock(return_value=mock_model)
    return ultralytics


def test_predict_image_returns_detections_with_correct_schema(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake-png")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"fake-weights")

    mock_ul = _make_mock_ultralytics(
        xyxy=[[10.0, 20.0, 110.0, 80.0], [200.0, 100.0, 400.0, 300.0]],
        conf=[0.9, 0.75],
        cls=[0.0, 1.0],
        names={0: "button", 1: "card"},
    )
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)

    detections = predict_image(image_path, weights_path)

    assert len(detections) == 2

    d0 = detections[0]
    assert d0["class"] == "button"
    assert abs(d0["score"] - 0.9) < 1e-6
    assert d0["bbox"] == [10.0, 20.0, 100.0, 60.0]  # x, y, w=110-10, h=80-20
    assert set(d0["attrs"].keys()) == {"tag", "display", "role", "has_children"}
    assert d0["attrs"]["tag"] is None

    d1 = detections[1]
    assert d1["class"] == "card"
    assert abs(d1["score"] - 0.75) < 1e-6
    assert d1["bbox"] == [200.0, 100.0, 200.0, 200.0]  # x, y, w=400-200, h=300-100


def test_predict_image_calls_yolo_with_weights_path(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake-png")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"fake-weights")

    mock_ul = _make_mock_ultralytics(
        xyxy=[], conf=[], cls=[], names={}
    )
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ul)

    predict_image(image_path, weights_path)

    mock_ul.YOLO.assert_called_once_with(str(weights_path))
    mock_model = mock_ul.YOLO.return_value
    mock_model.assert_called_once_with(str(image_path))


def test_predict_image_raises_when_ultralytics_missing(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake-png")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"fake-weights")

    monkeypatch.setitem(sys.modules, "ultralytics", None)  # type: ignore[arg-type]

    from ia_visao_web.eval.predict import UltralyticsUnavailableError

    with pytest.raises(UltralyticsUnavailableError, match="pip install ultralytics"):
        predict_image(image_path, weights_path)


def test_predict_image_returns_empty_when_boxes_is_none(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake-png")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"fake-weights")

    ultralytics = types.ModuleType("ultralytics")
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model = MagicMock(return_value=[mock_result])
    ultralytics.YOLO = MagicMock(return_value=mock_model)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    detections = predict_image(image_path, weights_path)

    assert detections == []
