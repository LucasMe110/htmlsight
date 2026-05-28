import importlib
import sys
from pathlib import Path
from typing import Any

from ia_visao_web.labeler.geometry import BBox


class UltralyticsUnavailableError(RuntimeError):
    """Raised when ultralytics is not installed."""


def detection_to_json(
    class_name: str,
    score: float,
    bbox: BBox,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "class": class_name,
        "score": score,
        "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
        "attrs": attrs,
    }


def _load_ultralytics() -> Any:
    sentinel = object()
    mod: Any = sys.modules.get("ultralytics", sentinel)
    if mod is sentinel:
        try:
            mod = importlib.import_module("ultralytics")
        except ImportError:
            mod = None
    if mod is None:
        raise UltralyticsUnavailableError(
            "ultralytics nao esta instalado. Instale com: pip install ultralytics"
        )
    return mod


def predict_image(image_path: Path, weights_path: Path) -> list[dict[str, Any]]:
    ultralytics = _load_ultralytics()

    model = ultralytics.YOLO(str(weights_path))
    results = model(str(image_path))

    detections: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.tolist()
        conf = boxes.conf.tolist()
        cls = boxes.cls.tolist()
        names: dict[int, str] = result.names

        for coords, score, cls_id in zip(xyxy, conf, cls, strict=True):
            x1, y1, x2, y2 = coords
            bbox = BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
            class_name = names.get(int(cls_id), str(int(cls_id)))
            attrs: dict[str, Any] = {
                "tag": None,
                "display": None,
                "role": None,
                "has_children": None,
            }
            detections.append(detection_to_json(class_name, float(score), bbox, attrs))

    return detections
