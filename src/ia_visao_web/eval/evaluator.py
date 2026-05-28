import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ia_visao_web.eval.predict import UltralyticsUnavailableError, predict_image

_ATTR_KEYS = ("tag", "display", "role", "has_children")


@dataclass
class EvaluationReport:
    map50: float
    map50_95: float
    per_class: list[dict[str, Any]]
    attr_accuracy: dict[str, float]


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


def evaluate_model(
    dataset_path: Path,
    weights_path: Path,
    split: str = "test",
) -> EvaluationReport:
    ultralytics = _load_ultralytics()

    data_yaml = dataset_path / "data.yaml"
    model = ultralytics.YOLO(str(weights_path))
    val_results = model.val(data=str(data_yaml), split=split)

    map50: float = float(val_results.box.map50)
    map50_95: float = float(val_results.box.map)

    names: dict[int, str] = val_results.names
    ap50_per_class: list[float] = (
        list(val_results.box.ap50)
        if hasattr(val_results.box, "ap50")
        else []
    )
    per_class: list[dict[str, Any]] = [
        {
            "class": names.get(class_id, str(class_id)),
            "class_id": class_id,
            "map50_95": float(m),
            "map50": float(ap50_per_class[class_id]) if class_id < len(ap50_per_class) else 0.0,
        }
        for class_id, m in enumerate(val_results.box.maps)
    ]

    attr_acc = _compute_attr_accuracy(dataset_path, weights_path, split)

    return EvaluationReport(
        map50=map50,
        map50_95=map50_95,
        per_class=per_class,
        attr_accuracy=attr_acc,
    )


def _compute_attr_accuracy(dataset_path: Path, weights_path: Path, split: str) -> dict[str, float]:
    attrs_dir = dataset_path / "attrs" / split
    images_dir = dataset_path / "images" / split

    totals: dict[str, int] = {k: 0 for k in _ATTR_KEYS}
    matches: dict[str, int] = {k: 0 for k in _ATTR_KEYS}

    for sidecar in sorted(attrs_dir.glob("*.json")):
        gt_attrs: list[dict[str, Any]] = json.loads(sidecar.read_text())
        image_path = images_dir / f"{sidecar.stem}.png"

        if not image_path.exists():
            continue

        try:
            detections = predict_image(image_path, weights_path)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            import sys as _sys
            print(f"Warning: predict falhou para {image_path}: {exc}", file=_sys.stderr)
            continue

        pred_attrs = [d["attrs"] for d in detections]
        n = min(len(gt_attrs), len(pred_attrs))

        for i in range(n):
            for key in _ATTR_KEYS:
                totals[key] += 1
                if pred_attrs[i].get(key) == gt_attrs[i].get(key):
                    matches[key] += 1

    return {
        key: (matches[key] / totals[key] if totals[key] > 0 else 0.0)
        for key in _ATTR_KEYS
    }
