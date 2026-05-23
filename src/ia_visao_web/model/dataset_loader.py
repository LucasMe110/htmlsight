import json
from pathlib import Path
from typing import Any

try:
    import torch as _torch
except ImportError:  # pragma: no cover - tested by monkeypatching sys.modules
    _torch = None


class TorchUnavailableError(RuntimeError):
    """Raised when torch-backed dataset code is used without torch installed."""


class WebComponentDataset:
    """Dataset loader for YOLO-format multi-task web component detection."""

    def __init__(self, root: Path, split: str) -> None:
        if _torch is None:
            raise TorchUnavailableError(
                "torch não está instalado. Instale com: pip install torch"
            )
        self.root = root
        self.split = split
        self._image_paths = sorted((root / "images" / split).glob("*.png"))

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        image_path = self._image_paths[idx]
        sample_id = image_path.stem

        attrs_path = self.root / "attrs" / self.split / f"{sample_id}.json"
        if not attrs_path.exists():
            raise FileNotFoundError(
                f"Sidecar JSON ausente: {attrs_path} (sample: {sample_id})"
            )

        label_path = self.root / "labels" / self.split / f"{sample_id}.txt"
        boxes: list[dict[str, Any]] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid, cx, cy, w, h = parts
                    boxes.append(
                        {
                            "class_id": int(cid),
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(w),
                            "h": float(h),
                        }
                    )

        attrs: list[dict[str, Any]] = json.loads(attrs_path.read_text())
        image = Image.open(image_path)

        return {
            "image": image,
            "boxes": boxes,
            "attrs": attrs,
            "image_id": sample_id,
        }
