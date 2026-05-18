import json
from pathlib import Path

import yaml
from PIL import Image

from ia_visao_web.labeler.geometry import normalize_bbox
from ia_visao_web.labeler.selectors import TAXONOMY
from ia_visao_web.labeler.walker import LabeledDetection


class DatasetWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_sample(
        self,
        sample_id: str,
        image: Image.Image,
        detections: list[LabeledDetection],
        split: str,
    ) -> None:
        self._ensure_split_dirs(split)
        image_path = self.root / "images" / split / f"{sample_id}.png"
        label_path = self.root / "labels" / split / f"{sample_id}.txt"
        attrs_path = self.root / "attrs" / split / f"{sample_id}.json"

        image.save(image_path)
        label_lines = []
        attrs = []
        for detection in detections:
            cx, cy, width, height = normalize_bbox(
                detection.bbox,
                image_width=image.width,
                image_height=image.height,
            )
            label_lines.append(
                f"{detection.class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
            )
            attrs.append(detection.attrs)

        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        attrs_path.write_text(json.dumps(attrs, indent=2, sort_keys=True))
        self.write_data_yaml()

    def write_data_yaml(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(TAXONOMY),
            "names": TAXONOMY,
        }
        (self.root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    def _ensure_split_dirs(self, split: str) -> None:
        for dirname in ("images", "labels", "attrs"):
            (self.root / dirname / split).mkdir(parents=True, exist_ok=True)
