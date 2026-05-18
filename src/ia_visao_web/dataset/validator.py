import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from ia_visao_web.labeler.selectors import TAXONOMY


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    class_counts: dict[str, int]
    split_class_counts: dict[str, dict[str, int]]


class DatasetValidator:
    def __init__(
        self,
        root: Path,
        min_train_instances: int = 200,
        require_split_coverage: bool = False,
    ) -> None:
        self.root = root
        self.min_train_instances = min_train_instances
        self.require_split_coverage = require_split_coverage

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        split_counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }
        for split in ("train", "val", "test"):
            labels_dir = self.root / "labels" / split
            attrs_dir = self.root / "attrs" / split
            if not labels_dir.exists():
                continue
            for label_path in sorted(labels_dir.glob("*.txt")):
                attr_path = attrs_dir / f"{label_path.stem}.json"
                label_lines = [line for line in label_path.read_text().splitlines() if line.strip()]
                if not attr_path.exists():
                    errors.append(f"sidecar ausente para {label_path}")
                    continue
                attrs = json.loads(attr_path.read_text())
                if len(attrs) != len(label_lines):
                    errors.append(
                        f"sidecar desalinhado para {label_path}: "
                        f"{len(label_lines)} labels vs {len(attrs)} attrs"
                    )
                for line_number, line in enumerate(label_lines, start=1):
                    class_index = _validate_yolo_line(line, label_path, line_number, errors)
                    if class_index is not None:
                        split_counts[split][TAXONOMY[class_index]] += 1

        for class_name in TAXONOMY:
            if split_counts["train"][class_name] < self.min_train_instances:
                count = split_counts["train"][class_name]
                errors.append(
                    f"classe {class_name} tem {count} instancias no train; "
                    f"minimo {self.min_train_instances}"
                )
            if self.require_split_coverage:
                for split in ("val", "test"):
                    if split_counts[split][class_name] == 0:
                        errors.append(f"classe {class_name} ausente no split {split}")

        return ValidationResult(
            ok=not errors,
            errors=errors,
            class_counts=dict(split_counts["train"]),
            split_class_counts={
                split: dict(counts)
                for split, counts in split_counts.items()
            },
        )

    def write_qa_overlays(self, sample_count: int) -> list[Path]:
        if sample_count <= 0:
            return []

        output_dir = self.root / "_qa"
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for split in ("train", "val", "test"):
            images_dir = self.root / "images" / split
            labels_dir = self.root / "labels" / split
            if not images_dir.exists():
                continue
            for image_path in sorted(images_dir.glob("*.png")):
                if len(written) >= sample_count:
                    return written
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                output_path = output_dir / f"{split}-{image_path.name}"
                _write_overlay(image_path, label_path, output_path)
                written.append(output_path)

        return written

    def write_report(self) -> Path:
        result = self.validate()
        report_path = self.root / "_qa" / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": result.ok,
            "errors": result.errors,
            "class_counts": result.split_class_counts,
            "area_buckets": self._area_buckets(),
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return report_path

    def _area_buckets(self) -> dict[str, dict[str, int]]:
        buckets: dict[str, Counter[str]] = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }
        for split in ("train", "val", "test"):
            labels_dir = self.root / "labels" / split
            if not labels_dir.exists():
                continue
            for label_path in sorted(labels_dir.glob("*.txt")):
                for line in label_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    _, _, _, width, height = _parse_yolo_line(line)
                    buckets[split][_area_bucket(width * height)] += 1
        return {split: dict(counts) for split, counts in buckets.items()}


def _validate_yolo_line(
    line: str,
    label_path: Path,
    line_number: int,
    errors: list[str],
) -> int | None:
    parts = line.split()
    if len(parts) != 5:
        errors.append(f"label invalida em {label_path}:{line_number}: esperado 5 campos")
        return None

    try:
        class_index = int(parts[0])
    except ValueError:
        errors.append(f"classe invalida em {label_path}:{line_number}: {parts[0]}")
        return None

    if class_index < 0 or class_index >= len(TAXONOMY):
        errors.append(f"classe fora da taxonomia em {label_path}:{line_number}: {class_index}")
        return None

    try:
        cx, cy, width, height = (float(value) for value in parts[1:])
    except ValueError:
        errors.append(f"bbox invalida em {label_path}:{line_number}: valores nao numericos")
        return None

    if not all(0 <= value <= 1 for value in (cx, cy, width, height)):
        errors.append(f"bbox fora do intervalo 0-1 em {label_path}:{line_number}")
        return None

    if width <= 0 or height <= 0:
        errors.append(f"bbox com largura/altura zerada em {label_path}:{line_number}")
        return None

    return class_index


def _write_overlay(image_path: Path, label_path: Path, output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_index, cx, cy, width, height = _parse_yolo_line(line)
            left = (cx - width / 2) * image.width
            top = (cy - height / 2) * image.height
            right = (cx + width / 2) * image.width
            bottom = (cy + height / 2) * image.height
            label = TAXONOMY[class_index]
            draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=2)
            if top >= 14:
                label_top = top - 14
                label_bottom = top
            else:
                label_top = top
                label_bottom = top + 14
            label_box = (left, label_top, left + len(label) * 7 + 6, label_bottom)
            draw.rectangle(label_box, fill=(255, 0, 0))
            draw.text((left + 3, label_top + 1), label, fill=(255, 255, 255))
        image.save(output_path)


def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    class_id_text, cx_text, cy_text, width_text, height_text = line.split()
    return (
        int(class_id_text),
        float(cx_text),
        float(cy_text),
        float(width_text),
        float(height_text),
    )


def _area_bucket(area: float) -> str:
    if area < 0.005:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"
