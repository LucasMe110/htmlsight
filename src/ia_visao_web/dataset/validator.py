import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ia_visao_web.labeler.selectors import TAXONOMY


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    class_counts: dict[str, int]


class DatasetValidator:
    def __init__(self, root: Path, min_train_instances: int = 200) -> None:
        self.root = root
        self.min_train_instances = min_train_instances

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        train_counts: Counter[str] = Counter()
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
                if split == "train":
                    for line in label_lines:
                        class_index = int(line.split()[0])
                        train_counts[TAXONOMY[class_index]] += 1

        for class_name in TAXONOMY:
            if train_counts[class_name] < self.min_train_instances:
                errors.append(
                    f"classe {class_name} tem {train_counts[class_name]} instancias no train; "
                    f"minimo {self.min_train_instances}"
                )

        return ValidationResult(
            ok=not errors,
            errors=errors,
            class_counts=dict(train_counts),
        )
