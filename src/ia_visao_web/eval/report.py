import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ia_visao_web.eval.predict import UltralyticsUnavailableError


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


@dataclass
class ModelReport:
    map50: float
    map50_95: float
    precision: float
    recall: float
    per_class: list[dict[str, Any]]
    dataset_stats: dict[str, Any]
    model_info: dict[str, Any]
    split: str
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines: list[str] = []

        lines.append("# htmlsight — Relatório de Avaliação do Modelo")
        lines.append(f"\n**Gerado em:** {self.generated_at}  ")
        lines.append(f"**Split avaliado:** `{self.split}`  ")
        lines.append(f"**Arquitetura:** {self.model_info.get('architecture', 'YOLOv8s')}  ")
        if self.model_info.get("params"):
            params_m = self.model_info["params"] / 1_000_000
            lines.append(f"**Parâmetros:** {params_m:.1f}M  ")

        lines.append("\n---\n")
        lines.append("## Métricas Gerais\n")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| mAP@50 | **{self.map50:.3f}** |")
        lines.append(f"| mAP@50-95 | **{self.map50_95:.3f}** |")
        lines.append(f"| Precision | {self.precision:.3f} |")
        lines.append(f"| Recall | {self.recall:.3f} |")

        lines.append("\n---\n")
        lines.append("## Métricas por Classe (ordenado por mAP@50-95 ↓)\n")
        lines.append("| Classe | mAP@50-95 | mAP@50 |")
        lines.append("|--------|-----------|--------|")
        for c in self.per_class:
            bar = _bar(c["map50_95"])
            map50_val = c.get("map50", 0.0)
            lines.append(f"| `{c['class']}` | {c['map50_95']:.3f} {bar} | {map50_val:.3f} |")

        lines.append("\n---\n")
        lines.append("## Dataset\n")
        ds = self.dataset_stats
        total = ds.get("train_images", 0) + ds.get("val_images", 0) + ds.get("test_images", 0)
        lines.append(f"- **Total de imagens:** {total:,}")
        lines.append(f"- **Train:** {ds.get('train_images', 0):,} imagens")
        lines.append(f"- **Val:** {ds.get('val_images', 0):,} imagens")
        lines.append(f"- **Test:** {ds.get('test_images', 0):,} imagens")
        lines.append(f"- **Classes:** {ds.get('num_classes', 17)}")
        if ds.get("total_train_instances"):
            lines.append(f"- **Instâncias no train:** {ds['total_train_instances']:,}")

        lines.append("\n---\n")
        lines.append("## Como Reproduzir\n")
        lines.append("```bash")
        lines.append("# Gerar dataset")
        lines.append("PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright \\")
        lines.append("  PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \\")
        lines.append("  --count 3000 --workers 4 --output data/dataset")
        lines.append("")
        lines.append("# Treinar")
        lines.append("PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \\")
        lines.append("  --dataset data/dataset --output runs/baseline --epochs 100")
        lines.append("")
        lines.append("# Gerar este relatório")
        lines.append("PYTHONPATH=src venv/bin/python -m ia_visao_web.cli report \\")
        lines.append("  --dataset data/dataset --weights runs/baseline/weights/best.pt \\")
        lines.append("  --output runs/baseline/report")
        lines.append("```")

        lines.append("\n---\n")
        lines.append("*Gerado com [htmlsight](https://github.com/LucasMe110/htmlsight)*")

        return "\n".join(lines)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        (output_dir / "report.json").write_text(self.to_json(), encoding="utf-8")


def generate_report(
    dataset_path: Path,
    weights_path: Path,
    split: str = "test",
) -> ModelReport:
    ultralytics = _load_ultralytics()

    data_yaml = dataset_path / "data.yaml"
    model = ultralytics.YOLO(str(weights_path))
    val_results = model.val(data=str(data_yaml), split=split)

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    names: dict[int, str] = val_results.names
    maps_per_class: list[float] = list(val_results.box.maps)
    ap50_per_class: list[float] = (
        list(val_results.box.ap50)
        if hasattr(val_results.box, "ap50")
        else []
    )

    per_class_raw = [
        {
            "class": names.get(i, str(i)),
            "class_id": i,
            "map50_95": float(m),
            "map50": float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0,
        }
        for i, m in enumerate(maps_per_class)
    ]
    per_class = sorted(per_class_raw, key=lambda x: x["map50_95"], reverse=True)  # type: ignore[arg-type,return-value]

    dataset_stats = _collect_dataset_stats(dataset_path)

    try:
        params, _ = model.info(verbose=False)
        model_info: dict[str, Any] = {"architecture": "YOLOv8s", "params": int(params)}
    except Exception:
        model_info = {"architecture": "YOLOv8s"}

    return ModelReport(
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        per_class=per_class,
        dataset_stats=dataset_stats,
        model_info=model_info,
        split=split,
    )


def _collect_dataset_stats(dataset_path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_path / "images" / split
        count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        stats[f"{split}_images"] = count

    labels_dir = dataset_path / "labels" / "train"
    total_instances = 0
    if labels_dir.exists():
        for lf in labels_dir.glob("*.txt"):
            lines = [ln for ln in lf.read_text().splitlines() if ln.strip()]
            total_instances += len(lines)
    stats["total_train_instances"] = total_instances

    import yaml  # pyyaml bundled as dependency

    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        stats["num_classes"] = yaml_data.get("nc", 0)
        stats["class_names"] = yaml_data.get("names", [])

    return stats


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)
