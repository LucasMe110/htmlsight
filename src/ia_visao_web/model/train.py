import json
from dataclasses import asdict, dataclass
from pathlib import Path


class TrainingUnavailableError(RuntimeError):
    """Raised while the real Ultralytics training backend is not implemented."""


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "val"
    eval_every: int = 1
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    max_detections: int = 300
    save_predictions: bool = True
    save_plots: bool = True
    failure_examples: int = 50


@dataclass(frozen=True)
class LossWeights:
    cls: float = 0.5
    box: float = 7.5
    tag: float = 0.2
    display: float = 0.2
    role: float = 0.2
    has_children: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    dataset: Path
    output: Path
    model_size: str = "yolov8s"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 20
    seed: int = 0
    device: str = "auto"
    workers: int = 4
    save_period: int = 10
    mosaic: float = 1.0
    mixup: float = 0.0
    hsv: float = 1.0
    flip_lr: float = 0.0
    evaluation: EvaluationConfig = EvaluationConfig()
    loss_weights: LossWeights = LossWeights()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dataset"] = str(self.dataset)
        payload["output"] = str(self.output)
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def write_training_plan(config: TrainingConfig, path: Path | None = None) -> Path:
    output_path = path if path is not None else config.output / "training-plan.json"
    config.save(output_path)
    return output_path


def train_model(config: TrainingConfig) -> None:
    write_training_plan(config)
    raise TrainingUnavailableError(
        "treino real ainda nao foi implementado; use --dry-run para gerar o plano "
        "de hiperparametros e avaliacao sem executar o backend YOLO."
    )
