import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _resolve_device(requested: str) -> tuple[str, bool]:
    """Resolve o device final e se AMP deve ser desativado.

    Retorna (device_str, amp_disabled).
    Para DirectML devolve ("privateuseone", True).
    """
    if requested not in ("auto", "dml"):
        return requested, False

    try:
        import torch_directml  # type: ignore[import-untyped]
        import torch
        if not torch.cuda.is_available():
            return "privateuseone", True
    except ImportError:
        pass

    return requested, False


def _patch_directml() -> None:
    """Aplica patches no Ultralytics para compatibilidade com DirectML."""
    try:
        import sys
        import torch
        import torch_directml  # type: ignore[import-untyped]
        import ultralytics.utils.torch_utils as torch_utils
        import ultralytics.utils.loss as loss_mod
        import ultralytics.utils.tal as tal_mod
        import ultralytics.engine.trainer as trainer_mod
    except (ImportError, ModuleNotFoundError):
        return

    _dml_device = torch_directml.device(0)

    _orig_select = torch_utils.select_device

    def _patched_select(device="", newline=False, verbose=True):
        if str(device).startswith("privateuseone") or device == "dml":
            if verbose:
                print(f"DirectML GPU: {torch_directml.device_name(0)}")
            return _dml_device
        return _orig_select(device, newline=newline, verbose=verbose)

    # Patch o local canônico e também toda referência direta já importada
    torch_utils.select_device = _patched_select
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        _name = getattr(_mod, "__name__", "") or ""
        if _name.startswith("ultralytics") and hasattr(_mod, "select_device"):
            _mod.select_device = _patched_select

    _orig_preprocess = loss_mod.v8DetectionLoss.preprocess

    def _patched_preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        # DirectML tem suporte limitado — operadores como unique(return_counts) caem pro CPU
        dml_dev = self.device
        self.device = torch.device("cpu")
        try:
            result = _orig_preprocess(self, targets.cpu(), batch_size, scale_tensor.cpu())
        finally:
            self.device = dml_dev
        return result.to(dml_dev)

    loss_mod.v8DetectionLoss.preprocess = _patched_preprocess

    if hasattr(trainer_mod, "check_amp"):
        _orig_check_amp = trainer_mod.check_amp

        def _patched_check_amp(model):
            if str(next(model.parameters()).device).startswith("privateuseone"):
                return False
            return _orig_check_amp(model)

        trainer_mod.check_amp = _patched_check_amp

    # DirectML não suporta scatter_add_ com dimensões parciais — assigner roda no CPU
    _orig_tal_forward = tal_mod.TaskAlignedAssigner._forward

    def _patched_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if str(pd_scores.device).startswith("privateuseone"):
            dev = pd_scores.device
            result = _orig_tal_forward(
                self,
                pd_scores.cpu(), pd_bboxes.cpu(), anc_points.cpu(),
                gt_labels.cpu(), gt_bboxes.cpu(), mask_gt.cpu(),
            )
            return tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in result)
        return _orig_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    tal_mod.TaskAlignedAssigner._forward = _patched_tal_forward

    # DirectML não tem API CUDA de memória — retorna 0 para não crashar no _clear_memory
    _orig_get_memory = trainer_mod.BaseTrainer._get_memory

    def _patched_get_memory(self, fraction=True):
        if str(getattr(self, "device", "")).startswith("privateuseone"):
            return 0
        return _orig_get_memory(self, fraction=fraction)

    trainer_mod.BaseTrainer._get_memory = _patched_get_memory

    # Validação em DirectML falha em múltiplos operadores (BatchNorm, NMS warmup).
    # Pulamos completamente — validar depois com: cli eval --dataset ... --weights last.pt
    _orig_validate = trainer_mod.BaseTrainer.validate

    def _patched_validate(self):
        if str(getattr(self, "device", "")).startswith("privateuseone"):
            return {}, 0.0
        return _orig_validate(self)

    trainer_mod.BaseTrainer.validate = _patched_validate

    # final_eval também chama o validator — desabilitar para DirectML
    _orig_final_eval = trainer_mod.BaseTrainer.final_eval

    def _patched_final_eval(self):
        if str(getattr(self, "device", "")).startswith("privateuseone"):
            return
        return _orig_final_eval(self)

    trainer_mod.BaseTrainer.final_eval = _patched_final_eval


class TrainingUnavailableError(RuntimeError):
    """Raised while the real Ultralytics training backend is not available."""


class UltralyticsUnavailableError(RuntimeError):
    """Raised when ultralytics is not installed."""


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
        payload["dataset"] = self.dataset.as_posix()
        payload["output"] = self.output.as_posix()
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def write_training_plan(config: TrainingConfig, path: Path | None = None) -> Path:
    output_path = path if path is not None else config.output / "training-plan.json"
    config.save(output_path)
    return output_path


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


def train_model(config: TrainingConfig) -> None:
    write_training_plan(config)

    ultralytics = _load_ultralytics()

    device, amp_disabled = _resolve_device(config.device)
    if device == "privateuseone":
        _patch_directml()

    data_yaml = config.dataset / "data.yaml"
    model = ultralytics.YOLO(f"{config.model_size}.pt")
    model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        optimizer=config.optimizer,
        lr0=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        seed=config.seed,
        device=device,
        workers=config.workers,
        mosaic=config.mosaic,
        fliplr=config.flip_lr,
        patience=config.patience,
        project=str(config.output.parent),
        name=config.output.name,
        amp=not amp_disabled,
    )
