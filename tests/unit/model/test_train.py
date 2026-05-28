import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ia_visao_web.model.train import (
    EvaluationConfig,
    LossWeights,
    TrainingConfig,
    UltralyticsUnavailableError,
    train_model,
    write_training_plan,
)


def test_training_config_serializes_paths_and_evaluation_knobs():
    config = TrainingConfig(
        dataset=Path("data/dataset"),
        output=Path("runs/exp"),
        epochs=12,
        evaluation=EvaluationConfig(split="test", conf_threshold=0.4),
        loss_weights=LossWeights(tag=0.3),
    )

    payload = config.to_dict()

    assert payload["dataset"] == "data/dataset"
    assert payload["output"] == "runs/exp"
    assert payload["epochs"] == 12
    assert payload["evaluation"]["split"] == "test"
    assert payload["evaluation"]["conf_threshold"] == 0.4
    assert payload["loss_weights"]["tag"] == 0.3


def test_write_training_plan_creates_json_file(tmp_path):
    config = TrainingConfig(dataset=tmp_path / "dataset", output=tmp_path / "runs")

    output_path = write_training_plan(config)
    payload = json.loads(output_path.read_text())

    assert output_path == tmp_path / "runs/training-plan.json"
    assert payload["evaluation"]["eval_every"] == 1
    assert payload["loss_weights"]["box"] == 7.5


def _make_mock_ultralytics() -> types.ModuleType:
    ultralytics = types.ModuleType("ultralytics")
    mock_model = MagicMock()
    ultralytics.YOLO = MagicMock(return_value=mock_model)
    return ultralytics


def test_train_model_calls_ultralytics_yolo_with_correct_args(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    data_yaml = dataset_path / "data.yaml"
    data_yaml.write_text("nc: 1\n")

    config = TrainingConfig(
        dataset=dataset_path,
        output=tmp_path / "runs",
        model_size="yolov8s",
        epochs=50,
        batch_size=8,
        image_size=640,
        optimizer="SGD",
        learning_rate=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=42,
        device="cpu",
        workers=2,
        mosaic=0.5,
        flip_lr=0.2,
        patience=10,
    )

    mock_ultralytics = _make_mock_ultralytics()
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ultralytics)

    train_model(config)

    mock_ultralytics.YOLO.assert_called_once_with("yolov8s.pt")
    mock_model = mock_ultralytics.YOLO.return_value
    mock_model.train.assert_called_once()
    call_kwargs = mock_model.train.call_args.kwargs

    assert call_kwargs["data"] == str(data_yaml)
    assert call_kwargs["epochs"] == 50
    assert call_kwargs["batch"] == 8
    assert call_kwargs["imgsz"] == 640
    assert call_kwargs["optimizer"] == "SGD"
    assert call_kwargs["lr0"] == 0.01
    assert call_kwargs["momentum"] == 0.937
    assert call_kwargs["weight_decay"] == 0.0005
    assert call_kwargs["seed"] == 42
    assert call_kwargs["device"] == "cpu"
    assert call_kwargs["workers"] == 2
    assert call_kwargs["mosaic"] == 0.5
    assert call_kwargs["fliplr"] == 0.2
    assert call_kwargs["patience"] == 10
    assert call_kwargs["project"] == str(tmp_path)
    assert call_kwargs["name"] == "runs"


def test_train_model_raises_when_ultralytics_missing(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "data.yaml").write_text("nc: 1\n")

    config = TrainingConfig(dataset=dataset_path, output=tmp_path / "runs")

    monkeypatch.setitem(sys.modules, "ultralytics", None)  # type: ignore[arg-type]

    with pytest.raises(UltralyticsUnavailableError, match="pip install ultralytics"):
        train_model(config)


def test_train_model_writes_plan_then_trains(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "data.yaml").write_text("nc: 1\n")

    config = TrainingConfig(dataset=dataset_path, output=tmp_path / "runs")

    mock_ultralytics = _make_mock_ultralytics()
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ultralytics)

    train_model(config)

    assert (tmp_path / "runs/training-plan.json").exists()
