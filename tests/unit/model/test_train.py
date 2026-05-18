import json
from pathlib import Path

import pytest

from ia_visao_web.model.train import (
    EvaluationConfig,
    LossWeights,
    TrainingConfig,
    TrainingUnavailableError,
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


def test_train_model_writes_plan_before_reporting_missing_backend(tmp_path):
    config = TrainingConfig(dataset=tmp_path / "dataset", output=tmp_path / "runs")

    with pytest.raises(TrainingUnavailableError, match="treino real"):
        train_model(config)

    assert (tmp_path / "runs/training-plan.json").exists()
