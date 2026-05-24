"""End-to-end pipeline test: build -> validate -> train (dry-run) -> predict.

Runs without Playwright, torch, or ultralytics.
"""
import json
from pathlib import Path

from typer.testing import CliRunner

from ia_visao_web.cli import app

_RUNNER = CliRunner()
_COUNT = 5


def test_full_pipeline_build_then_validate_then_train_dryrun_then_predict(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    plan_path = tmp_path / "training-plan.json"

    # Step 1: build synthetic dataset
    result = _RUNNER.invoke(
        app,
        [
            "dataset", "build",
            "--synthetic-only",
            "--count", str(_COUNT),
            "--output", str(dataset_dir),
        ],
    )
    assert result.exit_code == 0, f"build failed: {result.output}"
    assert (dataset_dir / "data.yaml").exists()
    png_files = list((dataset_dir / "images").rglob("*.png"))
    assert len(png_files) == _COUNT

    # Step 2: validate dataset (relax minimum to allow small test dataset)
    result = _RUNNER.invoke(
        app,
        [
            "dataset", "validate",
            "--root", str(dataset_dir),
            "--min-train-instances", "0",
        ],
    )
    assert result.exit_code == 0, f"validate failed: {result.output}"

    # Step 3: train --dry-run produces a training plan JSON
    result = _RUNNER.invoke(
        app,
        [
            "train",
            "--dataset", str(dataset_dir),
            "--output", str(tmp_path / "runs"),
            "--epochs", "10",
            "--plan-output", str(plan_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"train --dry-run failed: {result.output}"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["epochs"] == 10

    # Step 4: predict on a generated image (no weights -> empty detections)
    sample_image = png_files[0]
    result = _RUNNER.invoke(app, ["predict", str(sample_image)])
    assert result.exit_code == 0, f"predict failed: {result.output}"
    payload = json.loads(result.stdout)
    assert "image" in payload
    assert payload["detections"] == []
