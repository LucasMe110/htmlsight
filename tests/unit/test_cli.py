import json

from typer.testing import CliRunner

from ia_visao_web.cli import app
from ia_visao_web.eval.predict import UltralyticsUnavailableError


def test_cli_shows_top_level_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "dataset" in result.stdout
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "predict" in result.stdout


def test_predict_without_model_returns_valid_empty_payload(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-image")

    result = CliRunner().invoke(app, ["predict", str(image)])

    assert result.exit_code == 0
    assert '"image"' in result.stdout
    assert '"detections": []' in result.stdout


def test_dataset_build_real_reports_missing_playwright(monkeypatch, tmp_path):
    monkeypatch.setattr("ia_visao_web.renderer.playwright_renderer.sync_playwright", None)

    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "build",
            "--output",
            str(tmp_path),
            "--count",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "playwright" in result.output


def test_predict_with_missing_weights_warns_and_returns_empty(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-image")
    missing_weights = tmp_path / "nonexistent.pt"

    result = CliRunner().invoke(app, ["predict", str(image), "--weights", str(missing_weights)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["detections"] == []


def test_predict_with_weights_calls_predict_image(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-image")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    fake_detections = [
        {
            "class": "button",
            "score": 0.9,
            "bbox": [0, 0, 100, 50],
            "attrs": {"tag": None, "display": None, "role": None, "has_children": None},
        }
    ]
    monkeypatch.setattr("ia_visao_web.cli.predict_image", lambda img, wts: fake_detections)

    result = CliRunner().invoke(app, ["predict", str(image), "--weights", str(weights)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["class"] == "button"


def test_predict_with_weights_handles_ultralytics_unavailable(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-image")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    def _raise(_img: object, _wts: object) -> object:
        raise UltralyticsUnavailableError("pip install ultralytics")

    monkeypatch.setattr("ia_visao_web.cli.predict_image", _raise)

    result = CliRunner().invoke(app, ["predict", str(image), "--weights", str(weights)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["detections"] == []


def test_train_dry_run_writes_evaluation_plan(tmp_path):
    plan_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset",
            str(tmp_path / "dataset"),
            "--output",
            str(tmp_path / "runs"),
            "--epochs",
            "5",
            "--eval-split",
            "test",
            "--eval-every",
            "2",
            "--conf-threshold",
            "0.35",
            "--iou-threshold",
            "0.6",
            "--lambda-tag",
            "0.4",
            "--plan-output",
            str(plan_path),
            "--dry-run",
        ],
    )

    payload = json.loads(plan_path.read_text())

    assert result.exit_code == 0, result.output
    assert payload["epochs"] == 5
    assert payload["evaluation"]["split"] == "test"
    assert payload["evaluation"]["eval_every"] == 2
    assert payload["evaluation"]["conf_threshold"] == 0.35
    assert payload["evaluation"]["iou_threshold"] == 0.6
    assert payload["loss_weights"]["tag"] == 0.4
