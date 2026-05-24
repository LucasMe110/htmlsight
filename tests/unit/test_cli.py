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


def test_eval_without_weights_exits_with_code_1(tmp_path):
    missing = tmp_path / "nonexistent.pt"

    result = CliRunner().invoke(
        app,
        ["eval", "--dataset", str(tmp_path / "dataset"), "--weights", str(missing)],
    )

    assert result.exit_code == 1


def test_eval_prints_json_report(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    from ia_visao_web.eval.evaluator import EvaluationReport

    fake_report = EvaluationReport(
        map50=0.85,
        map50_95=0.65,
        per_class=[{"class": "button", "class_id": 0, "map50_95": 0.65}],
        attr_accuracy={"tag": 0.9, "display": 0.8, "role": 0.7, "has_children": 0.95},
    )
    monkeypatch.setattr(
        "ia_visao_web.cli.evaluate_model", lambda d, w, split: fake_report
    )

    result = CliRunner().invoke(
        app,
        ["eval", "--dataset", str(dataset), "--weights", str(weights)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "mAP50" in payload
    assert "mAP50_95" in payload
    assert "per_class" in payload
    assert "attr_accuracy" in payload
    assert abs(payload["mAP50"] - 0.85) < 1e-6


def test_eval_exits_1_when_ultralytics_unavailable(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    from ia_visao_web.eval.predict import UltralyticsUnavailableError

    def raise_unavailable(d, w, split):
        raise UltralyticsUnavailableError("pip install ultralytics")

    monkeypatch.setattr("ia_visao_web.cli.evaluate_model", raise_unavailable)

    result = CliRunner().invoke(
        app,
        ["eval", "--dataset", str(dataset), "--weights", str(weights)],
    )

    assert result.exit_code == 1


def test_dataset_fetch_docs_creates_files(tmp_path, monkeypatch):
    from ia_visao_web.sources.fetch_bootstrap_docs import BOOTSTRAP_DOC_PAGES

    written = [tmp_path / f"{name}.html" for name in BOOTSTRAP_DOC_PAGES]
    monkeypatch.setattr("ia_visao_web.cli.fetch_docs", lambda out, force=False: written)

    result = CliRunner().invoke(
        app, ["dataset", "fetch-docs", "--output", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert str(len(written)) in result.stdout


def test_dataset_fetch_docs_force_flag(tmp_path, monkeypatch):
    called_with: dict[str, object] = {}

    def fake_fetch(out, force=False):
        called_with["out"] = out
        called_with["force"] = force
        return []

    monkeypatch.setattr("ia_visao_web.cli.fetch_docs", fake_fetch)

    result = CliRunner().invoke(
        app, ["dataset", "fetch-docs", "--output", str(tmp_path), "--force"]
    )

    assert result.exit_code == 0
    assert called_with["force"] is True


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
