from typer.testing import CliRunner

from ia_visao_web.cli import app


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
