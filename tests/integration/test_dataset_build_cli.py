from typer.testing import CliRunner

from ia_visao_web.cli import app


def test_dataset_build_synthetic_fixture_writes_expected_files(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "build",
            "--output",
            str(tmp_path),
            "--count",
            "2",
            "--synthetic-only",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "data.yaml").exists()
    assert len(list((tmp_path / "images").glob("*/*.png"))) == 2
