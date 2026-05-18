import json
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw

from ia_visao_web.dataset.splits import split_for_id
from ia_visao_web.dataset.validator import DatasetValidator
from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection
from ia_visao_web.sources.generator import BootstrapPageGenerator

app = typer.Typer(help="Detector visual multi-task de componentes web.")
dataset_app = typer.Typer(help="Gera e valida datasets YOLO com atributos HTML.")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("build")
def dataset_build(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("data/dataset"),
    count: Annotated[int, typer.Option("--count", min=1)] = 3000,
    synthetic_only: Annotated[bool, typer.Option("--synthetic-only")] = False,
) -> None:
    """Gera o dataset."""
    if not synthetic_only:
        raise typer.BadParameter("por enquanto use --synthetic-only neste ambiente sem Playwright")

    writer = DatasetWriter(output)
    for index in range(count):
        sample_id = f"synthetic-{index:05d}"
        page = BootstrapPageGenerator(seed=index).generate_page(page_id=sample_id)
        image = _synthetic_image(page.viewport)
        split = split_for_id(sample_id)
        writer.write_sample(sample_id, image, _fixture_detections(), split=split)
    typer.echo(f"dataset escrito em {output}")


@dataset_app.command("validate")
def dataset_validate(
    root: Annotated[Path, typer.Option("--root", "-r")] = Path("data/dataset"),
) -> None:
    """Valida um dataset gerado."""
    result = DatasetValidator(root).validate()
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("dataset ok")


@app.command()
def train() -> None:
    """Treina o modelo multi-task."""
    raise typer.BadParameter("train ainda não foi implementado")


@app.command(name="eval")
def eval_command(
    split: Annotated[str, typer.Option("--split")] = "test",
) -> None:
    """Calcula métricas no split informado."""
    raise typer.BadParameter(f"eval ainda não foi implementado para split={split}")


@app.command()
def predict(image: Path) -> None:
    """Roda inferência em uma imagem e imprime JSON."""
    if not image.exists():
        raise typer.BadParameter(f"imagem não encontrada: {image}")
    typer.echo(json.dumps({"image": str(image), "detections": []}, indent=2))


def main() -> None:
    app()


def _synthetic_image(viewport: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", viewport, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, viewport[0] - 24, 80), fill=(13, 110, 253))
    draw.rectangle((48, 120, 248, 260), outline=(210, 210, 210), width=2)
    draw.rectangle((64, 210, 164, 244), fill=(13, 110, 253))
    draw.rectangle((48, 300, 360, 340), outline=(120, 120, 120), width=2)
    return image


def _fixture_detections() -> list[LabeledDetection]:
    return [
        LabeledDetection(
            "navbar",
            8,
            BBox(24, 24, 600, 56),
            {
                "tag": "nav",
                "display": "flex",
                "role": "navigation",
                "has_children": True,
                "n_descendants": 3,
            },
        ),
        LabeledDetection(
            "card",
            7,
            BBox(48, 120, 200, 140),
            {
                "tag": "div",
                "display": "block",
                "role": None,
                "has_children": True,
                "n_descendants": 4,
            },
        ),
        LabeledDetection(
            "button",
            0,
            BBox(64, 210, 100, 34),
            {
                "tag": "button",
                "display": "inline-block",
                "role": None,
                "has_children": False,
                "n_descendants": 0,
            },
        ),
        LabeledDetection(
            "input",
            1,
            BBox(48, 300, 312, 40),
            {
                "tag": "input",
                "display": "block",
                "role": None,
                "has_children": False,
                "n_descendants": 0,
            },
        ),
    ]


if __name__ == "__main__":
    main()
