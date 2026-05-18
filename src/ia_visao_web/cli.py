import json
from io import BytesIO
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw

from ia_visao_web.dataset.splits import split_for_id
from ia_visao_web.dataset.validator import DatasetValidator
from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.dom_walker import DomWalker
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection, filter_matches
from ia_visao_web.model.train import (
    EvaluationConfig,
    LossWeights,
    TrainingConfig,
    TrainingUnavailableError,
    train_model,
    write_training_plan,
)
from ia_visao_web.renderer.playwright_renderer import PlaywrightRenderer, RendererUnavailableError
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
    writer = DatasetWriter(output)
    renderer = None if synthetic_only else _playwright_renderer()
    walker = DomWalker()
    for index in range(count):
        sample_id = f"synthetic-{index:05d}"
        page = BootstrapPageGenerator(seed=index).generate_page(page_id=sample_id)
        if synthetic_only:
            image = _synthetic_image(page.viewport)
            detections = _fixture_detections()
        else:
            if renderer is None:  # pragma: no cover - defensive narrowing
                raise typer.BadParameter("renderer indisponivel")
            image, detections = _render_and_label(renderer, walker, page.html, page.viewport)
        split = split_for_id(sample_id)
        writer.write_sample(sample_id, image, detections, split=split)
    typer.echo(f"dataset escrito em {output}")


@dataset_app.command("validate")
def dataset_validate(
    root: Annotated[Path, typer.Option("--root", "-r")] = Path("data/dataset"),
    min_train_instances: Annotated[int, typer.Option("--min-train-instances", min=0)] = 200,
    require_split_coverage: Annotated[bool, typer.Option("--require-split-coverage")] = False,
    qa_samples: Annotated[int, typer.Option("--qa-samples", min=0)] = 0,
    report: Annotated[bool, typer.Option("--report")] = False,
) -> None:
    """Valida um dataset gerado."""
    validator = DatasetValidator(
        root,
        min_train_instances=min_train_instances,
        require_split_coverage=require_split_coverage,
    )
    result = validator.validate()
    if qa_samples:
        written = validator.write_qa_overlays(qa_samples)
        typer.echo(f"qa escrito em {root / '_qa'} ({len(written)} imagens)")
    if report:
        report_path = validator.write_report()
        typer.echo(f"relatorio escrito em {report_path}")
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("dataset ok")


@app.command()
def train(
    dataset: Annotated[Path, typer.Option("--dataset", "-d")] = Path("data/dataset"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("runs/train"),
    model_size: Annotated[str, typer.Option("--model-size")] = "yolov8s",
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 100,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    image_size: Annotated[int, typer.Option("--image-size", min=32)] = 640,
    optimizer: Annotated[str, typer.Option("--optimizer")] = "SGD",
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0)] = 0.01,
    momentum: Annotated[float, typer.Option("--momentum", min=0.0)] = 0.937,
    weight_decay: Annotated[float, typer.Option("--weight-decay", min=0.0)] = 0.0005,
    patience: Annotated[int, typer.Option("--patience", min=0)] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    device: Annotated[str, typer.Option("--device")] = "auto",
    workers: Annotated[int, typer.Option("--workers", min=0)] = 4,
    save_period: Annotated[int, typer.Option("--save-period", min=1)] = 10,
    eval_split: Annotated[str, typer.Option("--eval-split")] = "val",
    eval_every: Annotated[int, typer.Option("--eval-every", min=1)] = 1,
    conf_threshold: Annotated[float, typer.Option("--conf-threshold", min=0.0, max=1.0)] = 0.25,
    iou_threshold: Annotated[float, typer.Option("--iou-threshold", min=0.0, max=1.0)] = 0.50,
    max_detections: Annotated[int, typer.Option("--max-detections", min=1)] = 300,
    failure_examples: Annotated[int, typer.Option("--failure-examples", min=0)] = 50,
    save_predictions: Annotated[
        bool,
        typer.Option("--save-predictions/--no-save-predictions"),
    ] = True,
    save_plots: Annotated[bool, typer.Option("--save-plots/--no-save-plots")] = True,
    mosaic: Annotated[float, typer.Option("--mosaic", min=0.0)] = 1.0,
    mixup: Annotated[float, typer.Option("--mixup", min=0.0)] = 0.0,
    hsv: Annotated[float, typer.Option("--hsv", min=0.0)] = 1.0,
    flip_lr: Annotated[float, typer.Option("--flip-lr", min=0.0, max=1.0)] = 0.0,
    lambda_cls: Annotated[float, typer.Option("--lambda-cls", min=0.0)] = 0.5,
    lambda_box: Annotated[float, typer.Option("--lambda-box", min=0.0)] = 7.5,
    lambda_tag: Annotated[float, typer.Option("--lambda-tag", min=0.0)] = 0.2,
    lambda_display: Annotated[float, typer.Option("--lambda-display", min=0.0)] = 0.2,
    lambda_role: Annotated[float, typer.Option("--lambda-role", min=0.0)] = 0.2,
    lambda_has_children: Annotated[
        float,
        typer.Option("--lambda-has-children", min=0.0),
    ] = 0.1,
    plan_output: Annotated[Path | None, typer.Option("--plan-output")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Treina o modelo multi-task."""
    config = TrainingConfig(
        dataset=dataset,
        output=output,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        device=device,
        workers=workers,
        save_period=save_period,
        mosaic=mosaic,
        mixup=mixup,
        hsv=hsv,
        flip_lr=flip_lr,
        evaluation=EvaluationConfig(
            split=eval_split,
            eval_every=eval_every,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            save_predictions=save_predictions,
            save_plots=save_plots,
            failure_examples=failure_examples,
        ),
        loss_weights=LossWeights(
            cls=lambda_cls,
            box=lambda_box,
            tag=lambda_tag,
            display=lambda_display,
            role=lambda_role,
            has_children=lambda_has_children,
        ),
    )
    if dry_run:
        output_path = write_training_plan(config, plan_output)
        typer.echo(f"plano de treino escrito em {output_path}")
        return

    try:
        train_model(config)
    except TrainingUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc


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


def _playwright_renderer() -> PlaywrightRenderer:
    try:
        return PlaywrightRenderer()
    except RendererUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _render_and_label(
    renderer: PlaywrightRenderer,
    walker: DomWalker,
    html: str,
    viewport: tuple[int, int],
) -> tuple[Image.Image, list[LabeledDetection]]:
    try:
        with renderer.open_page(html, viewport) as page:
            raw_matches = walker.collect(page)
            png = page.screenshot(full_page=True)
    except RendererUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc

    image = Image.open(BytesIO(png)).convert("RGB")
    detections = filter_matches(
        raw_matches,
        viewport_width=image.width,
        viewport_height=image.height,
    )
    return image, detections


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
