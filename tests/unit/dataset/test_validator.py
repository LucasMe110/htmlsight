import json

from PIL import Image

from ia_visao_web.dataset.validator import DatasetValidator
from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection


def test_validator_detects_label_attr_mismatch(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample(
        "one",
        image,
        [
            LabeledDetection(
                "button",
                0,
                BBox(10, 10, 20, 20),
                {
                    "tag": "button",
                    "display": "inline-block",
                    "role": None,
                    "has_children": False,
                    "n_descendants": 0,
                },
            )
        ],
        split="train",
    )
    (tmp_path / "attrs/train/one.json").write_text("[]")

    result = DatasetValidator(tmp_path, min_train_instances=1).validate()

    assert not result.ok
    assert any("sidecar" in error for error in result.errors)


def test_validator_detects_invalid_yolo_label_values(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample(
        "one",
        image,
        [
            LabeledDetection(
                "button",
                0,
                BBox(10, 10, 20, 20),
                {
                    "tag": "button",
                    "display": "inline-block",
                    "role": None,
                    "has_children": False,
                    "n_descendants": 0,
                },
            )
        ],
        split="train",
    )
    (tmp_path / "labels/train/one.txt").write_text("999 1.2 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n")

    result = DatasetValidator(tmp_path, min_train_instances=0).validate()

    assert not result.ok
    assert any("classe fora da taxonomia" in error for error in result.errors)
    assert any("largura/altura zerada" in error for error in result.errors)


def test_validator_can_write_qa_overlays(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample(
        "one",
        image,
        [
            LabeledDetection(
                "button",
                0,
                BBox(10, 10, 20, 20),
                {
                    "tag": "button",
                    "display": "inline-block",
                    "role": None,
                    "has_children": False,
                    "n_descendants": 0,
                },
            )
        ],
        split="train",
    )

    written = DatasetValidator(tmp_path, min_train_instances=0).write_qa_overlays(1)

    assert written == [tmp_path / "_qa/train-one.png"]
    assert written[0].exists()


def test_validator_can_require_val_and_test_coverage(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample(
        "one",
        image,
        [
            LabeledDetection(
                "button",
                0,
                BBox(10, 10, 20, 20),
                {
                    "tag": "button",
                    "display": "inline-block",
                    "role": None,
                    "has_children": False,
                    "n_descendants": 0,
                },
            )
        ],
        split="train",
    )

    result = DatasetValidator(
        tmp_path,
        min_train_instances=0,
        require_split_coverage=True,
    ).validate()

    assert not result.ok
    assert any("ausente no split val" in error for error in result.errors)


def test_validator_can_write_distribution_report(tmp_path):
    writer = DatasetWriter(tmp_path)
    image = Image.new("RGB", (100, 100), "white")
    writer.write_sample(
        "one",
        image,
        [
            LabeledDetection(
                "button",
                0,
                BBox(10, 10, 20, 20),
                {
                    "tag": "button",
                    "display": "inline-block",
                    "role": None,
                    "has_children": False,
                    "n_descendants": 0,
                },
            )
        ],
        split="train",
    )

    report_path = DatasetValidator(tmp_path, min_train_instances=0).write_report()
    report = json.loads(report_path.read_text())

    assert report["class_counts"]["train"]["button"] == 1
    assert report["area_buckets"]["train"]["medium"] == 1
