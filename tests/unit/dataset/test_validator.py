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
