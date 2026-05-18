import json

from PIL import Image

from ia_visao_web.dataset.writer import DatasetWriter
from ia_visao_web.labeler.geometry import BBox
from ia_visao_web.labeler.walker import LabeledDetection


def test_writer_keeps_yolo_labels_and_attrs_aligned(tmp_path):
    image = Image.new("RGB", (100, 200), "white")
    detections = [
        LabeledDetection(
            "button",
            0,
            BBox(10, 20, 30, 40),
            {
                "tag": "button",
                "display": "inline-block",
                "role": None,
                "has_children": False,
                "n_descendants": 0,
            },
        )
    ]

    writer = DatasetWriter(tmp_path)
    writer.write_sample("abc", image, detections, split="train")

    label_text = (tmp_path / "labels/train/abc.txt").read_text()
    attrs = json.loads((tmp_path / "attrs/train/abc.json").read_text())

    assert label_text.strip() == "0 0.250000 0.200000 0.300000 0.200000"
    assert attrs[0]["tag"] == "button"
    assert (tmp_path / "images/train/abc.png").exists()
    assert (tmp_path / "data.yaml").exists()
