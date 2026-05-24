import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def _make_synthetic_dataset(root: Path, split: str = "train", n: int = 2) -> list[str]:
    for dirname in ("images", "labels", "attrs"):
        (root / dirname / split).mkdir(parents=True, exist_ok=True)

    ids = []
    for i in range(n):
        sample_id = f"sample-{i:05d}"
        img = Image.new("RGB", (640, 640), color=(i * 30, i * 60, i * 90))
        img.save(root / "images" / split / f"{sample_id}.png")

        label_lines = [
            f"0 {0.5:.6f} {0.4:.6f} {0.2:.6f} {0.3:.6f}",
            f"1 {0.6:.6f} {0.7:.6f} {0.1:.6f} {0.1:.6f}",
        ]
        (root / "labels" / split / f"{sample_id}.txt").write_text(
            "\n".join(label_lines) + "\n"
        )

        attrs = [
            {"tag": "button", "display": "inline-block", "role": "button", "has_children": False},
            {"tag": "div", "display": "flex", "role": None, "has_children": True},
        ]
        (root / "attrs" / split / f"{sample_id}.json").write_text(json.dumps(attrs))
        ids.append(sample_id)
    return ids


def _inject_torch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _Dataset:
        pass

    data_mod.Dataset = _Dataset  # type: ignore[attr-defined]
    torch_mod.utils = utils_mod  # type: ignore[attr-defined]
    torch_mod.utils.data = data_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_mod)
    monkeypatch.delitem(sys.modules, "ia_visao_web.model.dataset_loader", raising=False)
    return torch_mod


def test_dataset_len_returns_number_of_images(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    assert len(ds) == 2


def test_dataset_getitem_returns_required_keys(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    item = ds[0]

    assert set(item.keys()) >= {"image", "boxes", "attrs", "image_id"}


def test_dataset_getitem_image_is_pil(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    item = ds[0]

    assert isinstance(item["image"], Image.Image)


def test_dataset_getitem_boxes_are_yolo_dicts(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    item = ds[0]

    boxes = item["boxes"]
    assert isinstance(boxes, list)
    assert len(boxes) == 2
    assert all(set(b.keys()) >= {"class_id", "cx", "cy", "w", "h"} for b in boxes)
    assert boxes[0]["class_id"] == 0
    assert abs(boxes[0]["cx"] - 0.5) < 1e-5


def test_dataset_getitem_attrs_align_with_boxes(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    item = ds[0]

    attrs = item["attrs"]
    assert isinstance(attrs, list)
    assert len(attrs) == len(item["boxes"])
    assert attrs[0]["tag"] == "button"
    assert attrs[1]["tag"] == "div"


def test_dataset_getitem_image_id_is_stem(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    ids = _make_synthetic_dataset(tmp_path, split="train", n=2)

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    retrieved_ids = {ds[i]["image_id"] for i in range(len(ds))}
    assert retrieved_ids == set(ids)


def test_dataset_missing_sidecar_raises_error(tmp_path, monkeypatch) -> None:
    _inject_torch(monkeypatch)
    _make_synthetic_dataset(tmp_path, split="train", n=1)

    sidecar = tmp_path / "attrs" / "train" / "sample-00000.json"
    sidecar.unlink()

    from ia_visao_web.model.dataset_loader import WebComponentDataset

    ds = WebComponentDataset(tmp_path, "train")
    with pytest.raises(FileNotFoundError, match="sample-00000"):
        _ = ds[0]


def test_dataset_raises_torch_unavailable_when_torch_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "torch.utils", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "torch.utils.data", None)  # type: ignore[arg-type]
    monkeypatch.delitem(sys.modules, "ia_visao_web.model.dataset_loader", raising=False)

    import importlib

    import ia_visao_web.model.dataset_loader as _mod

    importlib.reload(_mod)

    from ia_visao_web.model.dataset_loader import TorchUnavailableError, WebComponentDataset

    with pytest.raises(TorchUnavailableError, match="torch"):
        WebComponentDataset(tmp_path, "train")
