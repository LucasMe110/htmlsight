import pytest

from ia_visao_web.model.heads import MultiTaskHead, TorchUnavailableError


def test_multitask_head_requires_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr("ia_visao_web.model.heads.torch", None)

    with pytest.raises(TorchUnavailableError):
        MultiTaskHead(num_classes=17, tag_classes=4, display_classes=7, role_classes=8)
