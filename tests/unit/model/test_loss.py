import pytest

from ia_visao_web.model.loss import MultiTaskLoss, TorchUnavailableError


def test_multitask_loss_requires_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr("ia_visao_web.model.loss.torch", None)

    with pytest.raises(TorchUnavailableError):
        MultiTaskLoss()
