import pytest

from ia_visao_web.model.loss import AttributeLoss, MultiTaskLoss, TorchUnavailableError


def test_multitask_loss_requires_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr("ia_visao_web.model.loss.torch", None)

    with pytest.raises(TorchUnavailableError):
        MultiTaskLoss()


def test_attribute_loss_requires_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr("ia_visao_web.model.loss.torch", None)

    with pytest.raises(TorchUnavailableError):
        AttributeLoss(tag_classes=5, display_classes=7, role_classes=8)


def test_attribute_loss_positive_with_random_predictions():
    torch = pytest.importorskip("torch")
    from ia_visao_web.model.loss import AttributeLoss
    from ia_visao_web.model.train import LossWeights

    loss_fn = AttributeLoss(tag_classes=5, display_classes=7, role_classes=8)
    N = 4
    preds = {
        "tag": torch.randn(N, 5),
        "display": torch.randn(N, 7),
        "role": torch.randn(N, 8),
        "has_children": torch.randn(N, 1),
    }
    targets = {
        "tag": torch.randint(0, 5, (N,)),
        "display": torch.randint(0, 7, (N,)),
        "role": torch.randint(0, 8, (N,)),
        "has_children": torch.randint(0, 2, (N, 1)).float(),
    }
    mask = torch.ones(N, dtype=torch.bool)

    loss = loss_fn(preds, targets, LossWeights(), mask)
    assert loss.item() > 0


def test_attribute_loss_gradient_flows_through_all_heads():
    torch = pytest.importorskip("torch")
    from ia_visao_web.model.loss import AttributeLoss
    from ia_visao_web.model.train import LossWeights

    loss_fn = AttributeLoss(tag_classes=5, display_classes=7, role_classes=8)
    N = 4
    preds = {
        "tag": torch.randn(N, 5, requires_grad=True),
        "display": torch.randn(N, 7, requires_grad=True),
        "role": torch.randn(N, 8, requires_grad=True),
        "has_children": torch.randn(N, 1, requires_grad=True),
    }
    targets = {
        "tag": torch.randint(0, 5, (N,)),
        "display": torch.randint(0, 7, (N,)),
        "role": torch.randint(0, 8, (N,)),
        "has_children": torch.randint(0, 2, (N, 1)).float(),
    }
    mask = torch.ones(N, dtype=torch.bool)

    loss = loss_fn(preds, targets, LossWeights(), mask)
    loss.backward()

    for key in ("tag", "display", "role", "has_children"):
        assert preds[key].grad is not None, f"no gradient for head: {key}"


def test_attribute_loss_zero_when_predictions_are_perfect():
    torch = pytest.importorskip("torch")
    from ia_visao_web.model.loss import AttributeLoss
    from ia_visao_web.model.train import LossWeights

    loss_fn = AttributeLoss(tag_classes=5, display_classes=7, role_classes=8)
    N = 4
    t_tag = torch.randint(0, 5, (N,))
    t_display = torch.randint(0, 7, (N,))
    t_role = torch.randint(0, 8, (N,))
    t_hc = torch.randint(0, 2, (N, 1)).float()

    # Perfect logits: very high value for correct class, very low for others
    p_tag = torch.full((N, 5), -1e9)
    p_tag[range(N), t_tag] = 1e9
    p_display = torch.full((N, 7), -1e9)
    p_display[range(N), t_display] = 1e9
    p_role = torch.full((N, 8), -1e9)
    p_role[range(N), t_role] = 1e9
    # BCE: large positive for has_children=1, large negative for 0
    p_hc = torch.where(t_hc > 0.5, torch.tensor(1e9), torch.tensor(-1e9))

    preds = {"tag": p_tag, "display": p_display, "role": p_role, "has_children": p_hc}
    targets = {"tag": t_tag, "display": t_display, "role": t_role, "has_children": t_hc}
    mask = torch.ones(N, dtype=torch.bool)

    loss = loss_fn(preds, targets, LossWeights(), mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)
