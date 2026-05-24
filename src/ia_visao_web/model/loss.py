from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - tested by monkeypatching module global
    torch = None
    nn = None


class TorchUnavailableError(RuntimeError):
    """Raised when torch-backed loss code is used without torch installed."""


class MultiTaskLoss:
    def __init__(
        self,
        lambda_cls: float = 0.5,
        lambda_box: float = 7.5,
        lambda_tag: float = 0.2,
        lambda_display: float = 0.2,
        lambda_role: float = 0.2,
        lambda_has_children: float = 0.1,
    ) -> None:
        if torch is None or nn is None:
            raise TorchUnavailableError("torch não está instalado; instale o extra `model`.")
        self.weights = {
            "cls": lambda_cls,
            "box": lambda_box,
            "tag": lambda_tag,
            "display": lambda_display,
            "role": lambda_role,
            "has_children": lambda_has_children,
        }
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()


class AttributeLoss:
    """Attribute head losses: CE for tag/display/role, BCE for has_children."""

    def __init__(self, tag_classes: int, display_classes: int, role_classes: int) -> None:
        if torch is None or nn is None:
            raise TorchUnavailableError("torch não está instalado; instale com: pip install torch")
        self.tag_classes = tag_classes
        self.display_classes = display_classes
        self.role_classes = role_classes
        self._ce: Any = nn.CrossEntropyLoss()
        self._bce: Any = nn.BCEWithLogitsLoss()

    def __call__(
        self,
        preds: dict[str, Any],
        targets: dict[str, Any],
        weights: Any,
        mask: Any,
    ) -> Any:
        if not mask.any():
            return torch.tensor(0.0)

        tag_loss = self._ce(preds["tag"][mask], targets["tag"][mask])
        display_loss = self._ce(preds["display"][mask], targets["display"][mask])
        role_loss = self._ce(preds["role"][mask], targets["role"][mask])
        hc_loss = self._bce(preds["has_children"][mask], targets["has_children"][mask])

        return (
            weights.tag * tag_loss
            + weights.display * display_loss
            + weights.role * role_loss
            + weights.has_children * hc_loss
        )
