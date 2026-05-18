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
