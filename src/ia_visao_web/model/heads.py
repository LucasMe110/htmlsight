try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - tested by monkeypatching module global
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class TorchUnavailableError(RuntimeError):
    """Raised when torch-backed model code is used without torch installed."""


class MultiTaskHead:
    def __init__(
        self,
        num_classes: int,
        tag_classes: int,
        display_classes: int,
        role_classes: int,
        in_channels: int = 256,
    ) -> None:
        if torch is None or nn is None:
            raise TorchUnavailableError("torch não está instalado; instale o extra `model`.")
        self.module = nn.ModuleDict(
            {
                "cls": nn.Conv2d(in_channels, num_classes, kernel_size=1),
                "attr_tag": nn.Conv2d(in_channels, tag_classes, kernel_size=1),
                "attr_display": nn.Conv2d(in_channels, display_classes, kernel_size=1),
                "attr_role": nn.Conv2d(in_channels, role_classes, kernel_size=1),
                "attr_haschld": nn.Conv2d(in_channels, 1, kernel_size=1),
            }
        )
