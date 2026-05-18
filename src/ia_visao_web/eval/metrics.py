from collections.abc import Mapping
from typing import Any


def attribute_accuracy(
    predicted: list[Mapping[str, Any]],
    expected: list[Mapping[str, Any]],
    key: str,
) -> float:
    if len(predicted) != len(expected):
        raise ValueError("predicted e expected precisam ter o mesmo tamanho")
    if not expected:
        return 0.0
    matches = sum(
        1 for pred, exp in zip(predicted, expected, strict=True) if pred.get(key) == exp.get(key)
    )
    return matches / len(expected)
