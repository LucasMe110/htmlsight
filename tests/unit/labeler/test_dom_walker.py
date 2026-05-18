from dataclasses import dataclass
from typing import Any

import pytest

from ia_visao_web.labeler.dom_walker import DomWalker, raw_match_from_payload, selector_payload
from ia_visao_web.labeler.selectors import SELECTORS


class FakePage:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.payload = payload
        self.expression = ""
        self.arg: object | None = None

    def evaluate(self, expression: str, arg: object | None = None) -> object:
        self.expression = expression
        self.arg = arg
        return self.payload


@dataclass(frozen=True)
class FakeRule:
    class_name: str
    selector: str


def test_selector_payload_preserves_selector_order():
    payload = selector_payload()

    assert payload[0] == {
        "class_name": SELECTORS[0].class_name,
        "selector": SELECTORS[0].selector,
    }
    assert [item["class_name"] for item in payload] == [rule.class_name for rule in SELECTORS]


def test_dom_walker_converts_browser_payload_to_raw_matches():
    page = FakePage(
        [
            {
                "class_name": "button",
                "x": 10,
                "y": 20,
                "width": 100,
                "height": 40,
                "tag": "button",
                "display": "inline-block",
                "role": None,
                "has_children": False,
                "n_descendants": 0,
                "visible": True,
            }
        ]
    )

    matches = DomWalker(rules=[FakeRule("button", "button.btn")]).collect(page)

    assert matches[0].class_name == "button"
    assert matches[0].tag == "button"
    assert matches[0].display == "inline-block"
    assert matches[0].role is None
    assert page.arg == [{"class_name": "button", "selector": "button.btn"}]


def test_raw_match_from_payload_rejects_non_mapping_payload():
    with pytest.raises(TypeError, match="match DOM"):
        raw_match_from_payload(["button"])
