from collections.abc import Sequence
from typing import Any, Protocol

from ia_visao_web.labeler.selectors import SELECTORS, SelectorRule
from ia_visao_web.labeler.walker import RawDomMatch


class EvaluatesJavaScript(Protocol):
    def evaluate(self, expression: str, arg: object | None = None) -> Any: ...


DOM_WALKER_SCRIPT = """
(rules) => {
  const matches = [];
  for (const rule of rules) {
    for (const element of document.querySelectorAll(rule.selector)) {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      matches.push({
        class_name: rule.class_name,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        tag: element.tagName.toLowerCase(),
        display: style.display,
        role: element.getAttribute("role"),
        has_children: element.children.length > 0,
        n_descendants: element.querySelectorAll("*").length,
        visible: style.visibility !== "hidden" && style.display !== "none"
      });
    }
  }
  return matches;
}
"""


class DomWalker:
    def __init__(self, rules: Sequence[SelectorRule] | None = None) -> None:
        self._rules = rules

    def collect(self, page: EvaluatesJavaScript) -> list[RawDomMatch]:
        payload = page.evaluate(DOM_WALKER_SCRIPT, selector_payload(self._rules))
        if not isinstance(payload, list):
            raise TypeError("DOM walker retornou payload invalido")
        return [raw_match_from_payload(item) for item in payload]


def selector_payload(rules: Sequence[SelectorRule] | None = None) -> list[dict[str, str]]:
    selector_rules = SELECTORS if rules is None else rules
    return [
        {
            "class_name": rule.class_name,
            "selector": rule.selector,
        }
        for rule in selector_rules
    ]


def raw_match_from_payload(payload: object) -> RawDomMatch:
    if not isinstance(payload, dict):
        raise TypeError("match DOM precisa ser um objeto")
    return RawDomMatch(
        class_name=str(payload["class_name"]),
        x=float(payload["x"]),
        y=float(payload["y"]),
        width=float(payload["width"]),
        height=float(payload["height"]),
        tag=str(payload["tag"]),
        display=str(payload["display"]),
        role=_optional_str(payload.get("role")),
        has_children=bool(payload["has_children"]),
        n_descendants=int(payload["n_descendants"]),
        visible=bool(payload["visible"]),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
