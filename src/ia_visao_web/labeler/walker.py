from dataclasses import dataclass
from typing import Any

from ia_visao_web.labeler.geometry import BBox, intersects_viewport, iou
from ia_visao_web.labeler.selectors import class_id


@dataclass(frozen=True)
class RawDomMatch:
    class_name: str
    x: float
    y: float
    width: float
    height: float
    tag: str
    display: str
    role: str | None
    has_children: bool
    n_descendants: int
    visible: bool


@dataclass(frozen=True)
class LabeledDetection:
    class_name: str
    class_id: int
    bbox: BBox
    attrs: dict[str, Any]


def filter_matches(
    matches: list[RawDomMatch],
    viewport_width: int,
    viewport_height: int,
    min_area: float = 16,
    dedupe_iou: float = 0.95,
) -> list[LabeledDetection]:
    kept: list[LabeledDetection] = []
    for match in matches:
        box = BBox(match.x, match.y, match.width, match.height)
        if box.area < min_area:
            continue
        if not match.visible or match.display == "none":
            continue
        if not intersects_viewport(box, viewport_width, viewport_height):
            continue
        if any(
            existing.class_name == match.class_name and iou(existing.bbox, box) > dedupe_iou
            for existing in kept
        ):
            continue
        kept.append(
            LabeledDetection(
                class_name=match.class_name,
                class_id=class_id(match.class_name),
                bbox=box,
                attrs={
                    "tag": match.tag,
                    "display": match.display,
                    "role": match.role,
                    "has_children": match.has_children,
                    "n_descendants": min(match.n_descendants, 50),
                },
            )
        )
    return kept
