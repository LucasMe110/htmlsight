from typing import Any

from ia_visao_web.labeler.geometry import BBox


def detection_to_json(
    class_name: str,
    score: float,
    bbox: BBox,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "class": class_name,
        "score": score,
        "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
        "attrs": attrs,
    }
