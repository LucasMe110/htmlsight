from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


def iou(first: BBox, second: BBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def dedupe_by_iou(boxes: list[BBox], threshold: float = 0.95) -> list[BBox]:
    kept: list[BBox] = []
    for box in boxes:
        if all(iou(box, existing) <= threshold for existing in kept):
            kept.append(box)
    return kept


def intersects_viewport(box: BBox, viewport_width: float, viewport_height: float) -> bool:
    return box.right > 0 and box.bottom > 0 and box.x < viewport_width and box.y < viewport_height


def normalize_bbox(
    box: BBox,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    cx = (box.x + box.width / 2) / image_width
    cy = (box.y + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height
    return (round(cx, 6), round(cy, 6), round(width, 6), round(height, 6))
