from ia_visao_web.labeler.geometry import BBox, dedupe_by_iou, normalize_bbox


def test_normalize_bbox_converts_pixels_to_yolo_center_format():
    assert normalize_bbox(BBox(10, 20, 30, 40), image_width=100, image_height=200) == (
        0.25,
        0.2,
        0.3,
        0.2,
    )


def test_dedupe_by_iou_keeps_first_high_overlap_box():
    boxes = [BBox(0, 0, 100, 100), BBox(1, 1, 100, 100), BBox(200, 200, 10, 10)]

    assert dedupe_by_iou(boxes, threshold=0.95) == [boxes[0], boxes[2]]
