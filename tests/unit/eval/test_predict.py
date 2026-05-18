from ia_visao_web.eval.predict import detection_to_json
from ia_visao_web.labeler.geometry import BBox


def test_detection_to_json_matches_cli_schema():
    payload = detection_to_json("button", 0.92, BBox(1, 2, 3, 4), {"tag": "button"})

    assert payload == {
        "class": "button",
        "score": 0.92,
        "bbox": [1, 2, 3, 4],
        "attrs": {"tag": "button"},
    }
