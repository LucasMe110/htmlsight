from ia_visao_web.labeler.walker import RawDomMatch, filter_matches


def test_filter_matches_drops_hidden_small_and_outside_elements():
    matches = [
        RawDomMatch("button", 0, 0, 20, 20, "button", "inline-block", None, False, 0, True),
        RawDomMatch("button", 0, 0, 2, 2, "button", "inline-block", None, False, 0, True),
        RawDomMatch("button", 10, 10, 20, 20, "button", "none", None, False, 0, True),
        RawDomMatch("button", 500, 500, 20, 20, "button", "inline-block", None, False, 0, True),
        RawDomMatch("button", 10, 10, 20, 20, "button", "inline-block", None, False, 0, False),
    ]

    filtered = filter_matches(matches, viewport_width=100, viewport_height=100)

    assert len(filtered) == 1
    assert filtered[0].class_name == "button"


def test_filter_matches_deduplicates_same_class_high_iou():
    matches = [
        RawDomMatch("card", 0, 0, 100, 100, "div", "block", None, True, 3, True),
        RawDomMatch("card", 1, 1, 100, 100, "div", "block", None, True, 3, True),
    ]

    filtered = filter_matches(matches, viewport_width=200, viewport_height=200)

    assert len(filtered) == 1
