from ia_visao_web.labeler.selectors import SELECTORS, TAXONOMY, class_id


def test_taxonomy_matches_spec_order():
    assert TAXONOMY == [
        "button",
        "input",
        "textarea",
        "checkbox",
        "radio",
        "select",
        "link",
        "card",
        "navbar",
        "tabs",
        "modal",
        "table",
        "alert",
        "accordion",
        "image",
        "text",
        "container",
    ]


def test_button_selector_has_priority_over_link():
    classes = [rule.class_name for rule in SELECTORS]

    assert classes.index("button") < classes.index("link")
    assert class_id("button") == 0
