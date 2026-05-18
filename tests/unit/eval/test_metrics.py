from ia_visao_web.eval.metrics import attribute_accuracy


def test_attribute_accuracy_counts_matching_values():
    result = attribute_accuracy(
        predicted=[{"tag": "button"}, {"tag": "div"}],
        expected=[{"tag": "button"}, {"tag": "span"}],
        key="tag",
    )

    assert result == 0.5
