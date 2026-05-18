from ia_visao_web.dataset.splits import split_for_id


def test_split_for_id_is_deterministic():
    assert split_for_id("abc") == split_for_id("abc")
    assert split_for_id("abc") in {"train", "val", "test"}
