from ia_visao_web.model.vocab import AttributeVocab


def test_vocab_maps_unknown_values_to_other():
    vocab = AttributeVocab.from_observations(tags=["button"], displays=["block"], roles=[None])

    assert vocab.encode_tag("missing") == vocab.tag_to_id["other"]
    assert vocab.encode_role(None) == vocab.role_to_id["none"]
    assert vocab.encode_display("flex") == vocab.display_to_id["flex"]
