from ia_visao_web.sources.generator import BootstrapPageGenerator


def test_generator_is_deterministic_for_seed():
    first = BootstrapPageGenerator(seed=123).generate_page(page_id="sample")
    second = BootstrapPageGenerator(seed=123).generate_page(page_id="sample")

    assert first.html == second.html
    assert first.viewport == second.viewport


def test_generator_includes_core_bootstrap_components():
    page = BootstrapPageGenerator(seed=1).generate_page(page_id="components")

    assert "navbar" in page.html
    assert "card" in page.html
    assert "form-control" in page.html
    assert "alert" in page.html
