from pathlib import Path
from unittest.mock import MagicMock, patch

from ia_visao_web.sources.fetch_bootstrap_docs import BOOTSTRAP_DOC_PAGES, fetch_docs


def _make_urlopen_mock(html: str = "<html>bootstrap docs</html>") -> MagicMock:
    mock_response = MagicMock()
    mock_response.read.return_value = html.encode("utf-8")
    mock_response.__enter__ = lambda self: self
    mock_response.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_response)


def test_bootstrap_doc_pages_has_required_components() -> None:
    required = {
        "buttons", "cards", "forms", "modals", "navbars",
        "alerts", "tables", "accordions", "tabs",
    }
    assert required.issubset(BOOTSTRAP_DOC_PAGES.keys())


def test_fetch_docs_creates_html_files_for_all_components(tmp_path: Path) -> None:
    mock_urlopen = _make_urlopen_mock()
    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        written = fetch_docs(tmp_path)

    assert len(written) == len(BOOTSTRAP_DOC_PAGES)
    for name in BOOTSTRAP_DOC_PAGES:
        assert (tmp_path / f"{name}.html").exists()


def test_fetch_docs_writes_correct_html_content(tmp_path: Path) -> None:
    mock_urlopen = _make_urlopen_mock("<html>hello bootstrap</html>")
    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        fetch_docs(tmp_path)

    for name in BOOTSTRAP_DOC_PAGES:
        content = (tmp_path / f"{name}.html").read_text(encoding="utf-8")
        assert content == "<html>hello bootstrap</html>"


def test_fetch_docs_skips_existing_files(tmp_path: Path) -> None:
    (tmp_path / "buttons.html").write_text("<html>cached</html>", encoding="utf-8")
    mock_urlopen = _make_urlopen_mock("<html>new</html>")

    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen) as m:
        fetch_docs(tmp_path)
        calls_made = m.call_count

    assert (tmp_path / "buttons.html").read_text(encoding="utf-8") == "<html>cached</html>"
    assert calls_made == len(BOOTSTRAP_DOC_PAGES) - 1


def test_fetch_docs_force_redownloads_existing_files(tmp_path: Path) -> None:
    (tmp_path / "buttons.html").write_text("<html>cached</html>", encoding="utf-8")
    mock_urlopen = _make_urlopen_mock("<html>new</html>")

    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        fetch_docs(tmp_path, force=True)

    assert (tmp_path / "buttons.html").read_text(encoding="utf-8") == "<html>new</html>"


def test_fetch_docs_returns_list_of_paths(tmp_path: Path) -> None:
    mock_urlopen = _make_urlopen_mock()
    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        written = fetch_docs(tmp_path)

    assert all(isinstance(p, Path) for p in written)
    assert len(written) == len(BOOTSTRAP_DOC_PAGES)


def test_fetch_docs_skipped_files_included_in_return(tmp_path: Path) -> None:
    (tmp_path / "buttons.html").write_text("<html>cached</html>", encoding="utf-8")
    mock_urlopen = _make_urlopen_mock()

    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        written = fetch_docs(tmp_path)

    assert len(written) == len(BOOTSTRAP_DOC_PAGES)
    assert tmp_path / "buttons.html" in written


def test_fetch_docs_creates_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "docs" / "bootstrap"
    mock_urlopen = _make_urlopen_mock()

    with patch("ia_visao_web.sources.fetch_bootstrap_docs.urlopen", mock_urlopen):
        fetch_docs(output)

    assert output.exists()
