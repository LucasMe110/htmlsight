from pathlib import Path
from urllib.request import urlopen

BOOTSTRAP_DOC_PAGES: dict[str, str] = {
    "accordions": "https://getbootstrap.com/docs/5.3/components/accordion/",
    "alerts": "https://getbootstrap.com/docs/5.3/components/alerts/",
    "buttons": "https://getbootstrap.com/docs/5.3/components/buttons/",
    "cards": "https://getbootstrap.com/docs/5.3/components/card/",
    "forms": "https://getbootstrap.com/docs/5.3/forms/overview/",
    "modals": "https://getbootstrap.com/docs/5.3/components/modal/",
    "navbars": "https://getbootstrap.com/docs/5.3/components/navbar/",
    "tables": "https://getbootstrap.com/docs/5.3/content/tables/",
    "tabs": "https://getbootstrap.com/docs/5.3/components/navs-tabs/",
}


def fetch_docs(output: Path, force: bool = False) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url in BOOTSTRAP_DOC_PAGES.items():
        dest = output / f"{name}.html"
        if dest.exists() and not force:
            written.append(dest)
            continue
        with urlopen(url) as response:  # noqa: S310
            html = response.read().decode("utf-8")
        dest.write_text(html, encoding="utf-8")
        written.append(dest)
    return written
