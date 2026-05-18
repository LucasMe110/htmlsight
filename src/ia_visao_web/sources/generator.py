from dataclasses import dataclass
from random import Random

from faker import Faker
from jinja2 import Template


@dataclass(frozen=True)
class GeneratedPage:
    page_id: str
    html: str
    viewport: tuple[int, int]


class BootstrapPageGenerator:
    VIEWPORTS = ((1280, 720), (1920, 1080), (768, 1024), (375, 667))
    COLORS = ("primary", "success", "warning", "danger", "info")

    def __init__(self, seed: int, locale: str = "pt_BR") -> None:
        self._random = Random(seed)
        self._faker = Faker(locale)
        self._faker.seed_instance(seed)

    def generate_page(self, page_id: str) -> GeneratedPage:
        viewport = self._random.choice(self.VIEWPORTS)
        color = self._random.choice(self.COLORS)
        card_count = self._random.randint(2, 4)
        cards = [
            {
                "title": self._faker.sentence(nb_words=3).rstrip("."),
                "body": self._faker.text(max_nb_chars=80),
            }
            for _ in range(card_count)
        ]
        html = _TEMPLATE.render(page_id=page_id, color=color, cards=cards)
        return GeneratedPage(page_id=page_id, html=html, viewport=viewport)


_TEMPLATE = Template(
    """<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet"
    >
    <title>{{ page_id }}</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg bg-{{ color }}" role="navigation">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">IA Visao</a>
        <button class="btn btn-light" type="button">Entrar</button>
      </div>
    </nav>
    <main class="container py-4">
      <div class="alert alert-{{ color }}" role="alert">Dataset sintético Bootstrap</div>
      <section class="row g-3">
        {% for card in cards %}
        <article class="col">
          <div class="card">
            <div class="card-body">
              <h2 class="card-title h5">{{ card.title }}</h2>
              <p class="card-text">{{ card.body }}</p>
              <a href="#" class="btn btn-{{ color }}" role="button">Abrir</a>
            </div>
          </div>
        </article>
        {% endfor %}
      </section>
      <form class="mt-4">
        <label class="form-label" for="email">Email</label>
        <input class="form-control" id="email" type="email" placeholder="nome@example.com">
        <select class="form-select mt-2" aria-label="Plano">
          <option>Basico</option>
          <option>Pro</option>
        </select>
        <button class="btn btn-primary mt-3" type="submit">Salvar</button>
      </form>
    </main>
  </body>
</html>
"""
)
