# ruff: noqa: E501
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
        rows = [
            {
                "name": self._faker.first_name(),
                "email": self._faker.email(),
                "status": self._random.choice(("Ativo", "Pendente", "Bloqueado")),
            }
            for _ in range(3)
        ]
        cards = [
            {
                "title": self._faker.sentence(nb_words=3).rstrip("."),
                "body": self._faker.text(max_nb_chars=80),
            }
            for _ in range(card_count)
        ]
        html = _TEMPLATE.render(page_id=page_id, color=color, cards=cards, rows=rows)
        return GeneratedPage(page_id=page_id, html=html, viewport=viewport)


_TEMPLATE = Template(
    """<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ page_id }}</title>
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; font-family: Arial, sans-serif; color: #212529; background: #f8f9fa; }
      a { color: #0d6efd; }
      .navbar { min-height: 58px; padding: 12px 24px; color: #fff; }
      .navbar .container-fluid { display: flex; align-items: center; justify-content: space-between; }
      .navbar-brand { color: #fff; font-weight: 700; text-decoration: none; }
      .bg-primary { background: #0d6efd; }
      .bg-success { background: #198754; }
      .bg-warning { background: #ffc107; color: #212529; }
      .bg-danger { background: #dc3545; }
      .bg-info { background: #0dcaf0; color: #212529; }
      .container { width: min(1120px, calc(100% - 48px)); margin: 0 auto; }
      .container-fluid { width: 100%; }
      .py-4 { padding-top: 24px; padding-bottom: 24px; }
      .mt-2 { margin-top: 8px; }
      .mt-3 { margin-top: 16px; }
      .mt-4 { margin-top: 24px; }
      .row { display: flex; flex-wrap: wrap; gap: 16px; }
      .col { flex: 1 1 220px; min-width: 220px; }
      .alert { padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }
      .alert-primary { background: #cfe2ff; border: 1px solid #9ec5fe; }
      .alert-success { background: #d1e7dd; border: 1px solid #a3cfbb; }
      .alert-warning { background: #fff3cd; border: 1px solid #ffda6a; }
      .alert-danger { background: #f8d7da; border: 1px solid #f1aeb5; }
      .alert-info { background: #cff4fc; border: 1px solid #9eeaf9; }
      .card { display: block; background: #fff; border: 1px solid #dee2e6; border-radius: 6px; }
      .card-body { padding: 16px; }
      .card-title { margin: 0 0 8px; font-size: 20px; }
      .card-text { margin: 0 0 14px; }
      .btn { display: inline-block; border: 1px solid transparent; border-radius: 6px; padding: 8px 14px; text-decoration: none; cursor: pointer; }
      .btn-primary, .btn-success, .btn-danger, .btn-info { color: #fff; }
      .btn-primary { background: #0d6efd; }
      .btn-success { background: #198754; }
      .btn-warning { background: #ffc107; color: #212529; }
      .btn-danger { background: #dc3545; }
      .btn-info { background: #0dcaf0; color: #212529; }
      .btn-light { background: #f8f9fa; color: #212529; }
      .form-label { display: block; margin-bottom: 6px; font-weight: 600; }
      .form-control, .form-select { display: block; width: min(420px, 100%); min-height: 38px; padding: 8px 10px; border: 1px solid #adb5bd; border-radius: 6px; background: #fff; }
      textarea.form-control { min-height: 80px; }
      .form-check { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
      .nav-tabs { display: flex; gap: 4px; padding-left: 0; margin: 24px 0 16px; list-style: none; border-bottom: 1px solid #dee2e6; }
      .nav-link { display: block; padding: 8px 12px; text-decoration: none; border: 1px solid #dee2e6; border-bottom: 0; border-radius: 6px 6px 0 0; background: #fff; }
      .modal.show { display: block; position: static; max-width: 520px; margin: 24px 0; }
      .modal-dialog { border: 1px solid #adb5bd; border-radius: 6px; background: #fff; }
      .modal-content { padding: 16px; }
      table { width: 100%; border-collapse: collapse; background: #fff; margin-top: 24px; }
      th, td { border: 1px solid #dee2e6; padding: 8px 10px; text-align: left; }
      .accordion { margin-top: 24px; border: 1px solid #dee2e6; border-radius: 6px; background: #fff; }
      .accordion-item + .accordion-item { border-top: 1px solid #dee2e6; }
      .accordion-button { width: 100%; text-align: left; background: #fff; border: 0; padding: 12px 16px; font-weight: 700; }
      .accordion-body { padding: 0 16px 16px; }
      img { max-width: 160px; height: auto; display: block; margin-top: 16px; }
    </style>
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
        <label class="form-label mt-3" for="message">Mensagem</label>
        <textarea class="form-control" id="message">Texto de exemplo</textarea>
        <select class="form-select mt-2" aria-label="Plano">
          <option>Basico</option>
          <option>Pro</option>
        </select>
        <label class="form-check">
          <input type="checkbox" checked>
          Aceito os termos
        </label>
        <label class="form-check">
          <input type="radio" name="periodo" checked>
          Mensal
        </label>
        <button class="btn btn-primary mt-3" type="submit">Salvar</button>
      </form>
      <ul class="nav nav-tabs" role="tablist">
        <li><a class="nav-link" role="tab" href="#">Resumo</a></li>
        <li><a class="nav-link" role="tab" href="#">Detalhes</a></li>
      </ul>
      <p class="lead">
        Veja tambem <a href="/relatorios">relatorios recentes</a> e componentes relacionados.
      </p>
      <div class="modal show" role="dialog">
        <div class="modal-dialog">
          <div class="modal-content">
            <h2>Confirmar acao</h2>
            <p>Esta janela modal aparece aberta para rotulacao visual.</p>
            <button class="btn btn-{{ color }}" type="button">Confirmar</button>
          </div>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Nome</th><th>Email</th><th>Status</th></tr>
        </thead>
        <tbody>
          {% for row in rows %}
          <tr><td>{{ row.name }}</td><td>{{ row.email }}</td><td>{{ row.status }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
      <div class="accordion">
        <div class="accordion-item">
          <button class="accordion-button" type="button">Pergunta frequente</button>
          <div class="accordion-body">
            <p>Resposta curta para o componente accordion.</p>
          </div>
        </div>
      </div>
      <img
        alt="Grafico simples"
        src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='90'%3E%3Crect width='160' height='90' fill='%23dee2e6'/%3E%3Ccircle cx='45' cy='45' r='24' fill='%230d6efd'/%3E%3C/svg%3E"
      >
    </main>
  </body>
</html>
"""
)
