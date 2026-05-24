# htmlsight

Detector visual multi-task de componentes web Bootstrap treinado com dataset sintético auto-rotulado.

Dado um screenshot de uma página web, o modelo detecta **botões, inputs, cards, navbars, modais** e mais 12 classes, retornando bounding boxes com score de confiança — tudo a partir de uma imagem, sem tocar no DOM.

```json
{
  "image": "pagina.png",
  "detections": [
    { "class": "button", "score": 0.94, "bbox": [120, 45, 80, 32], "attrs": {...} },
    { "class": "navbar", "score": 0.99, "bbox": [0, 0, 1280, 56],  "attrs": {...} },
    { "class": "card",   "score": 0.87, "bbox": [40, 120, 320, 200], "attrs": {...} }
  ]
}
```

---

## Pré-requisitos

| Requisito | Versão mínima | Notas |
|-----------|---------------|-------|
| Python | 3.11+ | |
| Git | qualquer | |
| GPU CUDA | recomendado | CPU funciona, mas o treino é muito lento |

---

## Instalação rápida

```bash
git clone git@github.com:LucasMe110/htmlsight.git
cd htmlsight
INSTALL_MODEL=1 bash scripts/install-deps.sh
```

Isso cria um `venv/` local com **todas** as dependências:
- `torch` + `ultralytics` (YOLOv8)
- `playwright` + Chromium headless
- `pytest`, `ruff`, `mypy` (dev)

> **Sem GPU?** O treino funciona em CPU mas leva horas. Recomendado usar Google Colab ou uma máquina com GPU para o passo de treino.

---

## Instalação por partes

### Só rodar testes / gerar dataset sintético (sem GPU, sem Chromium)

```bash
bash scripts/install-deps.sh INSTALL_CHROMIUM=0
```

### Com Playwright mas sem torch/ultralytics

```bash
bash scripts/install-deps.sh
```

### Completo (Playwright + torch + ultralytics)

```bash
INSTALL_MODEL=1 bash scripts/install-deps.sh
```

---

## Verificar instalação

```bash
venv/bin/python -m pytest -v          # 85 testes devem passar
venv/bin/python -m ruff check .       # lint
venv/bin/python -m mypy src           # typecheck
```

---

## Workflow completo

### 1. Gerar dataset (3000 imagens com Playwright)

```bash
PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright \
  PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 3000 \
  --workers 4 \
  --output data/dataset
```

> Para testar sem Playwright, use `--synthetic-only` (gera imagens PIL simples):
> ```bash
> PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
>   --synthetic-only --count 50 --output data/dataset
> ```

### 2. Validar dataset

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root data/dataset \
  --report
```

Escreve overlays de QA em `data/dataset/_qa/` e um JSON com distribuição de classes.

### 3. Treinar

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \
  --dataset data/dataset \
  --output runs/baseline \
  --epochs 100 \
  --batch-size 16
```

Os pesos são salvos em `runs/baseline/weights/best.pt` (melhor época) e `last.pt`.

> Ver o plano de treino sem executar:
> ```bash
> PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \
>   --dataset data/dataset --output runs/baseline --dry-run
> ```

### 4. Gerar relatório de métricas

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli report \
  --dataset data/dataset \
  --weights runs/baseline/weights/best.pt \
  --output runs/baseline/report
```

Gera `runs/baseline/report/report.md` (Markdown para post/README) e `report.json` (métricas completas).

### 5. Predição em imagem

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict \
  screenshot.png \
  --weights runs/baseline/weights/best.pt
```

Sem `--weights`, retorna `detections: []` (útil para testar o pipeline).

### 6. Avaliação de métricas

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli eval \
  --dataset data/dataset \
  --weights runs/baseline/weights/best.pt \
  --split test
```

---

## Taxonomia — 17 classes detectadas

| Classe | Seletor CSS principal |
|--------|-----------------------|
| `button` | `button.btn`, `.btn`, `[role='button']` |
| `input` | `input[type='text']`, email, password, ... |
| `textarea` | `textarea` |
| `checkbox` | `input[type='checkbox']` |
| `radio` | `input[type='radio']` |
| `select` | `select`, `.form-select` |
| `link` | `a:not(.btn):not(.nav-link)` |
| `card` | `.card` |
| `navbar` | `.navbar` |
| `tabs` | `.nav-tabs`, `.nav-pills` |
| `modal` | `.modal.show`, `.modal-dialog` |
| `table` | `table` |
| `alert` | `.alert` |
| `accordion` | `.accordion` |
| `image` | `img`, `picture` |
| `text` | `p`, `h1`–`h6`, `.lead` |
| `container` | `.container`, `.row`, `.col`, `.card-body` |

---

## Estrutura do projeto

```
htmlsight/
├── scripts/
│   ├── install-deps.sh        # setup do ambiente
│   └── ralph/                 # loop de desenvolvimento autônomo
├── src/ia_visao_web/
│   ├── cli.py                 # CLI Typer (fronteira pública)
│   ├── sources/               # gerador HTML Bootstrap + fetch docs
│   ├── renderer/              # fronteira Playwright (opcional)
│   ├── labeler/               # taxonomia, seletores, DOM walker, geometria
│   ├── dataset/               # split, writer YOLO+JSON, validator
│   ├── model/                 # vocab, dataset loader PyTorch, loss multi-task
│   └── eval/                  # mAP, evaluator, predict, report
├── tests/
│   ├── unit/                  # testes por módulo
│   └── integration/           # pipeline end-to-end
├── docs/
│   └── superpowers/           # spec e plano de implementação
├── pyproject.toml
└── CLAUDE.md                  # instruções para agentes IA
```

---

## Desenvolvimento

```bash
# Rodar testes
venv/bin/python -m pytest -v

# Lint
venv/bin/python -m ruff check .

# Typecheck
venv/bin/python -m mypy src

# Dataset sintético rápido para smoke test
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --synthetic-only --count 5 --output /tmp/smoke
```

O projeto usa **TDD** — testes são escritos antes da implementação. Todas as dependências pesadas (Playwright, torch, Ultralytics) falham com erros acionáveis quando não instaladas.

---

## Dependências

| Dependência | Extra | Uso |
|-------------|-------|-----|
| `typer`, `pillow`, `jinja2`, `faker`, `pyyaml` | _(base)_ | geração e CLI |
| `playwright` | `render` | renderização real com Chromium |
| `torch`, `ultralytics`, `pycocotools` | `model` | treino e avaliação |
| `pytest`, `ruff`, `mypy` | `dev` | qualidade de código |

---

## Casos de uso

- **Testes visuais automatizados** — verificar se componentes aparecem sem depender do DOM
- **QA de interfaces** — comparar screenshots entre versões
- **Agentes de IA** — localizar onde clicar ou preencher em uma tela
- **Image-to-code** — etapa intermediária para reconstruir HTML a partir de screenshot
- **Monitoramento visual** — checar se telas críticas renderizaram corretamente em produção
