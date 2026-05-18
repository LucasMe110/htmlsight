# ia-visao-web

MVP em Python para gerar screenshots sintéticos de páginas Bootstrap, rotular
componentes visuais por DOM e produzir um dataset YOLO com sidecars de atributos
HTML para treino multi-task.

## Estado atual

O projeto já possui um caminho leve e testável para desenvolvimento local:

- CLI Typer em `src/ia_visao_web/cli.py`.
- Gerador sintético determinístico de páginas Bootstrap.
- Render real opcional com Playwright/Chromium.
- DOM walker real que aplica a taxonomia sobre a página renderizada.
- Escrita de dataset em formato YOLO + sidecar JSON de atributos.
- Validação de labels YOLO, alinhamento de sidecars, cobertura opcional por
  split, QA visual e relatório de distribuição.
- Fronteiras opcionais para Playwright, Torch e Ultralytics.
- Testes unitários e de integração para o que está implementado.

Ainda não há treino real YOLO/Ultralytics nem inferência com pesos treinados.
O comando `predict` retorna JSON válido com `detections: []` enquanto o loader do
modelo não existir.

## Setup local

Este ambiente não tem `uv` instalado, então o fluxo validado usa `venv`:

```bash
scripts/install-deps.sh
```

O script instala o pacote em modo editable com os extras `dev` e `render`, e
baixa o Chromium do Playwright em `venv/ms-playwright`.

Para instalar sem baixar Chromium:

```bash
INSTALL_CHROMIUM=0 scripts/install-deps.sh
```

Para incluir dependências pesadas de modelo:

```bash
INSTALL_MODEL=1 scripts/install-deps.sh
```

## Rodar verificações

```bash
venv/bin/python -m pytest -v
venv/bin/python -m ruff check .
venv/bin/python -m mypy src
```

Resultado mais recente: `30 passed`.

## Rodar a CLI

Ver ajuda:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli --help
```

Gerar um dataset sintético pequeno, sem Playwright:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --synthetic-only \
  --count 2 \
  --output /tmp/ia-visao-web-dataset
```

Gerar um dataset real pequeno com Playwright + DOM walker:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-real
```

Validar um dataset:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset
```

Validar dataset pequeno sem exigir 200 instâncias por classe e gerar QA visual:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-real \
  --min-train-instances 0 \
  --qa-samples 2 \
  --report
```

Isso escreve overlays em `_qa/*.png` e um relatório em `_qa/report.json`.

Observação: o validator usa por padrão o critério do spec de pelo menos 200
instâncias por classe no `train`; datasets pequenos de smoke test podem falhar
nessa validação completa.

Gerar um plano de treino avaliável, sem executar o backend YOLO:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \
  --dataset /tmp/ia-visao-web-dataset-real \
  --output /tmp/ia-visao-web-runs/baseline \
  --epochs 100 \
  --batch-size 16 \
  --image-size 640 \
  --eval-split test \
  --eval-every 2 \
  --conf-threshold 0.25 \
  --iou-threshold 0.50 \
  --failure-examples 50 \
  --lambda-tag 0.2 \
  --lambda-display 0.2 \
  --lambda-role 0.2 \
  --lambda-has-children 0.1 \
  --dry-run
```

Isso escreve `training-plan.json` com hiperparâmetros, thresholds de avaliação,
pesos das losses multi-task e opções para salvar predições, plots e exemplos de
falha. O treino real ainda não executa Ultralytics; o plano serve para comparar
experimentos quando o backend for implementado.

Rodar predição stub:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict /tmp/ia-visao-web-dataset/images/train/synthetic-00000.png
```

Saída esperada no estado atual:

```json
{
  "image": "/tmp/ia-visao-web-dataset/images/train/synthetic-00000.png",
  "detections": []
}
```

## Estrutura

```text
docs/superpowers/specs/      Spec aprovado
docs/superpowers/plans/      Plano de implementação
docs/PDR-*.md                PDRs de evolução do projeto
scripts/install-deps.sh      Instala dependências e browsers Playwright
src/ia_visao_web/cli.py      CLI Typer
src/ia_visao_web/sources/    Gerador de HTML Bootstrap
src/ia_visao_web/renderer/   Fronteira Playwright
src/ia_visao_web/labeler/    Taxonomia, seletores, DOM walker, geometria
src/ia_visao_web/dataset/    Split, writer YOLO+JSON, validator
src/ia_visao_web/model/      Vocabulários e fronteiras Torch
src/ia_visao_web/eval/       Métricas leves e serialização de predição
tests/                       Testes unitários e integração
```

## Dependências pesadas

Playwright, Torch, Ultralytics e pycocotools são dependências opcionais no
`pyproject.toml`. O render real com Playwright foi validado neste ambiente. O
treino real e métricas mAP ainda não foram implementados.
# htmlsight
