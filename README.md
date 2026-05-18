# ia-visao-web

MVP em Python para gerar screenshots sintéticos de páginas Bootstrap, rotular
componentes visuais por DOM e produzir um dataset YOLO com sidecars de atributos
HTML para treino multi-task.

## Estado atual

O projeto já possui um caminho leve e testável para desenvolvimento local:

- CLI Typer em `src/ia_visao_web/cli.py`.
- Gerador sintético determinístico de páginas Bootstrap.
- Escrita de dataset em formato YOLO + sidecar JSON de atributos.
- Validação básica de alinhamento entre labels e atributos.
- Fronteiras opcionais para Playwright, Torch e Ultralytics.
- Testes unitários e de integração para o que está implementado.

Ainda não há treino real YOLO/Ultralytics nem inferência com pesos treinados.
O comando `predict` retorna JSON válido com `detections: []` enquanto o loader do
modelo não existir.

## Setup local

Este ambiente não tem `uv` instalado, então o fluxo validado usa `venv`:

```bash
python3 -m venv venv
venv/bin/python -m pip install pytest typer pillow pyyaml jinja2 faker ruff mypy
```

Como o pacote usa layout `src/` e ainda não foi instalado em modo editable, rode
os comandos da CLI com `PYTHONPATH=src`.

## Rodar verificações

```bash
venv/bin/python -m pytest -v
venv/bin/python -m ruff check .
venv/bin/python -m mypy src
```

Resultado mais recente: `20 passed`.

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

Validar um dataset:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset
```

Observação: o validator usa por padrão o critério do spec de pelo menos 200
instâncias por classe no `train`; datasets pequenos de smoke test podem falhar
nessa validação completa.

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
src/ia_visao_web/cli.py      CLI Typer
src/ia_visao_web/sources/    Gerador de HTML Bootstrap
src/ia_visao_web/renderer/   Fronteira Playwright opcional
src/ia_visao_web/labeler/    Taxonomia, seletores, geometria, filtros DOM
src/ia_visao_web/dataset/    Split, writer YOLO+JSON, validator
src/ia_visao_web/model/      Vocabulários e fronteiras Torch
src/ia_visao_web/eval/       Métricas leves e serialização de predição
tests/                       Testes unitários e integração
```

## Dependências pesadas

Playwright, Torch, Ultralytics e pycocotools são dependências opcionais no
`pyproject.toml`. As fronteiras já falham com erro acionável quando ausentes,
mas o render real, treino real e métricas mAP ainda não foram executados neste
ambiente.
# htmlsight
