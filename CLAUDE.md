# CLAUDE.md

## Comandos

### Setup

- Criar ambiente local: `python3 -m venv venv`
- Instalar dependências mínimas atuais: `venv/bin/python -m pip install pytest typer pillow pyyaml jinja2 faker ruff mypy`

### Verificação

- Rodar testes: `venv/bin/python -m pytest -v`
- Rodar lint: `venv/bin/python -m ruff check .`
- Rodar typecheck: `venv/bin/python -m mypy src`

### Execução local

- Ver CLI sem instalar o pacote: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli --help`
- Gerar dataset sintético leve sem Playwright: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --synthetic-only --count 2 --output /tmp/ia-visao-web-dataset`
- Validar dataset: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate --root /tmp/ia-visao-web-dataset`
- Predição stub sem pesos treinados: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict <imagem.png>`

### Fluxo recomendado para testar manualmente

```bash
python3 -m venv venv
venv/bin/python -m pip install pytest typer pillow pyyaml jinja2 faker ruff mypy
venv/bin/python -m pytest -v
venv/bin/python -m ruff check .
venv/bin/python -m mypy src
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --synthetic-only --count 2 --output /tmp/ia-visao-web-dataset
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict /tmp/ia-visao-web-dataset/images/train/synthetic-00000.png
```

## Estrutura

- `docs/superpowers/specs/`: spec aprovado do MVP.
- `docs/superpowers/plans/`: plano de implementação detalhado.
- `src/ia_visao_web/cli.py`: CLI Typer.
- `src/ia_visao_web/sources/`: gerador determinístico de HTML Bootstrap.
- `src/ia_visao_web/renderer/`: fronteira opcional com Playwright.
- `src/ia_visao_web/labeler/`: taxonomia, seletores, geometria e filtro de matches DOM.
- `src/ia_visao_web/dataset/`: split determinístico, writer YOLO+JSON e validator.
- `src/ia_visao_web/model/`: vocabulários e interfaces opcionais de heads/loss Torch.
- `src/ia_visao_web/eval/`: métricas leves e serialização de predição.
- `tests/`: testes unitários e de integração escritos antes do código de produção.
- `data/dataset/`: saída gerada e ignorada pelo controle de versão.

## Decisões Arquiteturais

- O pacote usa layout `src/` para evitar import acidental de arquivos fora do pacote.
- A CLI é a fronteira pública; módulos internos permanecem pequenos e testáveis.
- Dependências pesadas ou opcionais, como Playwright, Torch e Ultralytics, devem falhar com mensagens acionáveis quando ausentes.
- O caminho `dataset build --synthetic-only` gera imagens PIL simples com labels determinísticas para CI/TDD; o render real por Chromium fica atrás do módulo `renderer`.
- `predict` retorna JSON válido com `detections: []` enquanto não houver pesos/model loader implementado.
- O fluxo desta execução não usa commits nem inicializa git, por pedido explícito do usuário.

## Gotchas

- `uv` não está instalado neste ambiente (`uv --version` falhou).
- O sandbox padrão de comandos falhou com `bwrap: loopback: Failed RTM_NEWADDR`; comandos estão rodando com permissão escalada quando necessário.
- O pacote ainda não foi instalado editable; comandos `python -m ia_visao_web.cli` precisam de `PYTHONPATH=src` fora do pytest.
- `dataset validate` usa por padrão o mínimo de 200 instâncias por classe no `train`; datasets pequenos de smoke test podem falhar nessa validação completa.
- Playwright, Torch, Ultralytics e pycocotools não foram instalados neste ciclo; as fronteiras opcionais foram testadas por erro acionável, não por execução real desses backends.
- O critério final de 3000 imagens, treino de 100 épocas e métricas reais depende de Chromium/GPU/dependências pesadas e ainda não foi validado neste ambiente.

## Verificações da Última Execução

- `venv/bin/python -m pytest -v`: 20 testes passaram.
- `venv/bin/python -m ruff check .`: passou.
- `venv/bin/python -m mypy src`: passou.
- `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --synthetic-only --count 2 --output /tmp/ia-visao-web-dataset-smoke`: passou.
