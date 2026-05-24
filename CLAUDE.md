# CLAUDE.md

## Comandos

### Setup

- Instalar tudo (inclui Playwright + Chromium + torch + ultralytics): `INSTALL_MODEL=1 bash scripts/install-deps.sh`
- Instalar sem Chromium: `INSTALL_CHROMIUM=0 INSTALL_MODEL=1 bash scripts/install-deps.sh`
- Instalar sem dependências de modelo: `bash scripts/install-deps.sh`

### Verificação

- Rodar testes: `venv/bin/python -m pytest -v`
- Rodar lint: `venv/bin/python -m ruff check .`
- Rodar typecheck: `venv/bin/python -m mypy src`

### Execução local

- Ver ajuda da CLI: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli --help`
- Gerar dataset sintético leve (sem Playwright, para CI): `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --synthetic-only --count 2 --output /tmp/ds`
- Gerar dataset real com Playwright (produção): `PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --count 3000 --workers 4 --output data/dataset`
- Baixar docs Bootstrap 5.3 para usar como fonte real: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset fetch-docs --output data/sources/bootstrap-docs`
- Validar dataset: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate --root data/dataset --report`
- Validar dataset pequeno (sem mínimo de instâncias): `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate --root /tmp/ds --min-train-instances 0`
- Treinar (requer GPU para ser viável; sem GPU, use --epochs 1 para smoke test): `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train --dataset data/dataset --output runs/baseline --epochs 100`
- Gerar plano de treino sem executar: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train --dataset data/dataset --output runs/baseline --dry-run`
- Avaliar modelo treinado: `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli eval --dataset data/dataset --weights runs/baseline/weights/best.pt --split test`
- Predição em imagem (com pesos): `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict imagem.png --weights runs/baseline/weights/best.pt`
- Predição sem pesos (retorna `detections: []`): `PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict imagem.png`

### Fluxo completo de produção

```bash
# 1. Instalar tudo
INSTALL_MODEL=1 bash scripts/install-deps.sh

# 2. Gerar dataset de 3000 imagens
PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright \
  PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 3000 --workers 4 --output data/dataset

# 3. Validar
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root data/dataset --report

# 4. Treinar (GPU recomendada)
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \
  --dataset data/dataset --output runs/baseline --epochs 100

# 5. Avaliar
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli eval \
  --dataset data/dataset --weights runs/baseline/weights/best.pt --split test

# 6. Predição
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli predict imagem.png \
  --weights runs/baseline/weights/best.pt
```

## Estrutura

- `docs/superpowers/specs/`: spec aprovado do MVP.
- `docs/superpowers/plans/`: plano de implementação detalhado.
- `src/ia_visao_web/cli.py`: CLI Typer (fronteira pública).
- `src/ia_visao_web/sources/`: gerador determinístico de HTML Bootstrap + fetch de docs.
- `src/ia_visao_web/renderer/`: fronteira opcional com Playwright.
- `src/ia_visao_web/labeler/`: taxonomia, seletores, geometria e filtro de matches DOM.
- `src/ia_visao_web/dataset/`: split determinístico, writer YOLO+JSON e validator.
- `src/ia_visao_web/model/`: vocabulários, dataset loader PyTorch, loss multi-task.
- `src/ia_visao_web/eval/`: mAP metrics, evaluator, predict e serialização de predição.
- `tests/`: testes unitários e de integração escritos antes do código de produção.
- `data/dataset/`: saída gerada e ignorada pelo controle de versão.

## Decisões Arquiteturais

- O pacote usa layout `src/` para evitar import acidental de arquivos fora do pacote.
- A CLI é a fronteira pública; módulos internos permanecem pequenos e testáveis.
- Dependências pesadas ou opcionais, como Playwright, Torch e Ultralytics, devem falhar com mensagens acionáveis quando ausentes.
- O caminho `dataset build --synthetic-only` gera imagens PIL simples com labels determinísticas para CI/TDD; o render real por Chromium fica atrás do módulo `renderer`.
- `predict` retorna JSON válido com `detections: []` quando pesos não estão disponíveis; com `--weights` usa ultralytics YOLO.
- `_fixture_detections()` usa coordenadas absolutas que cabem em TODOS os viewports (máx 375px width): o bbox do navbar usa width=300 para caber no menor viewport (375x667).
- `_build_sample_worker()` é função module-level (necessário para pickling no ProcessPoolExecutor).
- `_PROCESS_RENDERER` global garante que cada worker process cria seu próprio renderer na primeira amostra.
- `ProcessPoolExecutor` e `as_completed` importados no topo do módulo para permitir monkeypatch em testes.
- `fetch_docs()` usa `urllib.request.urlopen` (stdlib) sem dependências extras.

## Possíveis Usos da IA

Quando o treinamento real estiver implementado, a IA pode ser usada para entender
interfaces web a partir de screenshots, detectando componentes visuais e
associando atributos HTML prováveis a cada detecção.

- Automação visual de testes: verificar se botões, inputs, cards, modais,
  alertas e outros componentes aparecem onde deveriam sem depender só do DOM.
- QA de interfaces: comparar screenshots entre versões e identificar mudanças
  visuais em componentes.
- Auditoria de UI: mapear automaticamente quais componentes existem em uma
  página ou aplicação.
- Agentes que usam sites: ajudar agentes de IA a localizar onde clicar,
  preencher campos ou navegar em uma tela.
- Extração de estrutura a partir de imagem: transformar screenshots em uma
  representação estruturada com classe, bbox, score e atributos HTML.
- Assistência para image-to-code: servir como etapa intermediária para
  reconstruir HTML/CSS a partir de uma imagem.
- Acessibilidade: detectar elementos que parecem botões, links ou inputs e
  comparar com tags, roles e estrutura HTML esperadas.
- Monitoramento visual de aplicações: checar em produção se telas essenciais
  renderizaram componentes críticos como login, navbar, formulário ou alerta.
- Análise de design systems: medir uso e distribuição de componentes Bootstrap
  e, futuramente, de outros frameworks.

## Ideia de Publicação Open Source

- Publicar o projeto como open source quando o pipeline estiver mais completo,
  com documentação clara de instalação, geração de dataset, treino, avaliação e
  inferência.
- Incluir exemplos pequenos e reproduzíveis para permitir que outras pessoas
  validem o funcionamento sem precisar de GPU ou dataset grande.
- Distribuir uma versão da IA já treinada, com pesos versionados, para que a
  comunidade possa testar `predict` diretamente em screenshots sem treinar do
  zero.
- Separar claramente o que é dataset sintético, modelo treinado, limitações
  conhecidas e próximos passos para evitar expectativas erradas sobre uso em
  sites reais arbitrários.
- Considerar publicar também cards de modelo/dataset com métricas, licença,
  escopo de uso, limitações e instruções para fine-tuning.

## Gotchas

- `uv` não está instalado neste ambiente; usar `venv` diretamente.
- O pacote não está instalado editable; comandos `python -m ia_visao_web.cli` precisam de `PYTHONPATH=src` fora do pytest.
- `dataset validate` exige por padrão ≥200 instâncias por classe no `train`; datasets pequenos de smoke test precisam de `--min-train-instances 0`.
- Playwright precisa da variável `PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright` para encontrar o Chromium instalado localmente.
- CUDA não está disponível nesta máquina (`torch.cuda.is_available()` = False); treino de 100 épocas precisa de GPU — sem GPU use `--epochs 1` para smoke test ou rode em ambiente com GPU.
- Atributos HTML (`tag`, `display`, `role`, `has_children`) ficam `null` no `predict` até que um modelo multi-task seja treinado; o modelo atual detecta classes e bboxes mas não infere atributos.
- `_build_sample_worker()` deve ser função module-level (não lambda/closure) para funcionar com `ProcessPoolExecutor` (pickle).
- `monkeypatch.setitem(sys.modules, "ultralytics", None)` é a forma correta de simular ultralytics ausente nos testes (não `setattr`).

## Ambiente atual

- Python: 3.12
- torch: 2.12.0+cu130 (instalado, sem GPU disponível)
- ultralytics: 8.4.53 (instalado)
- playwright: instalado, Chromium em `venv/ms-playwright`
- Testes: 80 passando, 0 pulados

## Verificações da Última Execução

- `venv/bin/python -m pytest -v`: 80 passaram.
- `venv/bin/python -m ruff check .`: passou.
- `venv/bin/python -m mypy src`: passou (26 arquivos fonte).
- `dataset build --count 3000 --workers 4`: em execução com Playwright real.
