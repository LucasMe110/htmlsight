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
