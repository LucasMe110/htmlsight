# PDR: Dataset Real com Playwright + DOM Walker

Status: concluido em 2026-05-18.

## Objetivo

Evoluir o projeto do estado atual, em que `dataset build --synthetic-only` gera
imagens PIL simples com deteccoes fixas, para um pipeline real de dataset:

1. Gerar HTML Bootstrap sintetico.
2. Renderizar o HTML em Chromium via Playwright.
3. Percorrer o DOM renderizado usando seletores CSS da taxonomia.
4. Extrair bounding boxes e atributos HTML reais.
5. Escrever imagens, labels YOLO e sidecars JSON alinhados.

Esse passo e pre-requisito para treinar uma IA util. Sem ele, o modelo aprenderia
apenas fixtures artificiais, nao componentes visuais reais.

## Problema Atual

Hoje o fluxo `dataset build --synthetic-only`:

- Gera HTML com `BootstrapPageGenerator`.
- Ignora o HTML para criar a imagem.
- Cria uma imagem simples com `PIL.ImageDraw`.
- Usa `_fixture_detections()` com bboxes fixos.
- Escreve o dataset no formato correto.

Isso e bom para testar contrato de arquivos, CLI, writer e validator, mas nao
produz dados suficientes para treino real.

## Resultado Esperado

Ao final desta etapa, o comando de build deve conseguir gerar amostras reais:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 100 \
  --output /tmp/ia-visao-web-dataset-real
```

Para cada amostra:

- `images/{split}/{id}.png`: screenshot real da pagina renderizada.
- `labels/{split}/{id}.txt`: labels YOLO normalizados.
- `attrs/{split}/{id}.json`: atributos HTML na mesma ordem das labels.
- `data.yaml`: configuracao de classes para YOLO/Ultralytics.

O modo `--synthetic-only` pode continuar existindo como smoke test rapido sem
Playwright.

## Escopo

### Dentro do Escopo

- Integrar `PlaywrightRenderer` no comando `dataset build`.
- Criar um walker DOM real executado via Playwright.
- Usar `SELECTORS` como fonte unica da taxonomia.
- Extrair:
  - `class_name`
  - `x`, `y`, `width`, `height`
  - `tag`
  - `display`
  - `role`
  - `has_children`
  - `n_descendants`
  - `visible`
- Reusar `filter_matches` para filtros de area, visibilidade, viewport e
  duplicatas.
- Escrever PNG, YOLO txt e JSON usando `DatasetWriter`.
- Manter caminho deterministico por `seed=index`.
- Adicionar testes unitarios para montagem dos scripts/contratos do walker.
- Adicionar pelo menos um teste de integracao com Playwright marcado/isolado ou
  protegido por skip quando Playwright/Chromium nao estiver instalado.
- Melhorar mensagens de erro quando Playwright ou Chromium estiver ausente.

### Fora do Escopo Nesta Etapa

- Treinar YOLOv8.
- Implementar `predict` real com pesos.
- Baixar snapshot da documentacao oficial do Bootstrap.
- Cobrir sites reais arbitrarios.
- Otimizar performance para milhares de paginas.
- Publicar pesos treinados.

## Status Atual da Implementacao

Data da ultima atualizacao: 2026-05-18.

Este PDR deve ser tratado como documento vivo. Sempre que uma atividade for
implementada, ela deve ser marcada como concluida nesta secao junto com a
validacao executada.

### Ja Foi Feito

- [x] Criado `src/ia_visao_web/labeler/dom_walker.py`.
  - Implementa `DomWalker`.
  - Executa JS via `page.evaluate`.
  - Usa `SELECTORS` como fonte unica dos seletores.
  - Converte payload do browser em `RawDomMatch`.
- [x] Criados testes unitarios do DOM walker.
  - Arquivo: `tests/unit/labeler/test_dom_walker.py`.
  - Cobre payload de seletores, conversao do retorno JS e erro para payload
    invalido.
- [x] Ajustado `PlaywrightRenderer` para expor pagina viva.
  - Arquivo: `src/ia_visao_web/renderer/playwright_renderer.py`.
  - Novo metodo `open_page(...)`.
  - Mantem browser aberto enquanto DOM e screenshot sao coletados.
  - Fecha browser e Playwright no `finally`.
- [x] Integrado build real no CLI.
  - Arquivo: `src/ia_visao_web/cli.py`.
  - `dataset build --synthetic-only` preserva fluxo antigo com fixtures.
  - `dataset build` sem `--synthetic-only` usa Playwright + `DomWalker`.
  - Screenshot e bboxes passam a vir da mesma pagina renderizada.
- [x] Melhorada a validacao de labels YOLO.
  - Arquivo: `src/ia_visao_web/dataset/validator.py`.
  - Valida quantidade de campos.
  - Valida `class_id` dentro da taxonomia.
  - Valida `cx`, `cy`, `w`, `h` entre `0` e `1`.
  - Rejeita largura/altura zeradas.
- [x] Criados testes do validator para labels invalidas.
  - Arquivo: `tests/unit/dataset/test_validator.py`.
- [x] Criado teste de integracao opcional com Playwright.
  - Arquivo: `tests/integration/test_playwright_dataset_build.py`.
  - O teste pula com `pytest.skip` quando Playwright/Chromium nao esta
    disponivel.
- [x] Criado script de instalacao de dependencias.
  - Arquivo: `scripts/install-deps.sh`.
  - Instala pacote em modo editable com extras `dev` e `render`.
  - Opcionalmente instala extras de modelo com `INSTALL_MODEL=1`.
  - Opcionalmente instala Chromium do Playwright com `INSTALL_CHROMIUM=1`.
  - Instala browsers em `$VENV_DIR/ms-playwright` para evitar cache read-only em
    `/home/npu/.cache`.
- [x] Instaladas dependencias Python base/dev/render no `venv`.
  - Comando executado: `venv/bin/python -m pip install -e '.[dev,render]'`.
  - Resultado: `playwright` Python instalado junto com o pacote editable.
- [x] Instalado Chromium do Playwright em `venv/ms-playwright`.
  - Primeira tentativa falhou por cache read-only.
  - Segunda tentativa com `PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright` e rede
    liberada completou.
- [x] Renderer ajustado para localizar automaticamente `venv/ms-playwright`.
  - Isso evita exigir `PLAYWRIGHT_BROWSERS_PATH` em todo comando local.

### Validacoes Ja Executadas

- [x] Testes automatizados:

```bash
venv/bin/python -m pytest -q
```

Resultado registrado final: `31 passed`.

- [x] Lint:

```bash
venv/bin/python -m ruff check .
```

Resultado registrado: `All checks passed!`.

- [x] Typecheck:

```bash
venv/bin/python -m mypy src
```

Resultado registrado: `Success: no issues found`.

- [x] Smoke test do modo fixture:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --synthetic-only \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-fixture
```

Resultado registrado: dataset fixture gerado com sucesso.

- [x] Smoke test de erro acionavel sem Chromium disponivel:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 1 \
  --output /tmp/ia-visao-web-dataset-real-smoke
```

Resultado registrado: comando falhou com mensagem orientando instalar
`playwright` e `chromium`, como esperado quando runtime esta ausente.

- [x] Teste de integracao real com Playwright:

```bash
PLAYWRIGHT_BROWSERS_PATH=venv/ms-playwright \
  venv/bin/python -m pytest tests/integration/test_playwright_dataset_build.py -q
```

Resultado registrado: `1 passed`.

- [x] Build real pequeno com Playwright:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-real
```

Resultado registrado: dataset real escrito com PNGs, labels, attrs e
`data.yaml`.

- [x] Validacao do dataset real pequeno:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-real
```

Resultado registrado: falhou somente por contagem minima de 200 instancias por
classe, esperado para `--count 2`. Nao houve erro de alinhamento entre labels e
attrs.

- [x] Build real pequeno sem `PLAYWRIGHT_BROWSERS_PATH` explicito:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 1 \
  --output /tmp/ia-visao-web-dataset-real-no-env
```

Resultado registrado: passou, confirmando que o renderer encontrou
`venv/ms-playwright`.

- [x] Build real final cobrindo todas as classes:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-final-pdr
```

Resultado registrado: passou.

- [x] Validacao final com QA visual e relatorio:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-final-pdr \
  --min-train-instances 0 \
  --qa-samples 2 \
  --report
```

Resultado registrado: `dataset ok`, overlays em `_qa/*.png` e relatorio em
`_qa/report.json`.

- [x] Cobertura da taxonomia no dataset real pequeno:

Resultado registrado: nenhuma classe ausente; total de 68 labels em `--count 2`.

### Ainda Nao Foi Feito

- [x] Confirmar instalacao do Chromium do Playwright neste ambiente.
  - Primeira tentativa com `venv/bin/python -m playwright install chromium`
    falhou porque o Playwright tentou criar cache em `/home/npu/.cache`, que e
    somente leitura no sandbox.
  - O script `scripts/install-deps.sh` foi ajustado para usar
    `PLAYWRIGHT_BROWSERS_PATH=$VENV_DIR/ms-playwright`, que fica em diretorio
    gravavel do projeto.
- [x] Rodar `dataset build` real com Chromium instalado.
  - Objetivo minimo:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-real
```

- [x] Validar visualmente imagens reais geradas.
  - Foram gerados overlays em `_qa/` para revisao visual.
- [x] Implementar `--qa-samples` no comando `dataset validate`.
  - Deve desenhar bboxes nas imagens e salvar em `_qa/`.
- [x] Expandir o gerador HTML para cobrir todas as 17 classes da taxonomia.
  - O build real final registrou nenhuma classe ausente.
- [x] Reduzir dependencia de CDN do Bootstrap.
  - O HTML gerado agora usa CSS embutido e nao depende de CDN.
- [x] Melhorar tratamento de screenshot `full_page=True`.
  - O writer normaliza as bboxes usando dimensoes reais do PNG final.
  - O validator confirma bboxes no intervalo `0-1`.
- [x] Adicionar validacao de cobertura por split.
  - Garantir presenca minima de classes relevantes em `train`, `val` e `test`.
- [x] Adicionar relatorio de distribuicao de classes/areas.
  - Ajuda a detectar excesso de `container` ou `text`.
- [x] Decidir se `container` deve ficar ligado por padrao.
  - Decisao: manter ligado por padrao no MVP porque representa estrutura visual.
  - Risco documentado: pode gerar muitas labels e devera ser reavaliado antes
    de treino grande.
- [x] Documentar fluxo atualizado no `README.md`.
  - Incluir instalacao via `scripts/install-deps.sh`.
  - Incluir build real com Playwright.
  - Incluir limitacoes atuais.

### Regra de Registro de Conclusao

Ao concluir qualquer item pendente:

1. Marcar o checkbox correspondente como `[x]`.
2. Adicionar o arquivo alterado, quando houver.
3. Registrar o comando de validacao executado.
4. Registrar o resultado observado.
5. Se algo nao puder ser validado, anotar o motivo explicitamente.

## Design Proposto

### 1. Renderer deve devolver screenshot e pagina viva

O `PlaywrightRenderer` atual devolve apenas `RenderedPage(png, viewport)`.
Para rotular via DOM, o pipeline precisa executar JS antes de fechar a pagina.

Opcoes:

- Adicionar metodo de alto nivel:

```python
render_and_label(html, viewport, labeler) -> RenderedLabeledPage
```

- Ou criar um metodo de contexto:

```python
with renderer.open_page(html, viewport) as page:
    matches = DomWalker().collect(page)
    png = page.screenshot(...)
```

Preferencia: segunda opcao. Ela mantem o renderer responsavel por Playwright e
permite que o labeler continue separado.

### 2. Criar `labeler/dom_walker.py`

Novo modulo sugerido:

```text
src/ia_visao_web/labeler/dom_walker.py
```

Responsabilidade:

- Receber uma pagina Playwright.
- Executar JS que aplica todos os seletores.
- Retornar `list[RawDomMatch]`.

API sugerida:

```python
class DomWalker:
    def collect(self, page: Any) -> list[RawDomMatch]:
        ...
```

O modulo deve evitar importar Playwright diretamente quando possivel, usando
`Any` ou `Protocol`, para manter testes leves.

### 3. Script JS de coleta

O JS deve receber as regras como dados:

```json
[
  {"class_name": "button", "selector": "button.btn, .btn, [role='button']"},
  ...
]
```

Para cada elemento encontrado:

```javascript
const rect = element.getBoundingClientRect();
const style = window.getComputedStyle(element);

return {
  class_name,
  x: rect.x,
  y: rect.y,
  width: rect.width,
  height: rect.height,
  tag: element.tagName.toLowerCase(),
  display: style.display,
  role: element.getAttribute("role"),
  has_children: element.children.length > 0,
  n_descendants: element.querySelectorAll("*").length,
  visible: style.visibility !== "hidden" && style.display !== "none"
};
```

### 4. CLI `dataset build`

Comportamento desejado:

- `--synthetic-only`: mantem fluxo atual com PIL e fixtures.
- Sem `--synthetic-only`: usa Playwright real.

Fluxo real:

```python
writer = DatasetWriter(output)
renderer = PlaywrightRenderer()
walker = DomWalker()

for index in range(count):
    sample_id = f"synthetic-{index:05d}"
    page_source = BootstrapPageGenerator(seed=index).generate_page(sample_id)
    rendered = renderer.render_html_with_page(page_source.html, page_source.viewport, walker)
    detections = filter_matches(...)
    writer.write_sample(sample_id, image, detections, split)
```

O desenho exato pode variar, mas o ponto central e que o screenshot e as bboxes
venham da mesma pagina renderizada.

## Criterios de Aceite

- `dataset build --synthetic-only` continua funcionando.
- `dataset build` sem `--synthetic-only` tenta usar Playwright e falha com erro
  acionavel se a dependencia ou Chromium estiver ausente.
- Com Playwright instalado, `dataset build --count 2` gera screenshots reais.
- Cada label YOLO tem um item correspondente em `attrs`.
- As bboxes ficam normalizadas entre `0` e `1`.
- O dataset gerado contem pelo menos algumas classes reais do HTML atual:
  `navbar`, `button`, `card`, `alert`, `input`, `select`, `text`, `container`.
- Testes existentes continuam passando.

## Validacao das Implementacoes

Cada parte implementada deve ter uma forma objetiva de provar que funcionou. A
validacao combina testes unitarios, testes de integracao opcionais, smoke tests
via CLI e inspecao visual de amostras.

### 1. Validar que o fluxo antigo nao quebrou

Rodar:

```bash
venv/bin/python -m pytest -q
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --synthetic-only \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-fixture
```

Resultado esperado:

- Todos os testes existentes passam.
- O comando `--synthetic-only` gera arquivos em `images/`, `labels/`, `attrs/`
  e `data.yaml`.
- Nenhuma dependencia de Playwright e exigida nesse modo.

### 2. Validar DOM walker sem Chromium

Adicionar testes unitarios para o novo `DomWalker` cobrindo:

- Conversao de `SELECTORS` para payload serializavel.
- Conversao do retorno bruto do JS para `RawDomMatch`.
- Preservacao da ordem dos seletores.
- Tratamento de `role: null`.
- Truncamento de `n_descendants` continua sendo feito em `filter_matches`.

Exemplo de teste esperado:

```python
def test_dom_walker_converts_browser_payload_to_raw_matches():
    page = FakePageReturning([...])

    matches = DomWalker().collect(page)

    assert matches[0].class_name == "button"
    assert matches[0].tag == "button"
    assert matches[0].display == "inline-block"
```

Resultado esperado:

- Esse teste roda sem Playwright instalado.
- Falhas no contrato entre JS e Python aparecem antes do teste de integracao.

### 3. Validar filtros de deteccao

Manter e expandir testes de `filter_matches` para cobrir:

- Elementos pequenos sao descartados.
- Elementos invisiveis sao descartados.
- Elementos fora da viewport sao descartados.
- Duplicatas da mesma classe com IoU alto sao descartadas.
- Elementos de classes diferentes com bbox similar nao sao descartados entre si.
- Atributos finais contem `tag`, `display`, `role`, `has_children` e
  `n_descendants`.

Resultado esperado:

- O dataset recebe somente deteccoes utilizaveis.
- Mudancas futuras nos filtros nao alteram silenciosamente o comportamento.

### 4. Validar renderer e lifecycle do Playwright

Adicionar teste de unidade ou integracao leve para:

- Erro claro quando `playwright` nao esta instalado.
- Erro claro quando Chromium/runtime nao esta instalado.
- Browser fecha mesmo quando a coleta do DOM falha.

Resultado esperado:

- A CLI orienta como instalar a dependencia ausente.
- Nao ficam processos de browser abertos apos erro.

### 5. Validar integracao real com Playwright

Adicionar um teste de integracao opcional, com skip automatico se Playwright ou
Chromium nao estiverem disponiveis:

```bash
venv/bin/python -m pytest tests/integration/test_playwright_dataset_build.py -q
```

O teste deve:

- Gerar uma pagina HTML simples com Bootstrap.
- Renderizar via Playwright.
- Coletar deteccoes pelo DOM walker.
- Escrever uma amostra com `DatasetWriter`.
- Validar que label e sidecar tem o mesmo numero de itens.

Resultado esperado:

- Com Playwright instalado, o teste passa.
- Sem Playwright instalado, o teste fica `skipped`, nao `failed`.

### 6. Validar CLI real

Rodar manualmente:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-real
```

Depois conferir:

```bash
find /tmp/ia-visao-web-dataset-real -type f | sort
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-real
```

Resultado esperado:

- Existem PNGs reais, labels txt, attrs JSON e `data.yaml`.
- Cada `.txt` tem o mesmo numero de linhas que o JSON correspondente.
- As classes detectadas incluem componentes presentes no HTML.
- A validacao completa pode falhar por baixa contagem em dataset pequeno, mas
  nao deve falhar por desalinhamento entre labels e attrs.

### 7. Validar normalizacao YOLO

Adicionar teste ou verificacao no validator para garantir:

- Cada linha tem 5 campos.
- `class_id` esta entre `0` e `len(TAXONOMY) - 1`.
- `cx`, `cy`, `w`, `h` estao entre `0` e `1`.
- `w` e `h` sao maiores que `0`.

Resultado esperado:

- Labels invalidas sao detectadas antes do treino.

### 8. Validar QA visual

Adicionar geracao opcional de overlays em `_qa/`, por exemplo:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-real \
  --qa-samples 10
```

Resultado esperado:

- O diretorio `_qa/` contem imagens com bboxes desenhadas.
- A revisao manual confirma que as caixas estao sobre os componentes corretos.
- Erros comuns ficam visiveis rapidamente: bbox deslocada, escala errada,
  labels demais em containers ou textos, elementos invisiveis incluidos.

### 9. Validar cobertura minima por classe

Depois que o gerador HTML for expandido, rodar build maior:

```bash
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --count 300 \
  --output /tmp/ia-visao-web-dataset-real-300
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset validate \
  --root /tmp/ia-visao-web-dataset-real-300
```

Resultado esperado:

- Classes centrais aparecem no `train`.
- O validator informa claramente classes ausentes ou abaixo do minimo.
- Antes do treino real, o objetivo e chegar ao minimo definido no validator para
  todas as classes relevantes.

### 10. Checklist de pronto

A implementacao desta etapa so deve ser considerada pronta quando:

- Testes unitarios passam.
- Testes de integracao com Playwright passam ou pulam corretamente quando a
  dependencia nao existe.
- `dataset build --synthetic-only` ainda funciona.
- `dataset build` real funciona em ambiente com Playwright/Chromium.
- Labels e attrs ficam alinhados.
- Bboxes YOLO ficam normalizadas corretamente.
- Existe pelo menos uma amostra QA visual revisada manualmente.
- Limitacoes conhecidas foram anotadas no `README.md` ou `CLAUDE.md`.

## Plano de Implementacao

### Passo 1: Preservar o smoke test atual

Status: concluido.

- [x] Manter `_synthetic_image()` e `_fixture_detections()` por enquanto.
- [x] Garantir que `dataset build --synthetic-only` continua funcionando.
- [x] Garantir que testes existentes continuam verdes.

Validacao registrada:

```bash
venv/bin/python -m pytest -q
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build \
  --synthetic-only \
  --count 2 \
  --output /tmp/ia-visao-web-dataset-fixture
```

### Passo 2: Criar DOM walker real

Status: concluido.

- [x] Adicionar `src/ia_visao_web/labeler/dom_walker.py`.
- [x] Converter `SELECTORS` em payload serializavel.
- [x] Executar `page.evaluate(...)`.
- [x] Transformar o retorno JS em `RawDomMatch`.
- [x] Testar conversao de payload sem depender de Chromium.

Validacao registrada:

```bash
venv/bin/python -m pytest tests/unit/labeler/test_dom_walker.py -q
```

### Passo 3: Ajustar renderer

- [x] Adicionar uma API que permita coletar DOM e screenshot no mesmo ciclo de vida
  da pagina.
- [x] Garantir fechamento de browser mesmo quando houver erro.
- [x] Manter `RendererUnavailableError` com mensagens claras.

Status: concluido.

Validacao registrada:

```bash
venv/bin/python -m pytest tests/unit/renderer/test_playwright_renderer.py -q
```

### Passo 4: Integrar no CLI

- [x] Alterar `dataset_build`.
- [x] Quando `synthetic_only=False`, usar renderer + walker real.
- [x] Converter bytes PNG em `PIL.Image` antes de escrever pelo `DatasetWriter`, ou
  permitir que o writer aceite bytes/imagem.

Status: concluido.

Validacao registrada:

```bash
venv/bin/python -m pytest tests/unit/test_cli.py -q
```

### Passo 5: Teste de integracao opcional

- [x] Criar teste de integracao para Playwright.
- [x] Se Playwright/Chromium nao estiver disponivel, usar `pytest.skip`.
- [x] Validar que pelo menos uma bbox e um sidecar sao gerados quando o runtime
  esta disponivel.

Status: concluido.

Validacao registrada:

```bash
venv/bin/python -m pytest tests/integration/test_playwright_dataset_build.py -q
```

Resultado final com Chromium instalado: `1 passed`.

### Passo 6: Validacao visual de QA

- [x] Depois da integracao basica, adicionar geracao opcional de overlays em
  `_qa/` com bboxes desenhadas.
- [x] Isso ajuda a revisar se as labels batem com os componentes renderizados.

Status: concluido.

### Passo 7: Instalar e validar Chromium

- [x] Identificar falha de cache read-only em `/home/npu/.cache`.
- [x] Ajustar `scripts/install-deps.sh` para instalar browsers em
  `$VENV_DIR/ms-playwright`.
- [x] Confirmar se a instalacao com `PLAYWRIGHT_BROWSERS_PATH` completa neste
  ambiente.
- [x] Rodar build real com `--count 2`.
- [x] Registrar resultado nesta mesma secao.

Status: concluido.

### Passo 8: Atualizar documentacao de uso

- [x] Atualizar `README.md` com `scripts/install-deps.sh`.
- [x] Documentar fluxo base:

```bash
scripts/install-deps.sh
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --count 2
```

- [x] Documentar fluxo sem Chromium:

```bash
INSTALL_CHROMIUM=0 scripts/install-deps.sh
PYTHONPATH=src venv/bin/python -m ia_visao_web.cli dataset build --synthetic-only --count 2
```

Status: concluido.

### Passo 9: Relatorio de distribuicao

- [x] Adicionar `--report` em `dataset validate`.
- [x] Escrever `_qa/report.json` com contagens por classe/split.
- [x] Escrever buckets de area (`small`, `medium`, `large`).
- [x] Cobrir com teste unitario.

Status: concluido.

### Passo 10: Fechamento do PDR

- [x] Build real com Playwright funciona.
- [x] QA visual e relatorio sao gerados.
- [x] Gerador HTML cobre as 17 classes.
- [x] HTML gerado nao depende de CDN.
- [x] Testes, lint e typecheck passam.
- [x] README foi atualizado.
- [x] Status deste PDR foi atualizado.

Status: concluido.

## Riscos e Cuidados

- Bootstrap via CDN depende de rede. Para determinismo e uso offline, sera
  melhor vendorizar CSS ou usar um bundle local em etapa futura.
- `page.screenshot(full_page=True)` pode gerar imagem maior que viewport; as
  bboxes devem usar as mesmas dimensoes da imagem final.
- Elementos parcialmente fora da viewport precisam ser tratados com cuidado.
- `container` pode gerar muitas labels e poluir o dataset. Talvez precise de
  flag para desligar essa classe.
- Textos podem gerar muitas caixas pequenas; filtros de area e deduplicacao
  precisam ser revisados com amostras reais.
- A ordem dos seletores importa: `button` deve continuar antes de `link`.

## Proximos Passos Depois Deste PDR

1. Expandir o gerador HTML para cobrir todas as 17 classes da taxonomia.
2. Fortalecer `dataset validate` com presenca por split e QA visual.
3. Gerar dataset real pequeno para validacao manual.
4. Implementar treino com Torch/Ultralytics.
5. Implementar `predict` real.
6. Preparar documentacao open source e publicar pesos pre-treinados.
