# ia-visao-web — Design

**Data:** 2026-05-11
**Status:** Proposta (aguardando revisão)
**Escopo:** MVP de um detector visual de componentes web que aprende não só a aparência mas também a estrutura HTML correspondente.

## 1. Visão geral

Treinar uma rede de detecção de objetos pra reconhecer componentes de interface (`button`, `input`, `card`, `modal`, etc.) em screenshots de páginas web. O diferencial: o modelo é **multi-task** — além de prever classe e bounding box, prevê atributos HTML do componente (tag, `display`, `role`, tem filhos?). Isso força a representação visual a alinhar com a estrutura do HTML, fazendo a IA "entender" o que torna um botão um botão e não apenas decorar pixels.

O dataset é **gerado e auto-rotulado**: o sistema renderiza HTML real do Bootstrap 5 com Playwright, percorre o DOM aplicando seletores CSS canônicos, e extrai screenshot + bboxes + atributos sem anotação manual.

### Objetivo do MVP

Dado um screenshot de uma página web (sintética, baseada em Bootstrap), o sistema produz:

- Lista de detecções `{classe, bbox, atributos_html}` com score de confiança.
- Métricas mAP@50 e mAP@50-95 no test set, e acurácia por atributo HTML.

### Não-objetivos do MVP

- Reconhecer screenshots de sites reais arbitrários (gap de domínio síntético→real fica pra v2).
- Frontend / app web. O entregável é Python + CLI.
- Gerar código HTML a partir de imagem (image-to-code).
- Cobrir frameworks além de Bootstrap 5 (Tailwind, MUI etc. ficam pra evoluções).

## 2. Arquitetura

Seis unidades de responsabilidade única, comunicando por arquivos no disco (dataset) e pela interface do modelo.

```
┌──────────────── DADOS ────────────────┐    ┌──────────── MODELO ────────────┐

  sources/          ──►  renderer/   ──►  dataset/   ──►   model/   ──►  eval/
  (HTML Bootstrap)       (Playwright)     (YOLO+JSON)      (YOLOv8+      (mAP +
                              │                            multi-task)    attrs)
                              ▼
                          labeler/
                          (selector→class
                           + atributos)
```

| Unidade      | Responsabilidade                                                 | Depende de             |
|--------------|------------------------------------------------------------------|------------------------|
| `sources/`   | Coletar/gerar HTML do Bootstrap (docs + variações Jinja+Faker)   | —                      |
| `renderer/`  | Renderizar HTML em Chromium headless e devolver página viva      | Playwright             |
| `labeler/`   | Aplicar seletores → taxonomia, extrair bboxes e atributos        | DOM (via `renderer`)   |
| `dataset/`   | Serializar em formato YOLO + sidecar JSON de atributos           | saída do `labeler`     |
| `model/`     | YOLOv8 com heads multi-task; loss combinada; treino              | PyTorch, ultralytics   |
| `eval/`      | Inferência CLI, métricas mAP, acurácia de atributos              | `model`                |

Cada unidade vira um módulo Python independente, testável em isolamento. Trocar a fonte de HTML (Bootstrap → Tailwind) requer mexer só em `sources/` + mapa de seletores em `labeler/`.

## 3. Geração e rotulação de dataset

### 3.1 Fontes de HTML

- **`data/sources/bootstrap-docs/`** — snapshot da doc oficial do Bootstrap 5 (páginas de cada componente). Versão fixada (Bootstrap `5.3.x`). Baixar uma vez via script `sources/fetch_bootstrap_docs.py` e checar no git.
- **Gerador programático (`sources/generator.py`)** — combina blocos Bootstrap (navbar, hero, grid de cards, formulário, modal, alert, tabela) em páginas variadas. Slots preenchidos com:
  - Texto: `Faker(pt_BR)` e `Faker(en_US)`.
  - Cores: paleta randomizada (Bootstrap utility classes `bg-primary`, `bg-success`, etc.).
  - Quantidade: número de cards/linhas/colunas sorteado dentro de intervalos seguros.
  - Viewport: sorteado de `{(1280,720), (1920,1080), (768,1024), (375,667)}`.

Alvo de tamanho do dataset MVP: **3.000 imagens** (~80% gerador, ~20% docs).

### 3.2 Renderização (`renderer/`)

`playwright.sync_api`, Chromium headless, modo deterministic (seed em geração e nome dos arquivos).

Para cada HTML:

1. `page.set_viewport_size(...)` com viewport sorteado.
2. `page.set_content(html)` + esperar `networkidle`.
3. Aguardar `document.fonts.ready` + 200ms (fontes assentam).
4. `page.screenshot(full_page=True)` → PNG.
5. Passar `page` adiante para o `labeler`.

### 3.3 Rotulação (`labeler/`)

Mapa fixo `seletor CSS → classe da taxonomia`, definido em `labeler/selectors.py`.

**Taxonomia (17 classes):**

```
button, input, textarea, checkbox, radio, select, link,
card, navbar, tabs, modal, table, alert, accordion,
image, text, container
```

Observação: o "container" funciona como classe de "fundo estrutural"; pode ser desligado por flag se gerar muito ruído.

**Mapa de seletores (esboço, ajustável durante implementação):**

```python
SELECTORS = [
    ("button",    "button.btn, .btn, [role='button']"),
    ("input",     "input[type='text'], input[type='email'], input[type='password'], "
                  "input[type='search'], input[type='url'], input[type='tel'], input[type='number']"),
    ("textarea",  "textarea"),
    ("checkbox",  "input[type='checkbox']"),
    ("radio",     "input[type='radio']"),
    ("select",    "select, .form-select"),
    ("link",      "a:not(.btn):not(.nav-link):not(.dropdown-item)"),
    ("card",      ".card"),
    ("navbar",    ".navbar"),
    ("tabs",      ".nav-tabs, .nav-pills"),
    ("modal",     ".modal.show, .modal-dialog"),
    ("table",     "table"),
    ("alert",     ".alert"),
    ("accordion", ".accordion"),
    ("image",     "img, picture"),
    ("text",      "p, h1, h2, h3, h4, h5, h6, .lead"),
    ("container", ".container, .container-fluid, .row, .col, .card-body"),
]
```

Resolução de ambiguidade: ordem na lista é a prioridade. Um `a.btn` é classificado como `button` (primeira regra), nunca como `link`.

**Walker no DOM** (executado via `page.evaluate(JS)`):

Para cada par `(classe, seletor)`:

1. `document.querySelectorAll(seletor)` retorna todos os matches.
2. Pra cada match: capturar `getBoundingClientRect()` → `(x, y, w, h)`.
3. Capturar atributos HTML:
   - `tag` — `element.tagName.toLowerCase()`
   - `display` — `getComputedStyle(el).display`
   - `role` — `el.getAttribute('role')` ou null
   - `has_children` — `el.children.length > 0`
   - `n_descendants` — `el.querySelectorAll('*').length` (truncado em 50)
4. Filtros:
   - Descartar se `w*h < 16` px.
   - Descartar se totalmente fora do viewport.
   - Descartar se `getComputedStyle(el).visibility === 'hidden'` ou `display === 'none'`.
   - Descartar duplicatas (IoU > 0.95 com outro match da mesma classe).

### 3.4 Formato de saída (`dataset/`)

Compatível com Ultralytics YOLO + sidecar JSON.

```
data/dataset/
  images/
    train/{id}.png
    val/{id}.png
    test/{id}.png
  labels/
    train/{id}.txt      # YOLO: "class_id cx cy w h" (normalizado 0-1)
    val/{id}.txt
    test/{id}.txt
  attrs/
    train/{id}.json     # [{"tag":"button","display":"inline-block","role":null,...}, ...]
    val/{id}.json
    test/{id}.json
  data.yaml             # config do YOLO (caminhos, nc, names)
```

Cada linha em `labels/{id}.txt` tem um sidecar correspondente em `attrs/{id}.json` na mesma ordem.

Split 80/10/10 determinístico por hash do nome (`hashlib.sha1(filename).hexdigest()` módulo 10).

### 3.5 Validação do dataset (porta antes do treino)

CLI `dataset validate` checa:

- Mínimo 200 instâncias por classe no train.
- Cada classe presente em val e test.
- Distribuição de área (histograma logarítmico) sem buracos.
- Amostragem aleatória de 50 imagens com bboxes desenhados em `data/dataset/_qa/` pra inspeção visual.

## 4. Modelo multi-task

### 4.1 Backbone e head de detecção

`YOLOv8s` (versão pequena, ~11M params) da Ultralytics, pré-treinado em COCO. Backbone CSP + neck PAN-FPN intactos.

### 4.2 Heads adicionais (multi-task)

Fork local do `ultralytics.nn.modules.head.Detect`, adicionando, em paralelo às saídas de classe e bbox, **uma head por atributo HTML categórico ou booleano**:

| Head            | Tipo            | Saídas                                                                |
|-----------------|-----------------|------------------------------------------------------------------------|
| `cls`           | classificação   | 17 (classes da taxonomia)                                              |
| `bbox`          | regressão DFL   | 4 × `reg_max` (padrão YOLOv8)                                          |
| `attr_tag`      | classificação   | vocab fechado das tags HTML observadas no dataset (~20: div, button, input, a, p, h1..h6, table, ul, li, ...) |
| `attr_display`  | classificação   | `{block, inline, inline-block, flex, grid, none, other}`               |
| `attr_role`     | classificação   | vocab fechado (`button, link, alert, navigation, tab, dialog, none, other`) |
| `attr_haschld`  | binária         | `{0, 1}`                                                               |

Vocabulários de `tag` e `role` são levantados na primeira passagem pelo dataset e fixados em `model/vocab.json`. Tudo fora do vocab vira `other`.

Cada head é uma `Conv` 1×1 paralela operando sobre o mesmo feature map que `cls`/`bbox`, com mesma resolução (3 escalas).

### 4.3 Função de loss

```
L_total = λ_cls · L_cls
        + λ_box · (L_iou + L_dfl)
        + λ_tag · L_tag
        + λ_disp · L_display
        + λ_role · L_role
        + λ_hc · L_haschld
```

- `L_cls`, `L_iou`, `L_dfl` — losses padrão do YOLOv8 (BCE, CIoU, DFL).
- `L_tag`, `L_display`, `L_role` — cross-entropy ponderada por frequência inversa de classe.
- `L_haschld` — BCE.
- Pesos iniciais: `λ_cls=0.5, λ_box=7.5, λ_tag=0.2, λ_disp=0.2, λ_role=0.2, λ_hc=0.1` (ajustar com search se desbalancear).

Atributos só contribuem pra loss em positives (bboxes com IoU > 0.5 com um GT) — assignment idêntico ao do head `cls`.

### 4.4 Pipeline de treino

`model/train.py` orquestra. Usa `ultralytics.YOLO(...)` por baixo, mas com modelo customizado e `DataLoader` customizado que carrega o sidecar `attrs/{id}.json` junto com as labels YOLO.

Hiperparâmetros iniciais (ajustáveis):

- Otimizador: SGD, lr=0.01, momentum=0.937, weight_decay=5e-4.
- Scheduler: cosine, 100 épocas, warmup 3 épocas.
- Batch size: 16 (ajusta pra GPU disponível).
- Augmentação: defaults do YOLOv8 (mosaic, mixup, hsv) — sem flip horizontal (texto vira espelhado e quebra a noção de "esquerda/direita" em UI).
- Resolução de input: 640×640.

## 5. Avaliação e inferência

### 5.1 Métricas

`eval/metrics.py` calcula:

- **Detecção:** mAP@50, mAP@50-95, precision/recall por classe (via `pycocotools`).
- **Atributos:** acurácia top-1 de cada head, em positives. Matriz de confusão por atributo (export PNG).
- **Tabela combinada por classe:** `classe | AP@50 | AP@50-95 | acc_tag | acc_display | acc_role | acc_haschld`.

### 5.2 CLI

```
$ ia-visao-web dataset build       # gera dataset
$ ia-visao-web dataset validate    # checa qualidade
$ ia-visao-web train               # treina o modelo
$ ia-visao-web eval --split test   # roda métricas
$ ia-visao-web predict <image.png> # inferência numa imagem, devolve JSON
```

`predict` devolve:

```json
{
  "image": "page.png",
  "detections": [
    {
      "class": "button",
      "score": 0.92,
      "bbox": [x, y, w, h],
      "attrs": {"tag": "button", "display": "inline-block", "role": null, "has_children": false}
    }
  ]
}
```

Opcional: flag `--overlay` salva PNG com bboxes desenhados.

## 6. Tratamento de erros e casos limite

| Risco                                                          | Mitigação                                                                                   |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Componentes invisíveis ou colapsados rotulados                  | Filtro de `display:none`, `visibility:hidden`, área < 16px no `labeler`.                    |
| Hierarquia: botão dentro de card → ambos rotulados              | Aceito por design (detector multi-instância suporta). `container` sai por flag se ruidoso.   |
| Duplicatas (mesmo elemento casa em 2 seletores)                 | Ordem do mapa = prioridade; pós-filtro de IoU > 0.95 entre matches da mesma classe.         |
| Desbalanceamento de classes (botão é abundante, accordion raro) | Pesos por frequência inversa nas losses + verificação no `dataset validate`.                |
| Texto longo gerado pelo Faker estourando o layout               | Faker com `max_chars` por slot; renderização em viewport mínimo 375px garante reflow OK.     |
| Tempo de Playwright dominando geração                           | Paralelismo com pool de processos (1 contexto Chromium por worker).                         |
| Modelo aprende viés do gerador (sempre mesmo cabeçalho)         | Variação forte de viewport, paleta, ordem dos blocos, presença/ausência de cada bloco.       |
| Mismatch entre vocabulário de atributos do train vs inferência  | Vocab fixado em `model/vocab.json` após primeiro build; `predict` mapeia desconhecido → `other`. |

## 7. Estratégia de testes

TDD em todos os módulos não-triviais.

**Testes unitários:**

- `sources/generator.py` — dado seed, página gerada é determinística; cada componente esperado aparece.
- `labeler/selectors.py` — HTML fixture com componentes conhecidos produz a lista esperada de matches.
- `labeler/walker.py` — descartar invisíveis, descartar pequenos, deduplicar por IoU.
- `dataset/writer.py` — formato YOLO correto, sidecar JSON alinhado com labels.
- `model/heads.py` — forward com input dummy retorna shapes esperados.
- `model/loss.py` — loss combinada > 0 com predições aleatórias; gradiente flui pelas heads novas.
- `eval/metrics.py` — mAP em fixtures sintéticas bate com valor calculado à mão.

**Testes de integração:**

- Pipeline end-to-end com fixture de 5 páginas: gera → renderiza → rotula → dataset → treina 2 épocas → avalia. Roda em CI (CPU), só checa que tudo executa sem erro e que mAP > 0 (sanity).

**Verificação visual:**

- `dataset validate` exporta 50 imagens com bboxes overlay em `data/dataset/_qa/`. Revisão manual antes de treinar de verdade.

## 8. Stack e dependências

- Python 3.11+
- `playwright` (Chromium headless)
- `beautifulsoup4` + `lxml` (parsing local quando necessário)
- `Jinja2` + `Faker` (geração)
- `torch` + `torchvision` (CUDA opcional)
- `ultralytics` (YOLOv8) — forkado/vendored pra modificar heads
- `pycocotools` (mAP)
- `pytest` + `pytest-mock`
- `click` ou `typer` (CLI)

Gerenciamento de deps: `uv` (preferência atual do ecossistema Python).

## 9. Estrutura de pastas

```
ia-visao-web/
├── docs/
│   └── superpowers/specs/2026-05-11-ia-visao-web-design.md
├── data/
│   ├── sources/bootstrap-docs/        # snapshot fixado
│   └── dataset/                       # gerado, gitignore
├── src/ia_visao_web/
│   ├── sources/
│   ├── renderer/
│   ├── labeler/
│   ├── dataset/
│   ├── model/
│   ├── eval/
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── pyproject.toml
├── README.md
└── CLAUDE.md                          # instruções do projeto
```

## 10. Roadmap pós-MVP (fora deste spec)

- Domínio síntese→real: treinar com mix de Bootstrap sintético + páginas reais raspadas com auto-rotulação por seletor.
- Suporte a Tailwind/MUI/shadcn-ui (módulos `sources/` adicionais + mapas de seletor).
- Trocar backbone YOLOv8 → RT-DETR pra layouts mais complexos.
- Pre-training contrastivo CLIP-style (imagem ↔ HTML) antes do fine-tune de detecção.
- Saída estruturada: árvore DOM reconstruída a partir das detecções.

## 11. Critérios de aceite do MVP

- [ ] CLI `dataset build` gera ≥ 3000 imagens com ≥ 200 instâncias por classe.
- [ ] `dataset validate` passa em todos os critérios.
- [ ] Treino completo (100 épocas) executa sem OOM em GPU de 8GB.
- [ ] No test set: mAP@50 ≥ 0.7 para pelo menos 12 das 17 classes.
- [ ] Acurácia média dos atributos HTML ≥ 0.8.
- [ ] `predict` numa imagem nova retorna JSON conforme spec em ≤ 2s (CPU) / ≤ 500ms (GPU).
- [ ] Testes unitários cobrem ≥ 80% das linhas; integração ponta-a-ponta passa em CI.
