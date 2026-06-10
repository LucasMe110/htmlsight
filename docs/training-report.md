# Relatório de Treinamento — HTMLSight YOLOv8s

**Data:** 2026-06-07 a 2026-06-09  
**Modelo base:** YOLOv8s (pretrained COCO)  
**Tarefa:** Detecção de componentes Bootstrap em screenshots de páginas web

---

## Hardware

| Componente | Especificação |
|---|---|
| CPU | AMD Ryzen 5 3400G — 4 núcleos / 8 threads @ 3.7 GHz |
| RAM | 16 GB (2× 8 GB) |
| GPU | AMD Radeon RX Vega 11 **integrada** — 2 GB VRAM compartilhada |
| Armazenamento | WD 1 TB HDD |
| Backend GPU | DirectML (torch-directml) — sem CUDA |

---

## Dataset

| Split | Imagens |
|---|---|
| Treino | 2.453 |
| Validação | 275 |
| Teste | 272 |
| **Total** | **3.000** |

**17 classes:** button, input, textarea, checkbox, radio, select, link, card, navbar, tabs, modal, table, alert, accordion, image, text, container

Dataset gerado sinteticamente via Playwright renderizando componentes Bootstrap 5.3 com labels determinísticos de elementos DOM.

---

## Configuração do Treinamento

| Parâmetro | Valor |
|---|---|
| Modelo | yolov8s.pt |
| Épocas | 100 |
| Batch size | 8 |
| Imagem | 640×640 |
| Otimizador | SGD |
| Learning rate inicial | 0.01 |
| LR final | 0.01 (cosine decay) |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| AMP | Desativado (não suportado no DirectML) |
| Workers | 0 |
| Device | `privateuseone:0` (DirectML) |

---

## Patches DirectML Aplicados

O Ultralytics não tem suporte oficial a DirectML. Os seguintes patches foram necessários para executar o treinamento na Vega 11:

| Patch | Motivo |
|---|---|
| `select_device` em todos os módulos | Ultralytics importa a função diretamente nos namespaces locais; patch no módulo canônico não é suficiente |
| `v8DetectionLoss.preprocess` → CPU | `torch.unique(return_counts=True)` não suportado no DirectML |
| `TaskAlignedAssigner._forward` → CPU | `scatter_add_` com dimensões parciais não suportado no DirectML |
| `BaseTrainer._get_memory` → retorna 0 | `torch.cuda.get_device_properties` falha sem CUDA |
| `BaseTrainer.validate` → skip | BatchNorm em `inference_mode` e NMS warmup falham no DirectML; validação é pulada durante o treino |
| `BaseTrainer.final_eval` → skip | Mesmo motivo acima |
| `check_amp` → False para DirectML | AMP não suportado |

> **Nota:** Com esses patches, a validação durante o treino é completamente desabilitada. Para avaliar o modelo, use o comando `cli eval` separadamente (roda no CPU).

---

## Curva de Loss (Treino)

| Época | box_loss | cls_loss | dfl_loss |
|---|---|---|---|
| 1 | 1.8997 | 4.3779 | 1.3370 |
| 5 | 1.7769 | 4.1599 | 1.1012 |
| 10 | 1.7557 | 4.1921 | 1.0906 |
| 20 | 1.7491 | 4.2424 | 1.0796 |
| 30 | 1.7020 | 4.2294 | 1.0694 |
| 50 | 1.7237 | 4.2988 | 1.0619 |
| 75 | 1.7153 | 4.2815 | 1.0537 |
| 100 | 1.7395 | 4.2457 | 1.0452 |

**Duração total:** ~38.3 horas (~23 min/época)

A curva de loss mostra convergência lenta porém estável. A `cls_loss` permaneceu elevada (~4.2), sugerindo dificuldade em distinguir as 17 classes com o dataset atual.

---

## Resultados da Avaliação (272 imagens de teste, CPU)

### Dataset completo — 100 épocas

| Métrica | Valor |
|---|---|
| **mAP50** | **0.021** |
| **mAP50-95** | **0.010** |

| Classe | mAP50 | mAP50-95 |
|---|---|---|
| card | 0.157 | 0.051 |
| accordion | 0.084 | 0.044 |
| container | 0.069 | 0.054 |
| modal | 0.018 | 0.010 |
| table | 0.012 | 0.006 |
| image | 0.008 | 0.002 |
| tabs | 0.001 | 0.000 |
| button | 0.002 | 0.000 |
| input, textarea, checkbox, radio, select, link, navbar, text | ~0.000 | ~0.000 |

**Acurácia de atributos HTML:**

| Atributo | Acurácia |
|---|---|
| role | 0.78 |
| tag | 0.00 |
| display | 0.00 |
| has_children | 0.00 |

### Comparativo: smoke (42 imgs) vs dataset completo (2.453 imgs)

| Treino | Épocas | mAP50 | mAP50-95 |
|---|---|---|---|
| smoke dataset | 50 | **0.308** | **0.186** |
| dataset completo | 100 | 0.021 | 0.010 |

---

## Análise e Diagnóstico

### Por que o dataset completo teve resultado pior?

1. **Ausência de `best.pt`:** A validação foi desabilitada durante o treino (limitação do DirectML), portanto o Ultralytics não pôde salvar o checkpoint com melhor mAP. O `last.pt` da época 100 pode não ser o ponto ótimo.

2. **Loss cls elevada e estagnada:** A `cls_loss` ficou em ~4.2 durante todo o treino (partindo de 4.4), indicando que o modelo não aprendeu a discriminar bem as 17 classes. Isso pode ser causado por:
   - Imbalance de classes no dataset sintético
   - Operações de fallback para CPU no DirectML introduzindo ruído numérico ao longo de 38h
   - Necessidade de mais épocas ou learning rate diferente

3. **Overfitting ao dataset smoke:** O dataset smoke tem apenas 42 imagens com distribuição muito uniforme — fácil de memorizar. No dataset real com 2.453 imagens e maior diversidade, o modelo precisa generalizar mais.

4. **Hardware limitante:** A Vega 11 com 2 GB VRAM compartilhada força batch=8 e múltiplos fallbacks para CPU, tornando o gradiente ruidoso.

---

## Recomendações

1. **Treinar no Google Colab** (GPU T4 gratuita): o notebook `htmlsight_colab.ipynb` já está configurado. As 100 épocas levariam ~1-2h com CUDA, validação funcionando e `best.pt` sendo salvo.

2. **Aumentar o dataset:** 3.000 imagens para 17 classes (~176/classe) é borderline. Recomendado ≥500 imagens por classe para YOLOv8s.

3. **Ajustar hiperparâmetros após Colab:** com `best.pt` e curva de validação visível, é possível identificar quando o modelo converge e ajustar `patience` e `lr`.

4. **Balancear classes:** classes como `button` e `input` têm baixo mAP — verificar distribuição e aumentar exemplos se necessário.

---

## Artefatos

| Arquivo | Descrição |
|---|---|
| `runs/detect/runs/full_train/weights/last.pt` | Pesos finais (época 100), 64 MB |
| `runs/detect/runs/full_train/results.csv` | Curva de loss por época |
| `runs/detect/runs/full_train/results.png` | Gráfico de loss |
| `runs/detect/runs/full_train/args.yaml` | Configuração completa do treino |
| `src/ia_visao_web/model/train.py` | Pipeline com patches DirectML |
| `htmlsight_colab.ipynb` | Notebook para treino no Google Colab |
