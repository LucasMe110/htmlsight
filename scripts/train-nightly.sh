#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/lucas/workspace/htmlsight"
LOG_DIR="$PROJECT/logs"
LOG_FILE="$LOG_DIR/train-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"

exec >> "$LOG_FILE" 2>&1

echo "========================================"
echo "Treino iniciado: $(date)"
echo "========================================"

cd "$PROJECT"

if [ ! -d "data/dataset/images/train" ]; then
    echo "ERRO: dataset não encontrado em $PROJECT/data/dataset"
    echo "Execute primeiro: dataset build --count 3000 --workers 4 --output data/dataset"
    exit 1
fi

PYTHONPATH=src venv/bin/python -m ia_visao_web.cli train \
    --dataset data/dataset \
    --output runs/nightly-"$(date +%Y%m%d)" \
    --epochs 100 \
    --device cpu

echo ""
echo "========================================"
echo "Treino concluído: $(date)"
echo "========================================"

# Mantém apenas os últimos 7 logs
ls -t "$LOG_DIR"/train-*.log 2>/dev/null | tail -n +8 | xargs -r rm
