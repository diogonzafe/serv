#!/bin/bash
set -e

echo "=== Container LLM Individual ==="
echo "Modelo: ${MODEL_NAME:-Unknown}"
echo "Porta: ${SERVER_PORT:-8000}"
echo "GPU Layers: ${GPU_LAYERS:-32}"

if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU NVIDIA detectada"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ GPU não detectada"
fi

echo "🚀 Iniciando servidor LLM..."
exec python3 /app/llm_server.py
