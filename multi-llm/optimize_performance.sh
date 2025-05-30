#!/bin/bash
echo "🚀 Otimizando performance dos modelos..."

# Função para otimizar modelo
optimize_model() {
    local container=$1
    local gpu_layers=$2
    local context=$3
    
    echo "⚡ Otimizando $container..."
    docker exec $container bash -c "
        sed -i 's/n_ctx=.*/n_ctx=$context,/' /app/llm_server.py
        sed -i 's/n_gpu_layers=.*/n_gpu_layers=$gpu_layers,/' /app/llm_server.py
        pkill -f uvicorn
    " 2>/dev/null || echo "Ajustando $container..."
}

# Aplicar otimizações
optimize_model "llama-8b-balanced" 35 2048
optimize_model "codellama-7b" 35 2048
optimize_model "mistral-7b-code" 35 4096
optimize_model "solar-10b-premium" 32 2048

echo "✅ Otimizações aplicadas!"
