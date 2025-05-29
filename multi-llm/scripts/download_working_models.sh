#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Baixando modelos CONFIRMADOS..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Verificando $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        size=$(du -h "$output_path" | cut -f1)
        echo "✅ $model_name já existe ($size)"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    echo "⬇️ Baixando $model_name..."
    if wget --header="Authorization: Bearer $HF_TOKEN" \
           --progress=bar:force:noscroll \
           -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            size=$(du -h "$output_path" | cut -f1)
            echo "✅ $model_name baixado! ($size)"
            return 0
        fi
    fi
    
    echo "❌ Falha no download"
    rm -f "$output_path"
    return 1
}

echo "=== MODELOS JÁ BAIXADOS ==="
find models/ -name "*.gguf" -exec ls -lh {} \; 2>/dev/null || true

echo ""
echo "=== BAIXANDO MODELOS ADICIONAIS ==="

# Mistral 7B (URL correta verificada)
download_model \
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
    "models/mistral/mistral-7b-instruct-q4_k_m.gguf" \
    "Mistral 7B Instruct"

# CodeLlama para código (já confirmei que existe)
download_model \
    "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf" \
    "models/llama/codellama-7b-instruct-q4_k_m.gguf" \
    "CodeLlama 7B (Código)"

# Gemma 2B (modelo pequeno e rápido)
download_model \
    "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf" \
    "models/gemma/gemma-2b-it-q4_k_m.gguf" \
    "Gemma 2B (Ultra Rápido)"

# Solar 10.7B (modelo intermediário potente)
download_model \
    "https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF/resolve/main/solar-10.7b-instruct-v1.0.Q4_K_M.gguf" \
    "models/solar/solar-10.7b-instruct-q4_k_m.gguf" \
    "Solar 10.7B (Premium)"

echo ""
echo "🎉 Downloads concluídos!"
echo ""
echo "📊 Inventário final:"
find models/ -name "*.gguf" -exec du -h {} \; | sort -h
