#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Baixando modelos para arquitetura multi-LLM (URLs corrigidos)..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Baixando $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "✅ $model_name já existe"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    if wget --header="Authorization: Bearer $HF_TOKEN" \
           --progress=bar:force:noscroll \
           -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            echo "✅ $model_name baixado!"
            return 0
        fi
    fi
    
    echo "❌ Falha no download"
    rm -f "$output_path"
    return 1
}

# Fast Tier Models (URLs corrigidos)
echo "=== FAST TIER ==="
download_model \
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    "models/llama/llama-3.2-3b-instruct-q4_k_m.gguf" \
    "Llama 3.2 3B Fast"

download_model \
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
    "models/phi/phi-3-mini-4k-instruct-q4.gguf" \
    "Phi-3 Mini"

# Balanced Tier Models (URLs verificados)
echo "=== BALANCED TIER ==="
download_model \
    "https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf" \
    "models/llama/llama-3.1-8b-instruct-q4_k_m.gguf" \
    "Llama 3.1 8B"

download_model \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "models/qwen/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "Qwen2.5 7B Multilang"

download_model \
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf" \
    "models/mistral/mistral-7b-instruct-v0.3-q4_k_m.gguf" \
    "Mistral 7B Code"

# Premium Tier Model
echo "=== PREMIUM TIER ==="
download_model \
    "https://huggingface.co/QuantFactory/Meta-Llama-3.1-13B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-13B-Instruct.Q4_K_M.gguf" \
    "models/llama/llama-3.1-13b-instruct-q4_k_m.gguf" \
    "Llama 3.1 13B Premium"

echo ""
echo "🎉 Downloads concluídos!"
echo ""
echo "🔍 Verificando arquivos:"
find models/ -name "*.gguf" -exec du -h {} \; 2>/dev/null | sort -h || echo "Pasta models ainda sendo criada..."
