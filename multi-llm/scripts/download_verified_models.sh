#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Baixando modelos VERIFICADOS para arquitetura multi-LLM..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Verificando $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "✅ $model_name já existe"
        ls -lh "$output_path"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    echo "⬇️ Baixando $model_name..."
    if wget --header="Authorization: Bearer $HF_TOKEN" \
           --progress=bar:force:noscroll \
           -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            echo "✅ $model_name baixado!"
            ls -lh "$output_path"
            return 0
        fi
    fi
    
    echo "❌ Falha no download"
    rm -f "$output_path"
    return 1
}

# Fast Tier Models (já temos)
echo "=== FAST TIER (JÁ BAIXADOS) ==="
ls -lh models/llama/llama-3.2-3b-instruct-q4_k_m.gguf 2>/dev/null || echo "Llama 3.2 3B não encontrado"
ls -lh models/phi/phi-3.5-mini-instruct-q4_k_m.gguf 2>/dev/null || echo "Phi-3.5 Mini não encontrado"

# Balanced Tier Models - URLs VERIFICADAS
echo "=== BALANCED TIER ==="
ls -lh models/llama/llama-3.1-8b-instruct-q4_k_m.gguf 2>/dev/null || echo "Llama 3.1 8B não encontrado"

# Qwen2.5 7B - URL correta verificada
download_model \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "models/qwen/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "Qwen2.5 7B Multilang"

# Mistral 7B Instruct v0.3 - URL verificada
download_model \
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf" \
    "models/mistral/mistral-7b-instruct-v0.3-q4_k_m.gguf" \
    "Mistral 7B Code"

# Premium Tier - Code Llama como alternativa
echo "=== PREMIUM TIER ==="
download_model \
    "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf" \
    "models/llama/codellama-13b-instruct-q4_k_m.gguf" \
    "CodeLlama 13B Premium"

echo ""
echo "🎉 Verificação/Downloads concluídos!"
echo ""
echo "🔍 Verificando todos os modelos:"
find models/ -name "*.gguf" -exec du -h {} \; | sort -h
