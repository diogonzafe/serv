#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Downloading LLAMA models (100% compatible)..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Downloading $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "✅ $model_name já existe"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    if wget --header="Authorization: Bearer $HF_TOKEN" --progress=bar:force:noscroll -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            echo "✅ $model_name baixado!"
            return 0
        fi
    fi
    
    echo "❌ Falha no download"
    rm -f "$output_path"
    return 1
}

# Modelo 1: Llama 2 7B Chat (mais compatível)
download_model \
    "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" \
    "models/llama/llama-2-7b-chat-q4_k_m.gguf" \
    "Llama 2 7B Chat"

# Modelo 2: Code Llama 7B (português + código)
download_model \
    "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf" \
    "models/llama/codellama-7b-instruct-q4_k_m.gguf" \
    "Code Llama 7B Instruct"

# Modelo 3: Llama 2 13B (melhor qualidade, se couber)
download_model \
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q3_K_M.gguf" \
    "models/llama/llama-2-13b-chat-q3_k_m.gguf" \
    "Llama 2 13B Chat"

echo ""
echo "🎉 Downloads concluídos!"
echo ""
echo "🔍 Verificando arquivos:"
for model_file in \
    "models/llama/llama-2-7b-chat-q4_k_m.gguf" \
    "models/llama/codellama-7b-instruct-q4_k_m.gguf" \
    "models/llama/llama-2-13b-chat-q3_k_m.gguf"; do
    
    if [ -f "$model_file" ] && [ -s "$model_file" ]; then
        size=$(du -h "$model_file" | cut -f1)
        echo "✅ $(basename "$model_file"): $size"
    else
        echo "❌ $(basename "$model_file"): Não encontrado!"
    fi
done
