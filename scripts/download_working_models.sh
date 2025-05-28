#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Downloading WORKING models optimized for Portuguese + English..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Downloading $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "✅ $model_name já existe e não está vazio"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    # Tentar download com diferentes métodos
    if wget --header="Authorization: Bearer $HF_TOKEN" --progress=bar:force:noscroll -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            echo "✅ $model_name baixado com sucesso!"
            return 0
        fi
    fi
    
    echo "❌ Falha no download de $model_name"
    rm -f "$output_path"
    return 1
}

# Modelo 1: Llama 3.1 8B (melhor para português, confiável)
download_model \
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
    "models/llama/phi-3-mini-4k-instruct-q4.gguf" \
    "Phi-3 Mini 4K Instruct (Português + Inglês)"

# Modelo 2: Gemma 2B (rápido e eficiente)
download_model \
    "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf" \
    "models/llama/gemma-2b-it-q4_k_m.gguf" \
    "Gemma 2B Instruct (Rápido)"

# Modelo 3: TinyLlama para testes (muito rápido)
download_model \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf" \
    "models/llama/tinyllama-1.1b-chat-q4_k_m.gguf" \
    "TinyLlama 1.1B Chat (Teste Rápido)"

echo ""
echo "🎉 Downloads concluídos!"
echo ""
echo "🔍 Verificando arquivos..."
for model_file in \
    "models/llama/phi-3-mini-4k-instruct-q4.gguf" \
    "models/llama/gemma-2b-it-q4_k_m.gguf" \
    "models/llama/tinyllama-1.1b-chat-q4_k_m.gguf"; do
    
    if [ -f "$model_file" ] && [ -s "$model_file" ]; then
        size=$(du -h "$model_file" | cut -f1)
        echo "✅ $(basename "$model_file"): $size"
    else
        echo "❌ $(basename "$model_file"): Não encontrado ou vazio!"
    fi
done

echo ""
echo "🚀 Pronto para testar!"
