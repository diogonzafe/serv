#!/bin/bash
set -e

HF_TOKEN="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"

echo "🚀 Baixando modelos TESTADOS E VERIFICADOS..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Baixando $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "  ✅ Já existe, pulando..."
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    echo "  ⬇️ URL: $url"
    if wget --header="Authorization: Bearer $HF_TOKEN" \
           --progress=bar:force:noscroll \
           -O "$output_path" "$url"; then
        if [ -s "$output_path" ]; then
            size=$(du -h "$output_path" | cut -f1)
            echo "  ✅ Sucesso! ($size)"
            
            # Verificar se é GGUF válido
            if head -c 4 "$output_path" | grep -q "GGUF"; then
                echo "  ✅ Formato GGUF válido"
                return 0
            else
                echo "  ❌ Formato GGUF inválido"
                rm -f "$output_path"
                return 1
            fi
        fi
    fi
    
    echo "  ❌ Falha no download"
    rm -f "$output_path"
    return 1
}

echo "=== MODELOS VERIFICADOS FUNCIONANDO ==="

# 1. TinyLlama (modelo pequeno e confiável)
download_model \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf" \
    "models/llama/tinyllama-1.1b-q4_k_m.gguf" \
    "TinyLlama 1.1B (Ultra Rápido)"

# 2. Phi-3 Mini (URL correta verificada)
download_model \
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
    "models/phi/phi-3-mini-working.gguf" \
    "Phi-3 Mini (Funcionando)"

# 3. Gemma 2B (URL oficial verificada)  
download_model \
    "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf" \
    "models/gemma/gemma-2b-working.gguf" \
    "Gemma 2B (Funcionando)"

# 4. Llama 3.2 1B (modelo menor, mais confiável)
download_model \
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "models/llama/llama-3.2-1b-q4_k_m.gguf" \
    "Llama 3.2 1B (Pequeno e Rápido)"

echo ""
echo "🎉 Downloads concluídos!"
echo ""
echo "🔍 Verificando downloads:"
find models/ -name "*.gguf" -newer models_backup 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "✅ $(basename "$file"): $size"
done
