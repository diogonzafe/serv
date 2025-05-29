#!/bin/bash

echo "🔍 Buscando URLs válidas..."

# Função para testar URL
test_url() {
    local url=$1
    local name=$2
    echo -n "Testing $name... "
    if curl -s --head "$url" | head -n 1 | grep -q "200 OK"; then
        echo "✅ VÁLIDA"
        echo "$url"
    else
        echo "❌ INVÁLIDA"
    fi
}

echo "=== TESTANDO URLs ==="

# Qwen2.5 possíveis URLs
test_url "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf" "Qwen2.5 Official"
test_url "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" "Qwen2.5 Bartowski"

# Mistral URLs
test_url "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" "Mistral TheBloke"
test_url "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" "Mistral Bartowski"

echo "=== RESULTADO ==="
