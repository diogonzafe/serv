#!/bin/bash
set -e

# Seu token do Hugging Face
HF_TOKEN="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"

echo "🚀 Downloading verified models for RTX 3060..."

download_model() {
    local url=$1
    local output_path=$2
    local model_name=$3
    
    echo "📥 Downloading $model_name..."
    
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        echo "✅ $model_name already exists and is not empty"
        return 0
    fi
    
    mkdir -p "$(dirname "$output_path")"
    
    # Download with Hugging Face token authentication
    wget --progress=bar:force:noscroll \
         --header="Authorization: Bearer $HF_TOKEN" \
         -O "$output_path" "$url" || {
        echo "❌ Failed to download $model_name"
        rm -f "$output_path"
        return 1
    }
    
    # Verify file is not empty
    if [ -s "$output_path" ]; then
        echo "✅ $model_name downloaded successfully!"
    else
        echo "❌ $model_name downloaded but is empty!"
        rm -f "$output_path"
        return 1
    fi
}

# Model 1: Llama 3.2 3B (teste rápido) - ~2GB
echo "=== Downloading test model first ==="
download_model \
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    "models/llama/llama-3.2-3b-instruct-q4_k_m.gguf" \
    "Llama 3.2 3B Instruct (Test Model)"

# Model 2: Llama 3.1 8B (principal) - ~5.5GB
echo "=== Downloading main model ==="
download_model \
    "https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
    "models/llama/llama-3.1-8b-instruct-q4_k_m.gguf" \
    "Llama 3.1 8B Instruct"

# Model 3: Mistral 7B - ~4.5GB
echo "=== Downloading Mistral model ==="
download_model \
    "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
    "models/mistral/mistral-7b-instruct-q4_k_m.gguf" \
    "Mistral 7B Instruct"

echo ""
echo "🎉 Downloads completed!"
echo ""
echo "📊 Model Summary:"
echo "├── Llama 3.2 3B (Test): ~2GB VRAM, fast for testing"
echo "├── Llama 3.1 8B: ~5.5GB VRAM, excellent for most tasks"
echo "└── Mistral 7B: ~4.5GB VRAM, great for coding and chat"
echo ""
echo "💡 All models fit your RTX 3060 12GB perfectly!"

# Verify downloads
echo ""
echo "🔍 Verifying downloads..."
for model_file in \
    "models/llama/llama-3.2-3b-instruct-q4_k_m.gguf" \
    "models/llama/llama-3.1-8b-instruct-q4_k_m.gguf" \
    "models/mistral/mistral-7b-instruct-q4_k_m.gguf"; do
    
    if [ -f "$model_file" ] && [ -s "$model_file" ]; then
        size=$(du -h "$model_file" | cut -f1)
        echo "✅ $(basename "$model_file"): $size"
    else
        echo "❌ $(basename "$model_file"): Not found or empty!"
    fi
done

echo ""
echo "🚀 Ready to start the GPU server!"
echo "   Test model: llama-3.2-3b (fastest)"
echo "   Main model: llama-3.1-8b (recommended)"
