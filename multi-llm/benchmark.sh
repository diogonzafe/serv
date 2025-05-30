#!/bin/bash
echo "⚡ Benchmark de Performance"
echo "=========================="

test_model() {
    local prompt="$1"
    local tokens="$2"
    
    echo -e "\n📝 Prompt: '$prompt' (max_tokens: $tokens)"
    
    # Teste direto no gateway
    start=$(date +%s.%N)
    response=$(curl -s -X POST http://192.168.15.31:8200/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_tokens\": $tokens}" 2>&1)
    end=$(date +%s.%N)
    
    total_time=$(echo "$end - $start" | bc)
    
    if echo "$response" | grep -q "gateway_info"; then
        model=$(echo "$response" | grep -o '"selected_model":"[^"]*"' | cut -d'"' -f4)
        gen_time=$(echo "$response" | grep -o '"generation_time":"[^"]*"' | cut -d'"' -f4)
        echo "✅ Modelo: $model"
        echo "⏱️  Gateway Total: ${total_time}s"
        echo "⚡ Geração: $gen_time"
    else
        echo "❌ Erro na resposta"
    fi
}

# Testes variados
test_model "Hi" 20
test_model "Write a Python hello world" 50
test_model "Explain Docker in one paragraph" 100
