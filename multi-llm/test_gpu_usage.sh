#!/bin/bash
echo "🔍 Testando uso de GPU pelos modelos..."
echo "=================================="

# Monitorar GPU antes do teste
echo "📊 GPU antes do teste:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Fazer requisição para cada modelo
for model in llama-8b-balanced codellama-7b mistral-7b-code solar-10b-premium; do
    echo -e "\n🤖 Testando $model..."
    
    # Iniciar monitoramento em background
    (while true; do 
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        sleep 0.5
    done) > gpu_${model}.log &
    MONITOR_PID=$!
    
    # Fazer requisição
    time curl -s -X POST http://192.168.15.31:8200/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Explain quantum computing in 3 sentences", "max_tokens": 100}' \
        > response_${model}.json
    
    # Parar monitoramento
    kill $MONITOR_PID 2>/dev/null
    
    # Mostrar pico de uso
    echo "📈 Pico de uso GPU:"
    sort -t',' -k1 -n gpu_${model}.log | tail -1
    
    # Limpar
    rm -f gpu_${model}.log
done
