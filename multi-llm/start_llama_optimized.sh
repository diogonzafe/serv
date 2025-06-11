#!/bin/bash
set -e

echo "🚀 INICIANDO LLAMA 8B MAXIMIZADO"
echo "================================"

# 1. Verificar se o modelo existe
if [ ! -f "models/llama/llama-3.1-8b-instruct-q4_k_m.gguf" ]; then
    echo "❌ ERRO: Modelo não encontrado em models/llama/llama-3.1-8b-instruct-q4_k_m.gguf"
    echo "Por favor, certifique-se de que o modelo está no local correto."
    exit 1
fi

# 2. Parar containers antigos
echo "🛑 Parando containers antigos..."
docker-compose -f docker-compose.llama-optimized.yml down 2>/dev/null || true

# 3. Iniciar novo sistema
echo "🚀 Iniciando sistema otimizado..."
docker-compose -f docker-compose.llama-optimized.yml up -d

# 4. Aguardar inicialização
echo "⏳ Aguardando sistema inicializar (20 segundos)..."
sleep 20

# 5. Modificar gateway para modelo único
echo "🔧 Configurando gateway para modelo único..."
./modify_gateway.sh 2>/dev/null || echo "Gateway será configurado manualmente"

# 6. Verificar status
echo -e "\n✅ SISTEMA INICIADO!"
echo "=================="
echo "📊 Status:"
curl -s http://localhost:8101/ | python3 -m json.tool | head -20 || echo "Modelo ainda iniciando..."
echo ""
echo "🌐 URLs:"
echo "- Modelo Direto: http://localhost:8101"
echo "- Gateway API: http://localhost:8200"
echo "- Dashboard: http://localhost:8300"
echo ""
echo "📝 Teste rápido:"
echo 'curl -X POST http://localhost:8200/generate -H "Content-Type: application/json" -d '"'"'{"prompt": "Hello, how are you?", "max_tokens": 50}'"'"
