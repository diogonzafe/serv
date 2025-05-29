#!/bin/bash

echo "🎛️ Multi-LLM Management Script"
echo "=============================="

case "$1" in
    "start")
        echo "🚀 Iniciando sistema multi-LLM..."
        docker-compose -f docker-compose.multi-llm.yml up -d
        echo "✅ Sistema iniciado!"
        echo "📊 Dashboard: http://localhost:8300"
        echo "🌐 Gateway: http://localhost:8200"
        ;;
    "stop")
        echo "🛑 Parando sistema multi-LLM..."
        docker-compose -f docker-compose.multi-llm.yml down
        ;;
    "restart")
        echo "🔄 Reiniciando sistema multi-LLM..."
        docker-compose -f docker-compose.multi-llm.yml restart
        ;;
    "status")
        echo "📊 Status dos containers:"
        docker-compose -f docker-compose.multi-llm.yml ps
        ;;
    "logs")
        if [ -n "$2" ]; then
            docker-compose -f docker-compose.multi-llm.yml logs -f "$2"
        else
            docker-compose -f docker-compose.multi-llm.yml logs
        fi
        ;;
    "health")
        echo "🏥 Verificando saúde dos modelos..."
        curl -s http://localhost:8200/health | python3 -m json.tool || echo "❌ Gateway offline"
        ;;
    "test")
        echo "🧪 Teste rápido..."
        curl -s -X POST http://localhost:8200/generate \
          -H "Content-Type: application/json" \
          -d '{"prompt": "Hello, how are you?", "max_tokens": 50}' \
          | python3 -m json.tool || echo "❌ Teste falhou"
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status|logs [service]|health|test}"
        echo ""
        echo "Comandos disponíveis:"
        echo "  start   - Inicia todos os containers"
        echo "  stop    - Para todos os containers"
        echo "  restart - Reinicia o sistema"
        echo "  status  - Mostra status dos containers"
        echo "  logs    - Mostra logs (opcional: especificar serviço)"
        echo "  health  - Verifica saúde do gateway"
        echo "  test    - Executa teste rápido"
        ;;
esac
