#!/bin/bash

echo "🖥️  GPU LLM Server Monitor"
echo "========================="

echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

echo -e "\n🐳 Container Status:"
docker-compose ps

echo -e "\n🏥 Server Health:"
curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool || echo "Server not responding"

echo -e "\n📈 GPU Stats:"
curl -s http://localhost:8001/gpu_stats 2>/dev/null | python3 -m json.tool || echo "GPU stats not available"
