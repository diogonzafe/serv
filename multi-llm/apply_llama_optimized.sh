#!/bin/bash
set -e

echo "🚀 APLICANDO CONFIGURAÇÃO OTIMIZADA PARA LLAMA 8B"
echo "================================================"

# Criar servidor otimizado
cat > containers/llm_server_optimized.py << 'EOF'
import os
import time
import gc
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import psutil

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python importado com sucesso")
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="LLM Container Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

def get_system_info():
    try:
        process = psutil.Process()
        return {
            "memory_mb": f"{process.memory_info().rss / 1024 / 1024:.1f}",
            "cpu_percent": f"{process.cpu_percent():.1f}%"
        }
    except:
        return {"memory_mb": "0", "cpu_percent": "0%"}

def get_gpu_memory():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(', '))
            return {"used_mb": used, "free_mb": total - used, "total_mb": total}
    except:
        pass
    return {"used_mb": 0, "free_mb": 12288, "total_mb": 12288}

def load_model():
    global model_instance, model_config
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/default.gguf")
    model_name = os.environ.get("MODEL_NAME", "Default")
    gpu_layers = int(os.environ.get("GPU_LAYERS", 50))
    context_size = int(os.environ.get("CONTEXT_SIZE", 8192))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🔄 Carregando modelo: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers}")
    print(f"📚 Context: {context_size}")
    
    try:
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=16,
            verbose=False,
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            n_batch=2048,
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers
        }
        
        print(f"✅ Modelo {model_name} carregado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

@app.get("/")
def read_root():
    uptime = time.time() - startup_time
    return {
        "status": "running",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "system": get_system_info(),
        "gpu": get_gpu_memory(),
        "specialties": os.environ.get("MODEL_SPECIALTIES", "general").split(",")
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "gpu_memory": get_gpu_memory()
    }

@app.get("/info")
def model_info():
    return {
        "config": model_config,
        "capabilities": {
            "generate": model_instance is not None,
            "stream": model_instance is not None,
            "context_size": model_config.get("context_size", 0)
        },
        "performance": get_system_info()
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        start_time = time.time()
        
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 4096),
            temperature=request.temperature,
            echo=False,
            stream=False
        )
        
        generation_time = time.time() - start_time
        
        return {
            "text": response["choices"][0]["text"],
            "model": model_config.get("name", "Unknown"),
            "generation_time": f"{generation_time:.2f}s",
            "finish_reason": response["choices"][0]["finish_reason"],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response["choices"][0]["text"].split()),
                "total_tokens": len(request.prompt.split()) + len(response["choices"][0]["text"].split())
            },
            "system": get_system_info()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

print("🚀 Iniciando carregamento do modelo...")
if load_model():
    print("✅ Modelo carregado com sucesso!")
else:
    print("❌ Falha ao carregar modelo")

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
EOF

# Criar docker-compose
cat > docker-compose.llama-optimized.yml << 'EOF'
version: '3.8'

services:
  llama-8b-optimized:
    image: llm-multi-base-cuda:latest
    container_name: llama-8b-optimized
    restart: unless-stopped
    environment:
      - MODEL_NAME=Llama 3.1 8B MAXIMIZED
      - MODEL_PATH=/app/models/llama/llama-3.1-8b-instruct-q4_k_m.gguf
      - SERVER_PORT=8000
      - GPU_LAYERS=50
      - CONTEXT_SIZE=8192
      - MODEL_SPECIALTIES=general,complex-tasks,supervisor
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./models:/app/models:ro
      - ./containers/llm_server_optimized.py:/app/llm_server.py:ro
    gpus: all
    networks:
      - llm-network
    ports:
      - "8101:8000"

  smart-gateway-v2:
    image: llm-smart-gateway-v2:latest
    container_name: smart-gateway-v2
    restart: unless-stopped
    ports:
      - "8200:8200"
    environment:
      - GATEWAY_VERSION=2.1
      - LOG_LEVEL=INFO
    networks:
      - llm-network
    depends_on:
      - llama-8b-optimized

networks:
  llm-network:
    driver: bridge
EOF

# Parar containers antigos
echo "🛑 Parando containers antigos..."
docker-compose -f docker-compose.cuda.yml down 2>/dev/null || true
docker stop $(docker ps -a | grep -E "(llama|codellama|mistral|solar)" | awk '{print $1}') 2>/dev/null || true

# Iniciar novo sistema
echo "🚀 Iniciando sistema otimizado..."
docker-compose -f docker-compose.llama-optimized.yml up -d

echo "⏳ Aguardando 20 segundos para inicialização..."
sleep 20

echo "✅ SISTEMA OTIMIZADO INICIADO!"
echo "=============================="
echo "📊 Verificando status..."
curl -s http://localhost:8101/ | python3 -m json.tool | head -15 || echo "Ainda iniciando..."

echo -e "\n🌐 ACESSO:"
echo "- Modelo Direto: http://localhost:8101"
echo "- Gateway API: http://localhost:8200"

echo -e "\n📝 TESTE RÁPIDO:"
echo 'curl -X POST http://localhost:8200/generate -H "Content-Type: application/json" -d '"'"'{"prompt": "What is AI?", "max_tokens": 100}'"'"

echo -e "\n📊 MONITORAR GPU:"
echo "watch -n 1 nvidia-smi"
