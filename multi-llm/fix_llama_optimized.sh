#!/bin/bash
set -e

echo "🔧 APLICANDO CORREÇÃO - LLAMA 8B OTIMIZADO"
echo "=========================================="

# 1. Parar containers antigos
echo "🛑 Parando containers..."
docker-compose -f docker-compose.cuda.yml down 2>/dev/null || true
docker-compose -f docker-compose.llama-maxed.yml down 2>/dev/null || true

# 2. Criar servidor otimizado (versão corrigida)
echo "📝 Criando servidor otimizado..."
cat > containers/llm_server_maxed.py << 'PYEOF'
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

app = FastAPI(title="Llama 8B Maximized Server")
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
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                              '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, total, util = map(int, result.stdout.strip().split(', '))
            return {
                "used_mb": used, 
                "free_mb": total - used, 
                "total_mb": total,
                "utilization": f"{util}%"
            }
    except:
        pass
    return {"used_mb": 0, "free_mb": 12288, "total_mb": 12288, "utilization": "0%"}

def load_model():
    global model_instance, model_config
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/llama/llama-3.1-8b-instruct-q4_k_m.gguf")
    model_name = "Llama 3.1 8B MAXIMIZED"
    gpu_layers = int(os.environ.get("GPU_LAYERS", 45))
    context_size = int(os.environ.get("CONTEXT_SIZE", 8192))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🚀 CARREGANDO MODELO MAXIMIZADO: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (MÁXIMO)")
    print(f"📚 Context: {context_size} (OTIMIZADO)")
    print(f"🔥 RTX 3060 12GB - DEDICADA 100%")
    
    try:
        # Parâmetros otimizados para máxima performance
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=16,
            n_threads_batch=16,
            verbose=False,
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            n_batch=2048,
            rope_scaling_type=-1,
            offload_kqv=True,
            mul_mat_q=True,
            low_vram=False,
            logits_all=False,
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "optimization": "MAXIMUM_PERFORMANCE",
            "dedicated_gpu": "RTX 3060 12GB"
        }
        
        # Limpar memória
        gc.collect()
        
        print(f"✅ MODELO CARREGADO COM SUCESSO!")
        print(f"🔥 GPU: {get_gpu_memory()}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 LLAMA 8B MAXIMUM PERFORMANCE SERVER")
    print("=" * 60)
    success = load_model()
    if not success:
        print("❌ Falha ao carregar modelo na inicialização")

@app.get("/")
def read_root():
    uptime = time.time() - startup_time
    gpu_info = get_gpu_memory()
    return {
        "status": "MAXIMUM_PERFORMANCE",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "system": get_system_info(),
        "gpu": gpu_info,
        "optimization": "RTX 3060 12GB DEDICATED",
        "context_size": model_config.get("context_size", 0),
        "gpu_layers": model_config.get("gpu_layers", 0),
        "specialties": ["general", "complex-tasks", "reasoning", "code"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "available_models": 1 if model_instance else 0,
        "models": ["llama-8b-maximized"] if model_instance else [],
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
            stream=False,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            tfs_z=1.0,
            typical_p=1.0,
            mirostat_mode=0,
            penalize_nl=False,
        )
        
        generation_time = time.time() - start_time
        gpu_info = get_gpu_memory()
        
        return {
            "text": response["choices"][0]["text"],
            "model": model_config.get("name", "Unknown"),
            "generation_time": f"{generation_time:.2f}s",
            "tokens_per_second": f"{len(response['choices'][0]['text'].split()) / generation_time:.1f}",
            "finish_reason": response["choices"][0]["finish_reason"],
            "usage": response.get("usage", {}),
            "system": get_system_info(),
            "gpu": gpu_info,
            "gateway_info": {
                "selected_model": "llama-8b-maximized",
                "model_tier": "premium",
                "selection_reason": "single_model_optimized"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
PYEOF

# 3. Criar docker-compose usando imagem existente
echo "📝 Criando docker-compose corrigido..."
cat > docker-compose.llama-fixed.yml << 'DCEOF'
version: '3.8'

services:
  llama-8b-maxed:
    # USAR A IMAGEM QUE JÁ EXISTE E FUNCIONA
    image: llm-multi-base-cuda:latest
    container_name: llama-8b-maxed
    restart: unless-stopped
    environment:
      - MODEL_NAME=Llama 3.1 8B MAXIMIZED
      - MODEL_PATH=/app/models/llama/llama-3.1-8b-instruct-q4_k_m.gguf
      - SERVER_PORT=8000
      - GPU_LAYERS=45              # MÁXIMO para RTX 3060
      - CONTEXT_SIZE=8192          # CONTEXTO GRANDE
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - OMP_NUM_THREADS=16
      - CUDA_LAUNCH_BLOCKING=0
      - MODEL_SPECIALTIES=general,complex-tasks,reasoning,code
    volumes:
      - ./models:/app/models:ro
      - ./containers/llm_server_maxed.py:/app/llm_server.py:ro
    ports:
      - "8101:8000"  # Porta direta do modelo
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # Para docker-compose local
    runtime: nvidia
    networks:
      - llm-network

  # Usar o gateway existente que já funciona
  gateway-optimized:
    image: llm-smart-gateway-v2:latest
    container_name: gateway-optimized
    restart: unless-stopped
    ports:
      - "8200:8200"
    environment:
      - GATEWAY_VERSION=2.1-SINGLE
      - SINGLE_MODEL_MODE=true
    networks:
      - llm-network
    depends_on:
      - llama-8b-maxed
    # Vamos montar um arquivo de configuração customizado
    volumes:
      - ./gateway-config:/app/config:ro

networks:
  llm-network:
    driver: bridge
DCEOF

# 4. Criar configuração do gateway para modo único
echo "🌐 Configurando gateway para modo único..."
mkdir -p gateway-config
cat > gateway-config/single_model_app.py << 'GWEOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import time

app = FastAPI(title="Llama 8B Gateway - Single Model Optimized")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LLAMA_URL = "http://llama-8b-maxed:8000"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None
    task_type: Optional[str] = None
    class Config:
        protected_namespaces = ()

@app.get("/")
def root():
    return {
        "service": "Llama 8B Gateway - Maximum Performance",
        "version": "2.1-SINGLE",
        "model": "llama-3.1-8b-maximized",
        "optimization": "RTX 3060 12GB Dedicated",
        "status": "operational"
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_URL}/health")
            return response.json()
    except:
        return {"status": "unhealthy", "available_models": 0}

@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_URL}/")
            data = response.json()
        return {
            "models": [{
                "id": "llama-8b-maximized",
                "name": "Llama 3.1 8B MAXIMIZED",
                "tier": "premium",
                "specialties": ["general", "complex-tasks", "reasoning", "code"],
                "available": data.get("model_loaded", False),
                "url": LLAMA_URL
            }]
        }
    except:
        return {"models": []}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{LLAMA_URL}/generate",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
GWEOF

# 5. Iniciar sistema corrigido
echo "🚀 Iniciando sistema corrigido..."
docker-compose -f docker-compose.llama-fixed.yml up -d

echo "✅ Sistema corrigido iniciado!"
echo "📊 Aguarde 30 segundos para o modelo carregar..."
echo ""
echo "URLs:"
echo "- Llama 8B Direct: http://localhost:8101"
echo "- Gateway API: http://localhost:8200"
echo ""
echo "Para verificar logs:"
echo "docker logs -f llama-8b-maxed"
