#!/bin/bash
set -e

SERVER_PORT=${SERVER_PORT:-8001}
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
GPU_LAYERS=${GPU_LAYERS:-32}
CONTEXT_SIZE=${CONTEXT_SIZE:-4096}
SERVER_THREADS=${SERVER_THREADS:-8}
DEFAULT_MODEL=${DEFAULT_MODEL:-llama2-7b}

echo "=== Servidor Llama GPU ==="
echo "GPU Layers: $GPU_LAYERS"
echo "Modelo: $DEFAULT_MODEL"

if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU NVIDIA detectada"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

cat > /app/server.py << 'PYEOF'
import os
import time
import threading
import gc
import traceback
from typing import Optional, Dict, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python importado com sucesso")
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="Servidor Llama GPU - Português")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

current_model: Optional[Llama] = None
current_model_name: str = ""
model_lock = threading.Lock()
startup_time = time.time()

MODELS = {
    "llama2-7b": {
        "path": "/app/models/llama/llama-2-7b-chat-q4_k_m.gguf",
        "name": "Llama 2 7B Chat (Português)",
        "estimated_vram": 4.5,
        "context_size": 4096
    },
    "codellama-7b": {
        "path": "/app/models/llama/codellama-7b-instruct-q4_k_m.gguf",
        "name": "Code Llama 7B (Código + Português)",
        "estimated_vram": 4.5,
        "context_size": 4096
    },
    "llama2-13b": {
        "path": "/app/models/llama/llama-2-13b-chat-q3_k_m.gguf",
        "name": "Llama 2 13B Chat (Alta Qualidade)",
        "estimated_vram": 7.5,
        "context_size": 4096
    }
}

GPU_LAYERS = int(os.environ.get("GPU_LAYERS", 32))
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 4096))
SERVER_THREADS = int(os.environ.get("SERVER_THREADS", 8))

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    model_id: Optional[str] = None

class ModelSwitchRequest(BaseModel):
    model_id: str

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

def test_model_file(model_path: str) -> bool:
    try:
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não existe: {model_path}")
            return False
        
        size = os.path.getsize(model_path)
        if size == 0:
            print(f"❌ Arquivo vazio: {model_path}")
            return False
        
        print(f"✅ Arquivo OK: {model_path} ({size // (1024*1024)} MB)")
        
        with open(model_path, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'GGUF'):
                print("✅ GGUF válido")
                return True
            else:
                print(f"❌ Header inválido: {header}")
                return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def load_model(model_id: str) -> bool:
    global current_model, current_model_name
    
    if model_id not in MODELS:
        print(f"❌ Modelo {model_id} não encontrado. Disponíveis: {list(MODELS.keys())}")
        return False
    
    model_config = MODELS[model_id]
    model_path = model_config["path"]
    
    if not test_model_file(model_path):
        return False
    
    try:
        if current_model is not None:
            print(f"Descarregando: {current_model_name}")
            del current_model
            gc.collect()
            time.sleep(2)
        
        print(f"🔄 Carregando: {model_config['name']}")
        print(f"📁 Caminho: {model_path}")
        print(f"🎯 GPU Layers: {GPU_LAYERS}")
        print(f"📚 Context: {model_config['context_size']}")
        
        current_model = Llama(
            model_path=model_path,
            n_ctx=model_config["context_size"],
            n_gpu_layers=GPU_LAYERS,
            n_threads=SERVER_THREADS,
            verbose=True,
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            logits_all=False,
            n_batch=512,
        )
        
        current_model_name = model_id
        print(f"✅ SUCESSO! Modelo {model_config['name']} carregado!")
        print(f"📊 GPU Memory: {get_gpu_memory()}")
        return True
        
    except Exception as e:
        print(f"❌ ERRO DETALHADO:")
        print(f"   Modelo: {model_id}")
        print(f"   Erro: {str(e)}")
        traceback.print_exc()
        current_model = None
        current_model_name = ""
        return False

@app.on_event("startup")
async def startup_event():
    default_model = os.environ.get("DEFAULT_MODEL", "llama2-7b")
    print(f"🚀 Carregando modelo: {default_model}")
    print(f"📋 Disponíveis: {list(MODELS.keys())}")
    
    # Tentar carregar na ordem de compatibilidade
    models_to_try = [default_model, "llama2-7b", "codellama-7b", "llama2-13b"]
    
    for model in models_to_try:
        if model in MODELS:
            print(f"🔄 Tentando: {model}")
            if load_model(model):
                print(f"✅ Sucesso com: {model}")
                break
            else:
                print(f"❌ Falhou: {model}")
    else:
        print("⚠️  Nenhum modelo carregado!")

@app.get("/")
def read_root():
    return {
        "status": "running",
        "current_model": current_model_name if current_model else "none",
        "available_models": list(MODELS.keys()),
        "gpu_memory": get_gpu_memory(),
        "uptime": f"{time.time() - startup_time:.1f}s",
        "language": "português + english"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": current_model is not None,
        "current_model": current_model_name,
        "gpu_info": get_gpu_memory()
    }

@app.get("/models")
def list_models():
    model_list = []
    for model_id, config in MODELS.items():
        model_list.append({
            "id": model_id,
            "name": config["name"],
            "estimated_vram_gb": config["estimated_vram"],
            "available": test_model_file(config["path"]),
            "currently_loaded": model_id == current_model_name,
            "file_size_mb": os.path.getsize(config["path"]) // (1024*1024) if os.path.exists(config["path"]) else 0
        })
    return {"models": model_list}

@app.post("/switch_model")
async def switch_model(request: ModelSwitchRequest):
    with model_lock:
        if request.model_id == current_model_name:
            return {"message": f"Modelo {request.model_id} já carregado"}
        
        if load_model(request.model_id):
            return {"message": f"Sucesso! Modelo: {request.model_id}", "gpu_memory": get_gpu_memory()}
        else:
            raise HTTPException(status_code=500, detail=f"Falha: {request.model_id}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    if current_model is None:
        raise HTTPException(status_code=503, detail="Nenhum modelo carregado")
    
    try:
        start_time = time.time()
        
        with model_lock:
            response = current_model.create_completion(
                request.prompt,
                max_tokens=min(request.max_tokens, 512),
                temperature=request.temperature,
                echo=False,
                stream=False
            )
        
        generation_time = time.time() - start_time
        tokens_generated = len(response["choices"][0]["text"].split())
        
        return {
            "text": response["choices"][0]["text"],
            "model": current_model_name,
            "finish_reason": response["choices"][0]["finish_reason"],
            "generation_time": f"{generation_time:.2f}s",
            "tokens_per_second": f"{tokens_generated / generation_time:.1f}" if generation_time > 0 else "0",
            "gpu_memory": get_gpu_memory()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host=os.environ.get("SERVER_HOST", "0.0.0.0"), port=int(os.environ.get("SERVER_PORT", 8001)), workers=1)
PYEOF

echo "🚀 Iniciando servidor Llama..."
exec python3 /app/server.py
