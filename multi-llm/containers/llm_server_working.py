import os
import time
import gc
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import psutil

try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="LLM Working Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Estado global
model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
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
    
    # Carregar configuração IGUAL AO QUE FUNCIONAVA
    model_path = os.environ.get("MODEL_PATH", "/app/models/default.gguf")
    model_name = os.environ.get("MODEL_NAME", "Default")
    gpu_layers = int(os.environ.get("GPU_LAYERS", 32))      # Pequeno ajuste: 30->32
    context_size = int(os.environ.get("CONTEXT_SIZE", 4096))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🔄 Carregando modelo (configuração que funcionava): {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (pequeno ajuste)")
    print(f"📚 Context: {context_size}")
    
    try:
        # MESMAS CONFIGURAÇÕES QUE FUNCIONAVAM + pequenos ajustes
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=8,                    # IGUAL AO ORIGINAL
            verbose=False,
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            n_batch=512,                    # IGUAL AO ORIGINAL
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "version": "working_with_speed_tweaks"
        }
        
        print(f"✅ Modelo {model_name} carregado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        print("❌ Falha ao carregar modelo na inicialização")

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
        "version": "working_with_speed_tweaks"
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
        
        # MESMAS CONFIGURAÇÕES QUE FUNCIONAVAM
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 1024),
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

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run("llm_server_working:app", host="0.0.0.0", port=port, workers=1)
