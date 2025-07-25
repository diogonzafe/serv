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
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="LLM Balanced Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7

def get_system_info():
    try:
        process = psutil.Process()
        return {
            "memory_mb": f"{process.memory_info().rss / 1024 / 1024:.1f}",
            "cpu_percent": f"{process.cpu_percent():.1f}%"
        }
    except:
        return {"memory_mb": "0", "cpu_percent": "0%"}

def load_model():
    global model_instance, model_config
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/default.gguf")
    model_name = os.environ.get("MODEL_NAME", "Default")
    gpu_layers = int(os.environ.get("GPU_LAYERS", 35))  # SEGURO
    context_size = int(os.environ.get("CONTEXT_SIZE", 2048))  # BALANCEADO
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🚀 Carregando modelo BALANCEADO: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (SEGURO)")
    print(f"📚 Context: {context_size} (BALANCEADO)")
    
    try:
        # CONFIGURAÇÕES BALANCEADAS E SEGURAS
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=8,              # BALANCEADO
            verbose=False,
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            n_batch=512,              # SEGURO
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "performance": "balanced"
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
        "status": "balanced",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "system": get_system_info(),
        "performance": "balanced"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "performance": "balanced"
    }

@app.get("/info")
def model_info():
    return {
        "config": model_config,
        "capabilities": {
            "generate": model_instance is not None,
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
            max_tokens=min(request.max_tokens, 300),  # LIMITE BALANCEADO
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
            "system": get_system_info(),
            "performance": "balanced"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run("llm_server_balanced:app", host="0.0.0.0", port=port, workers=1)
