import os
import time
import gc
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="LLM UltraFast Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

def load_model():
    global model_instance, model_config
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/default.gguf")
    model_name = os.environ.get("MODEL_NAME", "Default")
    gpu_layers = int(os.environ.get("GPU_LAYERS", 50))
    context_size = int(os.environ.get("CONTEXT_SIZE", 1024))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🚀 Carregando modelo ULTRA RÁPIDO: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (MÁXIMO)")
    print(f"📚 Context: {context_size} (OTIMIZADO)")
    
    try:
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=12,              # MAIS THREADS
            verbose=False,             # SEM LOGS VERBOSOS
            seed=42,
            use_mlock=True,
            use_mmap=True,
            f16_kv=True,
            n_batch=1024,              # BATCH MAIOR
            rope_scaling_type=-1,      # OTIMIZAÇÃO ROPE
            offload_kqv=True,         # OFFLOAD KQV PARA GPU
            flash_attn=True,          # FLASH ATTENTION
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "optimized": True
        }
        
        print(f"✅ Modelo {model_name} carregado ULTRA RÁPIDO!")
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
        "status": "ultrafast",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "optimizations": ["max_gpu_layers", "reduced_context", "flash_attention", "optimized_batch"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "performance": "ultrafast"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        start_time = time.time()
        
        # CONFIGURAÇÕES OTIMIZADAS PARA VELOCIDADE
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 500),  # LIMITE MENOR
            temperature=request.temperature,
            echo=False,
            stream=False,
            top_p=0.9,                # OTIMIZADO
            top_k=40,                 # OTIMIZADO
            repeat_penalty=1.1,       # OTIMIZADO
            tfs_z=1.0,               # TAIL FREE SAMPLING
            typical_p=1.0,           # TYPICAL SAMPLING
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
            "performance": "ultrafast"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run("llm_server_ultrafast:app", host="0.0.0.0", port=port, workers=1)
