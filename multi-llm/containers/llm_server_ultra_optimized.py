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
    print("✅ llama-cpp-python importado com sucesso")
except ImportError as e:
    print(f"❌ ERRO: {e}")
    exit(1)

app = FastAPI(title="LLM Ultra Optimized Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
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
    gpu_layers = int(os.environ.get("GPU_LAYERS", 60))  # MÁXIMO!
    context_size = int(os.environ.get("CONTEXT_SIZE", 8192))
    
    print(f"🚀 Carregando modelo ULTRA OTIMIZADO: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (MÁXIMO)")
    print(f"📚 Context: {context_size}")
    
    # Verificar se é arquivo dividido
    if "00001-of-00002" in model_path:
        base_path = model_path.replace("-00001-of-00002", "")
        print(f"📂 Detectado arquivo dividido, base: {base_path}")
        model_path = model_path  # llama.cpp carrega automaticamente
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    try:
        # Configurações ULTRA OTIMIZADAS para RTX 3060
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,      # Máximo de layers na GPU
            n_threads=16,                  # Máximo de threads
            n_threads_batch=16,            # Threads para batch
            verbose=False,
            seed=42,
            use_mlock=True,               # Lock na memória
            use_mmap=True,                # Memory mapping
            f16_kv=True,                  # FP16 para KV cache
            n_batch=1024,                 # Batch size maior
            rope_scaling_type=-1,         # RoPE scaling otimizado
            offload_kqv=True,            # Offload KQV para GPU
            flash_attn=True,             # Flash Attention se disponível
            split_mode=1,                # Split mode para multi-GPU (se tiver)
            tensor_split=None,           # Auto split
            main_gpu=0,                  # GPU principal
            mul_mat_q=True,              # Multiplicação quantizada
            logits_all=False,            # Só logits necessários
            embedding=False,             # Sem embeddings (economiza memória)
            rope_freq_base=10000000.0,   # Para Qwen 2.5
            rope_freq_scale=1.0,
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "optimizations": "ULTRA"
        }
        
        print(f"✅ Modelo {model_name} carregado com ULTRA OTIMIZAÇÃO!")
        print(f"💾 GPU Memory: {get_gpu_memory()}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        print("❌ Falha ao carregar modelo na inicialização")
    gc.collect()  # Limpar memória

@app.get("/")
def read_root():
    uptime = time.time() - startup_time
    return {
        "status": "ultra_optimized",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "system": get_system_info(),
        "gpu": get_gpu_memory(),
        "optimizations": ["max_gpu_layers", "flash_attention", "optimized_batch", "fp16_kv"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "gpu_memory": get_gpu_memory()
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        start_time = time.time()
        
        # Configurações otimizadas para velocidade
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 2048),
            temperature=request.temperature,
            echo=False,
            stream=False,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            tfs_z=1.0,
            mirostat_mode=0,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]  # Stop tokens para Qwen
        )
        
        generation_time = time.time() - start_time
        
        return {
            "text": response["choices"][0]["text"],
            "model": model_config.get("name", "Unknown"),
            "generation_time": f"{generation_time:.2f}s",
            "finish_reason": response["choices"][0]["finish_reason"],
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
            },
            "system": get_system_info(),
            "performance": "ultra_optimized"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")
    finally:
        gc.collect()  # Limpar memória após geração

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
