import os
import time
import gc
import torch
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
    
    # CONFIGURAÇÕES MAXIMIZADAS PARA RTX 3060 12GB
    model_path = os.environ.get("MODEL_PATH", "/app/models/llama/llama-3.1-8b-instruct-q4_k_m.gguf")
    model_name = "Llama 3.1 8B MAXIMIZED"
    
    # MÁXIMO DE GPU LAYERS PARA 8B Q4_K_M (~4.5GB)
    # Com 12GB VRAM, podemos colocar TUDO na GPU + contexto grande
    gpu_layers = 45  # Máximo para garantir tudo na GPU
    context_size = 8192  # Contexto grande mas estável
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"🚀 CARREGANDO MODELO MAXIMIZADO: {model_name}")
    print(f"📁 Caminho: {model_path}")
    print(f"🎯 GPU Layers: {gpu_layers} (MÁXIMO)")
    print(f"📚 Context: {context_size} (OTIMIZADO)")
    print(f"🔥 RTX 3060 12GB - DEDICADA 100%")
    
    try:
        # PARÂMETROS OTIMIZADOS PARA MÁXIMA PERFORMANCE
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,      # MÁXIMO DE LAYERS NA GPU
            n_threads=16,                  # MÁXIMO DE THREADS
            n_threads_batch=16,            # THREADS PARA BATCH
            verbose=False,                 # SEM LOGS PARA PERFORMANCE
            seed=42,
            use_mlock=True,               # LOCK MEMORY
            use_mmap=True,                # MEMORY MAPPING
            f16_kv=True,                  # FP16 KEY-VALUE
            n_batch=2048,                 # BATCH MÁXIMO
            rope_scaling_type=-1,         # DESABILITAR ROPE SCALING
            offload_kqv=True,            # OFFLOAD KQV PARA GPU
            flash_attn=True,             # FLASH ATTENTION SE DISPONÍVEL
            mul_mat_q=True,              # MULTIPLICAÇÃO QUANTIZADA
            tensor_split=None,           # SEM SPLIT (TUDO EM UMA GPU)
            low_vram=False,              # NÃO ECONOMIZAR VRAM
            logits_all=False,            # APENAS ÚLTIMO TOKEN
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "optimization": "MAXIMUM_PERFORMANCE",
            "dedicated_gpu": "RTX 3060 12GB"
        }
        
        # Forçar garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        "gpu_layers": model_config.get("gpu_layers", 0)
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
        
        # PARÂMETROS OTIMIZADOS PARA GERAÇÃO RÁPIDA
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 4096),  # Limite aumentado
            temperature=request.temperature,
            echo=False,
            stream=False,
            top_p=0.95,                   # Sampling otimizado
            top_k=40,                     # Top-K padrão
            repeat_penalty=1.1,           # Penalidade leve
            frequency_penalty=0.0,        # Sem penalidade de frequência
            presence_penalty=0.0,         # Sem penalidade de presença
            tfs_z=1.0,                   # Tail free sampling
            typical_p=1.0,               # Typical sampling
            mirostat_mode=0,             # Desabilitar mirostat
            penalize_nl=False,           # Não penalizar newlines
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
            "performance": "MAXIMIZED"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

# Teste rápido na inicialização
@app.on_event("startup")
async def warmup():
    await asyncio.sleep(2)  # Aguardar modelo carregar
    if model_instance:
        print("🔥 Aquecendo modelo...")
        try:
            model_instance.create_completion("Hello", max_tokens=1)
            print("✅ Modelo aquecido e pronto!")
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    # Usar apenas 1 worker para não dividir GPU
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
