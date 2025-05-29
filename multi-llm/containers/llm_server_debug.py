import os
import time
import traceback
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python importado com sucesso")
except ImportError as e:
    print(f"❌ ERRO DE IMPORTAÇÃO: {e}")
    exit(1)

app = FastAPI(title="LLM Debug Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_instance: Optional[Llama] = None
model_config: Dict[str, Any] = {}
startup_time = time.time()
last_error = ""

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

def load_model():
    global model_instance, model_config, last_error
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/default.gguf")
    model_name = os.environ.get("MODEL_NAME", "Default")
    gpu_layers = int(os.environ.get("GPU_LAYERS", 0))  # Começar com 0
    context_size = int(os.environ.get("CONTEXT_SIZE", 512))  # Menor para teste
    
    print(f"🔄 DEBUG: Carregando modelo")
    print(f"   Nome: {model_name}")
    print(f"   Caminho: {model_path}")
    print(f"   GPU Layers: {gpu_layers}")
    print(f"   Context: {context_size}")
    
    # Verificar se arquivo existe
    if not os.path.exists(model_path):
        error_msg = f"❌ Arquivo não encontrado: {model_path}"
        print(error_msg)
        last_error = error_msg
        return False
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(model_path)
    print(f"📊 Tamanho do arquivo: {file_size / (1024*1024*1024):.2f} GB")
    
    if file_size == 0:
        error_msg = f"❌ Arquivo vazio: {model_path}"
        print(error_msg)
        last_error = error_msg
        return False
    
    try:
        print("🚀 Iniciando carregamento do modelo...")
        
        # Parâmetros mais conservadores
        model_instance = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=4,        # Reduzido
            verbose=True,       # Para debug
            seed=42,
            use_mlock=False,    # Desabilitado para teste
            use_mmap=True,
            f16_kv=True,
            n_batch=256,        # Reduzido
        )
        
        model_config = {
            "name": model_name,
            "path": model_path,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "file_size_gb": file_size / (1024*1024*1024)
        }
        
        print(f"✅ SUCESSO! Modelo {model_name} carregado!")
        last_error = ""
        return True
        
    except Exception as e:
        error_msg = f"❌ ERRO DETALHADO: {str(e)}"
        print(error_msg)
        print("📋 Traceback completo:")
        traceback.print_exc()
        last_error = f"{str(e)} | {traceback.format_exc()}"
        return False

@app.get("/")
def read_root():
    uptime = time.time() - startup_time
    return {
        "status": "running",
        "model": model_config.get("name", "Unknown"),
        "model_loaded": model_instance is not None,
        "uptime_seconds": f"{uptime:.1f}",
        "last_error": last_error,
        "config": model_config
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_instance else "no_model",
        "model_loaded": model_instance is not None,
        "last_error": last_error
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail=f"Modelo não carregado. Erro: {last_error}")
    
    try:
        start_time = time.time()
        
        response = model_instance.create_completion(
            request.prompt,
            max_tokens=min(request.max_tokens, 100),  # Limitado para teste
            temperature=request.temperature,
            echo=False,
            stream=False
        )
        
        generation_time = time.time() - start_time
        
        return {
            "text": response["choices"][0]["text"],
            "model": model_config.get("name", "Unknown"),
            "generation_time": f"{generation_time:.2f}s",
            "finish_reason": response["choices"][0]["finish_reason"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

# Carregar modelo na inicialização
print("🚀 Iniciando carregamento do modelo...")
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    uvicorn.run("llm_server_debug:app", host="0.0.0.0", port=port, workers=1)
