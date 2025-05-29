from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import time

app = FastAPI(title="Multi-LLM Gateway - Working")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# MODELOS CONFIRMADOS FUNCIONANDO
MODELS = {
    "llama-8b-balanced": {
        "url": "http://llama-8b-balanced:8000",
        "tier": "balanced",
        "specialties": ["general", "complex-tasks"]
    },
    "codellama-7b": {
        "url": "http://codellama-7b:8000", 
        "tier": "specialist",
        "specialties": ["code", "programming"]
    },
    "mistral-7b-code": {
        "url": "http://mistral-7b-code:8000",
        "tier": "balanced", 
        "specialties": ["code", "technical"]
    },
    "solar-10b-premium": {
        "url": "http://solar-10b-premium:8000",
        "tier": "premium",
        "specialties": ["complex-reasoning"]
    }
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None
    task_type: Optional[str] = None

def select_model(prompt: str, task_type: str = None, preference: str = None) -> str:
    """Seleção simples e direta de modelo"""
    prompt_lower = prompt.lower()
    
    # Por preferência explícita
    if preference == "premium":
        return "solar-10b-premium"
    elif preference == "specialist":
        return "codellama-7b"
    elif preference == "balanced":
        return "llama-8b-balanced"
    
    # Por tipo de tarefa
    if task_type == "code" or any(word in prompt_lower for word in ["código", "code", "python", "função", "programa"]):
        return "codellama-7b"
    elif task_type == "reasoning" or any(word in prompt_lower for word in ["analise", "análise", "complexo", "estratégia"]):
        return "solar-10b-premium"
    
    # Default: modelo balanceado
    return "llama-8b-balanced"

async def check_model(model_id: str, url: str) -> bool:
    """Verifica se modelo está funcionando"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

@app.get("/")
def root():
    return {
        "service": "Multi-LLM Gateway - Working",
        "version": "2.1",
        "models": len(MODELS),
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check do gateway"""
    available = []
    
    # Verificar cada modelo
    for model_id, config in MODELS.items():
        if await check_model(model_id, config["url"]):
            available.append(model_id)
    
    return {
        "status": "healthy" if len(available) >= 2 else "degraded",
        "available_models": len(available),
        "total_models": len(MODELS),
        "models": available,
        "timestamp": time.time()
    }

@app.get("/models")
async def models():
    """Lista todos os modelos"""
    models_info = []
    
    for model_id, config in MODELS.items():
        available = await check_model(model_id, config["url"])
        models_info.append({
            "id": model_id,
            "tier": config["tier"],
            "specialties": config["specialties"],
            "available": available,
            "url": config["url"]
        })
    
    return {"models": models_info}

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Geração de texto usando modelo selecionado"""
    
    # Selecionar modelo
    selected_model = select_model(request.prompt, request.task_type, request.model_preference)
    model_url = MODELS[selected_model]["url"]
    
    # Verificar se modelo está disponível
    if not await check_model(selected_model, model_url):
        # Fallback: tentar qualquer modelo disponível
        for model_id, config in MODELS.items():
            if await check_model(model_id, config["url"]):
                selected_model = model_id
                model_url = config["url"]
                break
        else:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
    
    # Fazer requisição para o modelo
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{model_url}/generate",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Adicionar informações do gateway
            result["gateway_info"] = {
                "selected_model": selected_model,
                "model_tier": MODELS[selected_model]["tier"],
                "selection_reason": "intelligent_routing",
                "total_time": f"{total_time:.2f}s"
            }
            
            return result
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Erro do modelo: {response.text}"
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout na geração")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info")
