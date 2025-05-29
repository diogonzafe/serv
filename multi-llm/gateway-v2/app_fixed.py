from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import time
import json

app = FastAPI(title="Multi-LLM Smart Gateway v2 - Fixed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração APENAS com modelos que funcionam
MODELS_CONFIG = {
    "llama-8b-balanced": {"url": "http://llama-8b-balanced:8000", "tier": "balanced", "specialties": ["general", "complex-tasks"]},
    "codellama-7b": {"url": "http://codellama-7b:8000", "tier": "specialist", "specialties": ["code", "debugging"]},
    "mistral-7b-code": {"url": "http://mistral-7b-code:8000", "tier": "balanced", "specialties": ["code", "programming"]},
    "solar-10b-premium": {"url": "http://solar-10b-premium:8000", "tier": "premium", "specialties": ["complex-reasoning"]}
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None  # balanced, premium, specialist
    task_type: Optional[str] = None  # code, reasoning, general

def analyze_prompt_for_model_selection(prompt: str, task_type: Optional[str] = None) -> str:
    """Analisa o prompt para selecionar o melhor modelo."""
    prompt_lower = prompt.lower()
    
    # Se tipo de tarefa especificado, usar isso
    if task_type:
        if task_type in ["code", "programming", "debug"]:
            return "codellama-7b"  # Especialista em código
        elif task_type in ["reasoning", "analysis", "complex"]:
            return "solar-10b-premium"  # Premium para raciocínio complexo

    # Análise por palavras-chave
    if any(kw in prompt_lower for kw in ["código", "code", "python", "javascript", "programa", "debug", "error", "função"]):
        return "codellama-7b"
    
    if any(kw in prompt_lower for kw in ["complexo", "detalhado", "aprofundado", "análise", "estratégia"]):
        return "solar-10b-premium"
    
    # Default balanceado
    return "llama-8b-balanced"

async def check_model_health(model_id: str, url: str) -> bool:
    """Verifica se um modelo está saudável."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

async def get_available_models() -> List[str]:
    """Retorna lista de modelos disponíveis."""
    available = []
    
    tasks = []
    for model_id, config in MODELS_CONFIG.items():
        tasks.append(check_model_health(model_id, config["url"]))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, (model_id, config) in enumerate(MODELS_CONFIG.items()):
        if results[i] is True:
            available.append(model_id)
    
    return available

@app.get("/")
def read_root():
    return {
        "service": "Multi-LLM Smart Gateway v2 - Fixed",
        "version": "2.0-fixed",
        "models_configured": len(MODELS_CONFIG),
        "features": ["intelligent_model_selection", "working_models_only", "fast_responses"]
    }

@app.get("/health")
async def health_check():
    available_models = await get_available_models()
    
    return {
        "status": "healthy" if available_models else "degraded",
        "available_models": len(available_models),
        "total_models": len(MODELS_CONFIG),
        "models": available_models
    }

@app.get("/models")
async def list_models():
    available_models = await get_available_models()
    
    models_info = []
    for model_id, config in MODELS_CONFIG.items():
        model_info = {
            "id": model_id,
            "tier": config["tier"],
            "specialties": config["specialties"],
            "available": model_id in available_models,
            "url": config["url"]
        }
        models_info.append(model_info)
    
    return {"models": models_info}

@app.post("/generate")
async def smart_generate(request: GenerateRequest):
    # Selecionar modelo baseado no prompt e preferências
    if request.model_preference:
        # Filtrar por tier preferido
        candidates = [m for m, c in MODELS_CONFIG.items() if c["tier"] == request.model_preference]
        if candidates:
            selected_model = candidates[0]
        else:
            selected_model = analyze_prompt_for_model_selection(request.prompt, request.task_type)
    else:
        selected_model = analyze_prompt_for_model_selection(request.prompt, request.task_type)
    
    # Verificar se modelo está disponível
    available_models = await get_available_models()
    
    if selected_model not in available_models:
        # Fallback para qualquer modelo disponível
        if available_models:
            selected_model = available_models[0]
        else:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
    
    # Fazer requisição para o modelo selecionado
    model_url = MODELS_CONFIG[selected_model]["url"]
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Timeout maior
            response = await client.post(
                f"{model_url}/generate",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Adicionar informações do gateway
                result["gateway_info"] = {
                    "selected_model": selected_model,
                    "model_tier": MODELS_CONFIG[selected_model]["tier"],
                    "selection_reason": "smart_routing_fixed"
                }
                return result
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except Exception as e:
        # Log do erro
        print(f"Erro na geração com {selected_model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
