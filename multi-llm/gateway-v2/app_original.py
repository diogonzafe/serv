from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
import time

app = FastAPI(title="Multi-LLM Gateway - Fixed")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# CONFIGURAÇÃO ORIGINAL QUE FUNCIONAVA
MODELS = {
    "llama-8b-balanced": {"url": "http://llama-8b-balanced:8000", "tier": "balanced", "specialties": ["general"]},
    "codellama-7b": {"url": "http://codellama-7b:8000", "tier": "specialist", "specialties": ["code"]},
    "mistral-7b-code": {"url": "http://mistral-7b-code:8000", "tier": "balanced", "specialties": ["code"]},
    "solar-10b-premium": {"url": "http://solar-10b-premium:8000", "tier": "premium", "specialties": ["reasoning"]}
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None
    task_type: Optional[str] = None

def select_model(prompt: str, task_type: str = None, preference: str = None) -> str:
    if preference == "premium":
        return "solar-10b-premium"
    elif preference == "specialist" or (task_type == "code") or "code" in prompt.lower() or "python" in prompt.lower():
        return "codellama-7b"
    else:
        return "llama-8b-balanced"

async def check_health(model_id: str, url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

@app.get("/")
def root():
    return {"service": "Multi-LLM Gateway - Fixed", "version": "2.0-fixed", "models": len(MODELS)}

@app.get("/health")
async def health():
    available = []
    for model_id, config in MODELS.items():
        if await check_health(model_id, config["url"]):
            available.append(model_id)
    return {"status": "healthy", "available_models": len(available), "total_models": len(MODELS), "models": available}

@app.get("/models")
async def models():
    models_info = []
    for model_id, config in MODELS.items():
        available = await check_health(model_id, config["url"])
        models_info.append({"id": model_id, "tier": config["tier"], "specialties": config["specialties"], "available": available})
    return {"models": models_info}

@app.post("/generate")
async def generate(request: GenerateRequest):
    selected_model = select_model(request.prompt, request.task_type, request.model_preference)
    model_url = MODELS[selected_model]["url"]
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{model_url}/generate", json={
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            })
            
            if response.status_code == 200:
                result = response.json()
                result["gateway_info"] = {
                    "selected_model": selected_model,
                    "model_tier": MODELS[selected_model]["tier"]
                }
                return result
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
