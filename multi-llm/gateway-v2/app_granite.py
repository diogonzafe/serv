from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import time

app = FastAPI(title="Multi-LLM Gateway - Granite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODELS = {
    "granite-8b-optimized": {
        "url": "http://granite-8b-optimized:8000",
        "tier": "balanced",
        "specialties": ["general", "complex-tasks", "supervisor"]
    }
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None
    task_type: Optional[str] = None
    class Config:
        protected_namespaces = ()

async def check_model(model_id: str, url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

@app.get("/")
def root():
    return {
        "service": "Multi-LLM Gateway - Granite Edition",
        "version": "2.1-granite",
        "models": len(MODELS),
        "status": "operational"
    }

@app.get("/health")
async def health():
    available = []
    
    for model_id, config in MODELS.items():
        if await check_model(model_id, config["url"]):
            available.append(model_id)
    
    return {
        "status": "healthy" if available else "degraded",
        "available_models": len(available),
        "total_models": len(MODELS),
        "models": available,
        "timestamp": time.time()
    }

@app.get("/models")
async def models():
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
    selected_model = "granite-8b-optimized"
    model_url = MODELS[selected_model]["url"]
    
    if not await check_model(selected_model, model_url):
        raise HTTPException(status_code=503, detail="Modelo Granite não disponível")
    
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
            
            result["gateway_info"] = {
                "selected_model": selected_model,
                "model_tier": MODELS[selected_model]["tier"],
                "selection_reason": "granite_model",
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
