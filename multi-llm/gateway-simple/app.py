from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import os
import time

app = FastAPI(title="Llama 8B Gateway - Simplified")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                  allow_methods=["*"], allow_headers=["*"])

LLAMA_URL = os.environ.get("LLAMA_URL", "http://llama-8b-maxed:8000")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    # Campos compatíveis com API externa
    model_preference: Optional[str] = None
    task_type: Optional[str] = None

@app.get("/")
def root():
    return {
        "service": "Llama 8B Gateway - Maximum Performance",
        "model": "llama-3.1-8b",
        "optimization": "RTX 3060 12GB Dedicated",
        "status": "operational"
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_URL}/health")
            data = response.json()
        return {
            "status": "healthy",
            "model_loaded": data.get("model_loaded", False),
            "available_models": 1,
            "models": ["llama-8b-maximized"],
            "gpu_memory": data.get("gpu_memory", {})
        }
    except:
        return {"status": "unhealthy", "available_models": 0}

@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_URL}/")
            data = response.json()
        return {
            "models": [{
                "id": "llama-8b-maximized",
                "name": "Llama 3.1 8B MAXIMIZED",
                "tier": "premium",
                "specialties": ["general", "complex-tasks", "reasoning", "code"],
                "available": data.get("model_loaded", False),
                "optimization": data.get("optimization", ""),
                "context_size": data.get("context_size", 0),
                "gpu_layers": data.get("gpu_layers", 0)
            }]
        }
    except:
        return {"models": []}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{LLAMA_URL}/generate",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            # Adicionar informações compatíveis com API externa
            result["gateway_info"] = {
                "selected_model": "llama-8b-maximized",
                "model_tier": "premium",
                "selection_reason": "single_model_optimized",
                "total_time": f"{total_time:.2f}s"
            }
            return result
        else:
            raise HTTPException(status_code=response.status_code, 
                              detail=f"Model error: {response.text}")
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Endpoint de compatibilidade com sistemas externos
@app.post("/v1/completions")
async def completions_v1(request: dict):
    """Endpoint compatível com OpenAI API"""
    return await generate(GenerateRequest(
        prompt=request.get("prompt", ""),
        max_tokens=request.get("max_tokens", 1024),
        temperature=request.get("temperature", 0.7)
    ))
