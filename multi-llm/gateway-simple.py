from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import time
import uvicorn

app = FastAPI(title="Simple Gateway for Llama")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LLAMA_URL = "http://localhost:8101"  # Acessar direto pela porta do host

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    model_preference: Optional[str] = None
    task_type: Optional[str] = None
    class Config:
        protected_namespaces = ()

@app.get("/")
def root():
    return {
        "service": "Simple Gateway",
        "model": "llama-8b-optimized",
        "status": "operational"
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_URL}/health")
            data = response.json()
            data["available_models"] = 1
            data["models"] = ["llama-8b-optimized"]
            return data
    except:
        return {"status": "unhealthy", "available_models": 0}

@app.get("/models")
async def models():
    return {
        "models": [{
            "id": "llama-8b-optimized",
            "name": "Llama 3.1 8B MAXIMIZED",
            "tier": "premium",
            "specialties": ["general", "complex-tasks", "reasoning", "code"],
            "available": True,
            "url": LLAMA_URL
        }]
    }

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
            result["gateway_info"] = {
                "selected_model": "llama-8b-optimized",
                "model_tier": "premium",
                "selection_reason": "single_model_optimized",
                "total_time": f"{total_time:.2f}s"
            }
            return result
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
