from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import httpx
import os
import time

app = FastAPI(title="LLM Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server:8001")
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "1024"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300.0"))
ENABLE_MODEL_SWITCHING = os.getenv("ENABLE_MODEL_SWITCHING", "true").lower() == "true"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stop_words: Optional[List[str]] = None
    model_id: Optional[str] = None

class ModelSwitchRequest(BaseModel):
    model_id: str

@app.get("/")
def read_root():
    return {
        "status": "running",
        "llm_server": LLM_SERVER_URL,
        "max_tokens_limit": MAX_TOKENS_LIMIT,
        "model_switching_enabled": ENABLE_MODEL_SWITCHING,
        "features": ["dynamic_model_loading", "gpu_acceleration", "model_switching"]
    }

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLM_SERVER_URL}/health")
            llm_status = "ok" if response.status_code == 200 else "error"
            
            info_response = await client.get(f"{LLM_SERVER_URL}/")
            server_info = info_response.json() if info_response.status_code == 200 else {}
            
            return {
                "status": "healthy",
                "llm_server": llm_status,
                "current_model": server_info.get("current_model", "unknown"),
                "gpu_memory": server_info.get("gpu_memory", {})
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{LLM_SERVER_URL}/models")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_model")
async def switch_model(request: ModelSwitchRequest):
    if not ENABLE_MODEL_SWITCHING:
        raise HTTPException(status_code=403, detail="Model switching disabled")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(f"{LLM_SERVER_URL}/switch_model", json={"model_id": request.model_id})
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerateRequest):
    request.max_tokens = min(request.max_tokens, MAX_TOKENS_LIMIT)
    
    try:
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop_words": request.stop_words
        }
        
        if request.model_id and ENABLE_MODEL_SWITCHING:
            payload["model_id"] = request.model_id
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(f"{LLM_SERVER_URL}/generate", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
