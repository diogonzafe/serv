import os
import time
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="LLM Wrapper for llama.cpp server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# URL do servidor llama.cpp
LLAMA_SERVER = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LLAMA_SERVER}/health")
            return {"status": "healthy", "backend": "llama.cpp-cuda"}
    except:
        return {"status": "unhealthy"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        start_time = time.time()
        
        # Fazer requisição para llama.cpp server
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER}/v1/completions",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": False
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            generation_time = time.time() - start_time
            
            return {
                "text": data["choices"][0]["text"],
                "model": os.environ.get("MODEL_NAME", "Llama"),
                "generation_time": f"{generation_time:.2f}s",
                "finish_reason": data["choices"][0]["finish_reason"],
                "usage": data.get("usage", {}),
                "gpu_enabled": True
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Backend error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
