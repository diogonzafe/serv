#!/bin/bash
# Modifica o gateway para usar apenas um modelo
docker exec smart-gateway-single bash -c '
cat > /tmp/gateway_patch.py << "PATCH"
# Patch para modo single model
import sys
sys.path.insert(0, "/app/custom")

# Override da configuração de modelos
MODELS = {
    "llama-8b-optimized": {
        "url": "http://llama-8b-optimized:8000",
        "tier": "premium",
        "specialties": ["general", "complex-tasks", "reasoning", "code"]
    }
}

# Override da seleção de modelo
def select_model(*args, **kwargs):
    return "llama-8b-optimized"

# Aplicar patch
import app
app.MODELS = MODELS
app.select_model = select_model
PATCH

# Reiniciar o processo uvicorn
pkill -f uvicorn || true
cd /app && python -c "import app; exec(open(\"/tmp/gateway_patch.py\").read()); import uvicorn; uvicorn.run(app.app, host=\"0.0.0.0\", port=8200)" &
'
