# Configuração para redirecionar todas as requisições para o Llama otimizado
MODELS = {
    "llama-8b-optimized": {
        "url": "http://llama-8b-optimized:8000",
        "tier": "premium",
        "specialties": ["general", "complex-tasks", "reasoning", "code"]
    }
}

# Sempre retornar o mesmo modelo
def select_model(*args, **kwargs):
    return "llama-8b-optimized"
