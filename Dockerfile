FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalação robusta com CUDA Toolkit completo
RUN apt-get clean && apt-get update --fix-missing && \
    apt-get install -y --fix-missing \
    python3 python3-pip python3-dev \
    build-essential cmake ninja-build pkg-config git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Dependências Python
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.23.2 \
    pydantic==2.4.2 \
    httpx==0.25.1 \
    numpy==1.24.3 \
    psutil

# Instalar llama-cpp-python com CUDA usando flags corretas (GGML_CUDA)
RUN CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip3 install --no-cache-dir llama-cpp-python==0.2.19 --verbose

WORKDIR /app
RUN mkdir -p /app/models /app/config

COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/app/entrypoint.sh"]
