#!/bin/bash

echo "🧪 Testing GPU LLM Server..."

echo "1. Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo -e "\n2. Available models:"
curl -s http://localhost:8000/models | python3 -m json.tool

echo -e "\n3. Current model info:"
curl -s http://localhost:8000/current_model | python3 -m json.tool

echo -e "\n4. Test generation:"
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}' | python3 -m json.tool

echo -e "\n✅ Tests completed!"
