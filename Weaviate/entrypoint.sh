#!/bin/bash

# Wait for Ollama
echo "Waiting for Ollama..."
until curl -s http://ollama:11434 > /dev/null; do
    sleep 2
done
echo "Ollama ready"

# Pull model into the Ollama service (REST API, not local CLI)
echo "Pulling embeddinggemma model..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name":"embeddinggemma"}' | tail -1
echo "Model pull complete"

# Wait for Weaviate
echo "Waiting for Weaviate..."
until curl -s http://weaviate:8080/v1/.well-known/ready > /dev/null; do
    sleep 2
done
echo "Weaviate ready"

# Run pipeline
python embedding.py
python vectorize.py