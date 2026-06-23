#!/bin/bash
set -e

# Wait for Ollama API to be fully ready (not just TCP reachable)
echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama ready"

# Pull model only if not already cached
if curl -s http://ollama:11434/api/tags | grep -q '"embeddinggemma"'; then
    echo "embeddinggemma already cached, skipping pull"
else
    echo "Pulling embeddinggemma model..."
    while true; do
        status=$(curl -s -X POST http://ollama:11434/api/pull \
            -H "Content-Type: application/json" \
            -d '{"name":"embeddinggemma"}' | tail -1)
        echo "Pull status: $status"
        if echo "$status" | grep -q '"status":"success"'; then
            echo "Model pull complete"
            break
        fi
        echo "Pull not confirmed, retrying in 5s..."
        sleep 5
    done
fi

# Verify the model can actually run before kicking off embedding
echo "Verifying model is loadable..."
until curl -s -X POST http://ollama:11434/api/embed \
    -H "Content-Type: application/json" \
    -d '{"model":"embeddinggemma","input":"test"}' \
    | grep -q '"embeddings"'; do
    echo "Model not ready yet, retrying in 3s..."
    sleep 3
done
echo "Model verified"

# Wait for Weaviate
echo "Waiting for Weaviate..."
until curl -s http://weaviate:8080/v1/.well-known/ready > /dev/null; do
    sleep 2
done
echo "Weaviate ready"

# Run pipeline
echo "Running embedding..."
python embedding.py
echo "Running vectorize..."
python vectorize.py
echo "Pipeline complete"
