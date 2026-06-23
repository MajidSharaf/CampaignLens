#!/bin/bash
set -e

CSV="/app/pipeline_data/analysis_results.csv"

if [ -f "$CSV" ]; then
    echo "[chatbot] Loading knowledge graph into Neo4j in background..."
    python NEO4j/knowledge_graph.py "$CSV" 500 &
    echo "[chatbot] Knowledge graph loading started (PID $!). API starting now."
else
    echo "[chatbot] WARNING: $CSV not found — Neo4j graph will be empty. Check PIPELINE_DATA_PATH."
fi

exec uvicorn api:app --host 0.0.0.0 --port 8000
