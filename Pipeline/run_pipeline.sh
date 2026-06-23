#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/../Weaviate/dataset"

echo "==> Installing pipeline dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo "==> Downloading spaCy model..."
python -m spacy download en_core_web_sm --quiet

cd "$SCRIPT_DIR"

if [ ! -f "comments.csv" ]; then
    echo "==> Stage 0: Scraping YouTube comments..."
    python 0_scraper.py
else
    echo "==> Stage 0: comments.csv already exists, skipping scrape"
fi

echo "==> Stage 1: Preprocessing..."
python 1_preprocessing.py

echo "==> Stage 2: Sentiment, NER, topic modeling..."
python 2_analysis.py

echo "==> Copying analysis_results.csv to Weaviate/dataset/..."
mkdir -p "$DATASET_DIR"
cp analysis_results.csv "$DATASET_DIR/analysis_results.csv"

echo ""
echo "Pipeline complete. analysis_results.csv is ready in Weaviate/dataset/"
echo "Next: cd ../Weaviate && docker compose up --build"
