# Running CampaignLens Locally

## Prerequisites

- Docker Desktop (running)
- Python 3.11+ (for the pipeline scripts only)
- A YouTube Data API key (only needed if re-scraping — skip if you have `analysis_results.csv`)

---

## Step 1 — Run the pipeline

This runs all preprocessing, sentiment, NER, and topic modeling locally, then copies the output into the right place for Docker.

```bash
bash Pipeline/run_pipeline.sh
```

> If you already have `analysis_results.csv` and just need to put it in the right place:
> ```bash
> cp your_file.csv Weaviate/dataset/analysis_results.csv
> ```

---

## Step 2 — Set up the environment file

```bash
cd Weaviate
cp .env.example .env
```

The defaults in `.env` work as-is for local Docker. `NEO4J_PASSWORD` is already set to `campaignlens` — it's the password for the local container, you're not retrieving it from anywhere.

Only edit `.env` if you need to add a `YOUTUBE_API_KEY` for scraping.

---

## Step 3 — Start everything

```bash
cd Weaviate
docker compose up --build
```

This starts 6 services:

| Service | What it does | Port |
|---|---|---|
| `ollama` | Local LLM runtime | 11434 |
| `weaviate` | Vector store | 8080 |
| `neo4j` | Knowledge graph DB | 7474 / 7687 |
| `mlflow` | Experiment tracking | 5000 |
| `app` | Pulls embedding model, embeds comments, loads Weaviate | — |
| `chatbot` | FastAPI server — the main interface | **8000** |

**First run takes time.** The `app` service pulls the `embeddinggemma` model and embeds all comments. Watch progress:

```bash
docker compose logs -f app
```

Wait for: `Pipeline complete`

The `chatbot` service starts the API immediately and loads the Neo4j knowledge graph in the background at the same time. Graph-type questions (entity relationships) will return results once it finishes loading.

---

## Step 4 — Open the app

```
http://localhost:8000
```

---

## Stopping

```bash
docker compose down
```

To wipe all stored data and start fresh:

```bash
docker compose down -v
```

---

## Troubleshooting

**Chatbot returns empty responses**
The `app` service is still embedding. Check `docker compose logs app` and wait for `Pipeline complete`.

**Graph questions return nothing**
The knowledge graph is still loading in the background. Give it a few minutes after the app starts.

**Port conflict**
If 8000, 8080, or 7474 are in use, edit the port mappings in `Weaviate/docker-compose.yml`.
