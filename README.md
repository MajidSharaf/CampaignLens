# CampaignLens

End-to-end NLP pipeline for analysing YouTube comment sentiment around Donald Trump's 2024 campaign. Scrapes comments, runs sentiment analysis, NER, and topic modelling, then serves two opposing AI chatbots — a Supporter and a Critic — grounded entirely in what real commenters wrote.

Built for CSCI370 · Spring 2026.

---

## What it does

1. **Scrapes** YouTube comments via the Data API
2. **Preprocesses** text — slang expansion, URL/mention removal, deduplication
3. **Analyses** — RoBERTa sentiment (Pro/Anti/Neutral), spaCy NER, TF-IDF keyword extraction, LDA topic modelling per sentiment group
4. **Embeds** comments via Ollama and stores them in Weaviate (hybrid HNSW + BM25 index)
5. **Builds a knowledge graph** in Neo4j using REBEL triplet extraction
6. **Serves a RAG chatbot** — LangGraph routes each question to either Weaviate (comment retrieval) or Neo4j (entity relationships). DSPy ChainOfThought generates grounded answers
7. **Tracks experiments** with MLflow — every router and chatbot run is logged

---

## Architecture

```
YouTube API
    │
    ▼
0_scraper.py → comments.csv
    │
    ▼
1_preprocessing.py → processed_comments.csv
    │
    ▼
2_analysis.py → analysis_results.csv
  (RoBERTa sentiment · spaCy NER · LDA topics)
    │
    ├──────────────────────────┐
    ▼                          ▼
embedding.py            knowledge_graph.py
vectorize.py              (REBEL → Neo4j)
(Weaviate)
    │                          │
    └──────────┬───────────────┘
               ▼
         FastAPI (api.py)
               │
    ┌──────────┴──────────┐
    │                     │
 /chat/*              /api/*
 LangGraph router     analysis · health · eval
 DSPy chatbots
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Scraping | YouTube Data API v3 |
| Preprocessing | pandas, spaCy, custom slang dict |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment |
| NER | spaCy en_core_web_sm |
| Keyword extraction | sklearn TF-IDF (corpus-wide, top-5 per comment) |
| Topic modelling | sklearn LDA (separate Pro / Anti models) |
| Knowledge graph | REBEL (Babelscape/rebel-large) → Neo4j |
| Embeddings | Ollama (embeddinggemma) |
| Vector store | Weaviate — HNSW + BM25 hybrid |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| LLM | qwen2.5:0.5b via Ollama |
| Agent orchestration | LangGraph StateGraph |
| Structured prompting | DSPy ChainOfThought |
| Monitoring | MLflow autolog |
| API | FastAPI |

---

## Running locally

See **[RUN_Instructions.md](RUN_Instructions.md)** for the full walkthrough.

Short version:

```bash
# 1. Run the pipeline (preprocessing → analysis)
bash Pipeline/run_pipeline.sh

# 2. Configure environment
cd Weaviate && cp .env.example .env   # defaults work as-is

# 3. Start everything
docker compose up --build

# 4. Open
open http://localhost:8000
```

---

## Project structure

```
CampaignLens/
├── Pipeline/               # Stages 0–2: scraping, preprocessing, analysis
│   ├── 0_scraper.py
│   ├── 1_preprocessing.py
│   ├── 2_analysis.py
│   └── run_pipeline.sh
│
├── Weaviate/               # Stage 3: embedding + Docker orchestration
│   ├── embedding.py
│   ├── vectorize.py
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── .env.example
│
├── Frontend/               # LLM layer + API + dashboard
│   ├── api.py              # FastAPI — serves pages and all endpoints
│   ├── chatbot.py          # DSPy Supporter + Critic chatbots
│   ├── retrieval.py        # Weaviate hybrid retrieval + reranking
│   ├── analysis.py         # Reads analysis_results.csv for dashboard
│   ├── NEO4j/
│   │   ├── router.py       # LangGraph routing graph
│   │   ├── query.py        # DSPy Neo4j query module
│   │   └── knowledge_graph.py  # REBEL extraction + Neo4j load/query
│   └── static/
│       ├── index.html      # Landing page
│       ├── dashboard.html  # Chatbots · Analysis · Health tabs
│       └── setup.html      # Docker setup guide
│
├── Datasets/               # Raw and versioned data files
├── RUN_Instructions.md
└── README.md
```

---

## Team

| Member | Responsibility |
|---|---|
| Majid | LLM layer · DSPy chatbots · LangGraph routing · FastAPI · Frontend |
| Teammate 2 | Scraping · Preprocessing · Sentiment · NER · Topic modelling · Embedding |
| Teammate 3 | Weaviate · Neo4j · Docker · Integration |

---

## Design decisions

**Why Weaviate over FAISS?**
Weaviate supports hybrid search (vector + BM25 in a single query) and stores metadata (sentiment label, video ID, topic ID) as filterable properties. FAISS is vector-only and has no built-in metadata filtering. For comments where keyword matching matters ("economy", "border") alongside semantic similarity, hybrid retrieval outperforms pure vector search.

**Why two separate collections (PositiveComments / NegativeComments)?**
The chatbots represent opposing viewpoints. Keeping them in separate collections guarantees the Supporter only retrieves Pro comments and the Critic only retrieves Anti comments — no cross-contamination.

**Why qwen2.5:0.5b?**
Hardware constraint — runs on CPU without a GPU. The small model is compensated by strong prompt engineering via DSPy signatures and by grounding responses entirely in retrieved context rather than parametric knowledge.

**Why DSPy over plain prompting?**
DSPy signatures enforce structured outputs (response + confidence score) and the ChainOfThought wrapper adds an explicit reasoning step, which improves coherence on constrained tasks with small models.

**Why LangGraph for routing?**
The routing decision (RAG vs knowledge graph) is a stateful single-step graph — LangGraph's StateGraph maps cleanly to this pattern and is consistent with the lab material. The classifier uses temperature=0 for deterministic routing.

**Hallucination prevention**
- Prompts instruct the model to answer only from retrieved context
- A `confidence` field is returned with every response; anything below 0.6 is flagged `uncertain` in the UI
- Source comment IDs are returned alongside every answer for traceability
- The router always grounds answers in real retrieved comments, never generates from model memory alone
