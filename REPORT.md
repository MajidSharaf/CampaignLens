# CampaignLens — Project Report

## Overview

CampaignLens is a political sentiment analysis system built around YouTube comment data. It scrapes public comments from YouTube videos about political figures (primarily Donald Trump), runs a multi-stage NLP pipeline on them, and exposes two AI chatbots — a **Supporter** and a **Critic** — that answer user questions grounded exclusively in what real commenters said. The system is deployed as a web application on a Google Cloud Compute Engine VM and is accessible via a browser.

The project was built for CSCI370 and demonstrates a full-stack ML pipeline integrating data scraping, preprocessing, sentiment analysis, topic modeling, named entity recognition, vector storage, knowledge graph construction, RAG-based retrieval, LLM inference, and a multi-agent debate framework.

---

## Architecture

The system is composed of six Docker containers managed by Docker Compose:

| Container | Role | Port |
|-----------|------|------|
| `ollama` | Local LLM runtime (hosts `qwen2.5:0.5b` and `embeddinggemma`) | 11434 |
| `weaviate` | Vector database storing embedded Pro/Anti comments | 8080 |
| `neo4j` | Graph database storing entity relationship triplets | 7474 / 7687 |
| `mlflow` | Experiment tracking and logging | 5001 |
| `app` | Embedding pipeline (runs once on startup) | — |
| `chatbot` | FastAPI web server (the main app, serves UI + API) | 8000 |

All containers share Docker named volumes for persistence. The application is deployed on GCP (`europe-west1-d`, IP `35.187.54.35`, port `8000`).

---

## Data Pipeline

### Stage 0 — Scraping (`Pipeline/0_scraper.py`)

- Uses the **YouTube Data API v3** to pull all top-level comments from one or more video URLs
- Accepts full YouTube URLs in any format (watch, youtu.be, shorts, embed) — video ID is extracted automatically
- Filters to English-only comments using an ASCII ratio heuristic (≥85% ASCII characters)
- Outputs: `comments.csv` with columns: `author`, `updated_at`, `like_count`, `text`, `video_id`, `public`

### Stage 1 — Preprocessing (`Pipeline/1_preprocessing.py`)

- Generates a stable `comment_id` per comment using SHA-256 hash of `(video_id + author + updated_at)` — truncated to 16 hex characters. This ID never changes and is used to join data across all downstream stages.
- Cleans text by:
  - Removing Unicode emoji (full Unicode emoji block coverage)
  - Stripping URLs and @mentions
  - Removing excessive punctuation
  - Expanding slang terms using a custom `slang_dict.json`
  - Collapsing whitespace
- Outputs: `processed_comments.csv`

### Stage 2 — Analysis (`Pipeline/2_analysis.py`)

Runs four NLP tasks on each cleaned comment:

#### Sentiment Analysis
- Model: `cardiffnlp/twitter-roberta-base-sentiment` (RoBERTa transformer, fine-tuned on tweets)
- Labels: `LABEL_2` → **Pro**, `LABEL_0` → **Anti**, `LABEL_1` → **Neutral**
- Outputs `sentiment_label` and `roberta_confidence` per comment

#### Named Entity Recognition (NER)
- Library: **spaCy** (`en_core_web_sm`)
- Extracts entities with types: `GPE` (locations), `PERSON`, `ORG`, `NORP` (nationalities/groups)
- Stored as JSON per comment: `[{"text": "Trump", "label": "PERSON"}, ...]`

#### Keyword Extraction
- Method: **TF-IDF** (`sklearn.TfidfVectorizer`) with English stop words removed
- Corpus-level vectorizer fit, then top-5 terms extracted per comment by TF-IDF score
- Stored as JSON list per comment

#### Topic Modeling
- Algorithm: **LDA** (`sklearn.LatentDirichletAllocation`, 10 topics, 25 iterations)
- Run **separately** on Pro and Anti comment groups — producing "what supporters talk about" vs "what critics talk about"
- Top 5 words per topic stored; topic assignment per comment stored as `topic_id`
- Fitted models saved as `.pkl` files (`lda_pro.pkl`, `lda_anti.pkl`)
- Neutral comments are not topic-modeled

Outputs: `analysis_results.csv` with all columns joined on `comment_id`

---

## Vector Store (Weaviate)

### Embedding (`Weaviate/embedding.py`)

- Reads `analysis_results.csv`, splits into Pro and Anti DataFrames
- Embeds each cleaned comment using **DSPy's Embedder** with the `embeddinggemma` model (served via Ollama)
- Saves as Parquet: `positive_embeddings.parquet`, `negative_embeddings.parquet`

### Vectorization (`Weaviate/vectorize.py`)

- Creates two Weaviate collections: **`PositiveComments`** and **`NegativeComments`**
- Each collection uses:
  - **HNSW** vector index with cosine distance (`ef=128`, `ef_construction=128`, `max_connections=64`)
  - **BM25** inverted index (`k1=1.2`, `b=0.75`) for hybrid search
  - Self-provided vectors (embeddings from the embedding stage)
- Properties stored: `comment_id`, `cleaned_text`, `sentiment_label`, `video_id`, `topic_id`

### Retrieval (`Frontend/retrieval.py`)

- **Hybrid retrieval**: combines vector similarity (semantic) and BM25 (keyword) with `alpha=0.5` (balanced)
- Retrieves top `k=10` candidates from Weaviate
- **Cross-encoder reranking**: uses `cross-encoder/ms-marco-MiniLM-L6-v2` (sentence-transformers) to score all 10 candidates against the query simultaneously, then keeps the top `top_k=5`
- Returns: context string (joined comments), source comment IDs, full ranked results

---

## Knowledge Graph (Neo4j)

### Construction (`Frontend/NEO4j/knowledge_graph.py`)

- Model: **REBEL** (`Babelscape/rebel-large`) — a seq2seq transformer that extracts subject/relation/object triplets from text
- Runs on every comment in `analysis_results.csv`
- Each triplet is stored in Neo4j as:
  ```
  (Entity {name: subject}) -[:RELATION {type: relation, comment_id: id}]-> (Entity {name: object})
  ```
- The `comment_id` on each edge traces every relationship back to its source comment
- Supports entity queries, relationship queries, and neighbourhood traversal

### Querying (`Frontend/NEO4j/query.py`)

- Uses **DSPy** with two signatures:
  - `ExtractEntity`: extracts the key entity from a user question (e.g. "economy" from "what do critics think of the economy?")
  - `GraphAnswer`: generates a grounded answer from Neo4j triplets
- Returns the same dict shape as `run_chatbot()` so the router can treat both tools identically

---

## LLM Routing (LangGraph)

**`Frontend/NEO4j/router.py`** implements a multi-node routing graph using **LangGraph**:

```
START → classifier → [conditional] → rag_node  → END
                                   → graph_node → END
```

- **Classifier node**: uses `ChatOllama` (temperature=0) to classify whether the question asks about *opinions/feelings* (→ RAG) or *entity relationships/connections* (→ graph)
- **RAG node**: calls `run_chatbot()` with the appropriate Weaviate collection
- **Graph node**: calls `run_graph_query()` which hits Neo4j
- All runs are logged to **MLflow** (`CampaignLens_Router` experiment)

---

## Chatbots (`Frontend/chatbot.py`)

Two chatbot personas powered by **DSPy** with `ChainOfThought`:

### SupporterChatbot
- Grounded in `PositiveComments` (Pro sentiment comments)
- Signature instructs it to speak only from what real supporters said, passionately in first person
- Must not use outside knowledge or present the other side

### CriticChatbot
- Grounded in `NegativeComments` (Anti sentiment comments)
- Signature instructs it to speak only from what real critics said, sharply in first person
- Must not use outside knowledge or present the other side

Both produce:
- `response`: 2-3 sentence answer
- `confidence`: float 0–1 (model's self-assessment of how well the context supports the answer)
- `sources`: list of `comment_id` values used
- `uncertain`: boolean (True if confidence < threshold)

Model: `qwen2.5:0.5b` served via Ollama — a small, fast model suited to constrained single-task inference.

---

## Boxing Debate (`Frontend/debate.py`)

A multi-round adversarial debate between the two personas:

- **Critic attacks first** each round; **Supporter defends** in response
- Each turn retrieves fresh RAG context relevant to what the opponent just said
- Uses a direct LLM call (not DSPy structured output) to avoid field-parsing issues with the small model
- The prompt explicitly instructs grounding in the retrieved comments only
- Returns confidence score per turn (fixed at 0.8 — retrieval quality indicator)

### Judge LLM
A separate evaluation pass using `JudgeSignature` (DSPy ChainOfThought):
- Automatically triggered after every chat exchange where both bots answered the same question
- Scores each response 0–1 and gives one-sentence feedback
- Produces a verdict on which response was stronger
- Results stored in memory and displayed in the Judge tab with 15s auto-refresh

---

## API (`Frontend/api.py`)

FastAPI application serving both the static HTML frontend and the REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Dashboard (chatbots, analysis, health, judge) |
| `/boxing` | GET | Boxing debate page |
| `/setup` | GET | Pipeline setup page |
| `/progress` | GET | Pipeline progress page |
| `/api/health` | GET | Service health check (Weaviate, Ollama, Neo4j, MLflow) |
| `/api/eval` | GET | Live session evaluation metrics |
| `/api/analysis` | GET | Full analysis results from pipeline |
| `/api/pipeline/validate` | POST | Validate YouTube URLs before scraping |
| `/api/pipeline/run` | POST | Start pipeline in background thread |
| `/api/pipeline/status` | GET | Current pipeline stage |
| `/chat/supporter` | POST | Ask the Supporter chatbot |
| `/chat/critic` | POST | Ask the Critic chatbot |
| `/api/debate` | POST | Run a full boxing debate (all rounds, returns at once) |
| `/api/judge/auto` | POST | Log one side's response; triggers judge when both present |
| `/api/judge/history` | GET | All automatically judged exchanges, newest first |
| `/api/label-topic` | POST | Generate a 2-4 word LLM label for a topic's top words |

---

## Frontend

Single-page dashboard (`Frontend/static/dashboard.html`) with four tabs:

### Chatbots Tab
- Side-by-side Supporter (blue) and Critic (red) panels
- Sync toggle: when on, both bots receive the same question simultaneously; when off, each input is independent
- Boxing button in the centre divider links to the debate page
- Each response shows: text, confidence bar, route badge (RAG/graph), source comment IDs
- Every exchange auto-logs to `/api/judge/auto` for background evaluation

### Analysis Tab
Four sub-tabs (Sentiment / Topic Modeling / Keywords / NER):
- **Sentiment**: RoBERTa bar chart + 5 representative example comments per Pro/Anti/Neutral group
- **Topic Modeling**: LDA topics per sentiment group with LLM-generated 2-4 word labels (fetched async)
- **Keywords**: TF-IDF top keywords filterable by All/Pro/Anti
- **NER**: Top 30 entities with frequency bars, filterable by entity type (GPE, PERSON, ORG, NORP)

### Health Tab
- Live status of all 5 services (Weaviate, Ollama, Neo4j, MLflow, Chatbot)
- Weaviate collection sizes
- Live session evaluation metrics: query counts, routing split (RAG vs graph), average confidence, latency (avg + p95), low-confidence questions

### Judge Tab
- History of all automatically evaluated chat exchanges
- Each entry: timestamp, question, both responses with scores and feedback, verdict
- Auto-refreshes every 15 seconds

### Boxing Page (`Frontend/static/boxing.html`)
- Topic input + rounds selector (1–5)
- Pre-renders all round placeholders then reveals each card sequentially (300ms stagger)
- Critic on the left (attacker), Supporter on the right (defender)
- Each card shows: persona tag, confidence pill, response text, source comment IDs

---

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `cardiffnlp/twitter-roberta-base-sentiment` | Sentiment classification | ~125M params |
| `en_core_web_sm` (spaCy) | Named entity recognition | Small |
| `Babelscape/rebel-large` | Relation extraction for knowledge graph | ~400M params |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranking retrieved comments | ~22M params |
| `embeddinggemma` (Ollama) | Text embedding for Weaviate | — |
| `qwen2.5:0.5b` (Ollama) | Chat, routing, judging, debate, topic labeling | 0.5B params |
| LDA (sklearn) | Topic modeling | — |
| TF-IDF (sklearn) | Keyword extraction | — |

---

## Key Libraries

- **DSPy** — LLM module framework (signatures, ChainOfThought, Predict, Embedder)
- **LangGraph** — multi-agent routing graph (classifier → RAG or graph node)
- **LangChain Ollama** (`langchain_ollama`) — ChatOllama for routing classifier
- **Weaviate** Python client — vector store with hybrid search
- **Neo4j** Python driver — graph database for entity relationships
- **FastAPI** — REST API and static file serving
- **MLflow** — experiment tracking (DSPy autolog, LangChain autolog)
- **Transformers** (HuggingFace) — RoBERTa sentiment, REBEL relation extraction
- **sentence-transformers** — cross-encoder reranking
- **scikit-learn** — TF-IDF, LDA
- **spaCy** — NER
- **pandas, pyarrow** — data manipulation and Parquet I/O

---

## Deployment

- **Platform**: Google Cloud Compute Engine (`europe-west1-d`)
- **VM**: `campaignlens-vm`, `e2-standard-4` equivalent, 47GB disk
- **OS**: Ubuntu 24.04 LTS
- **Access**: HTTP on port 8000, GCP firewall rule opens it publicly
- **Orchestration**: Docker Compose with 6 services
- **Persistence**: Named Docker volumes for Weaviate data, Neo4j data, MLflow artifacts, Ollama models
- **Model caching**: Ollama models pulled once into a named volume; skipped on subsequent restarts if already present
- **Source control**: GitHub (`MajidSharaf/CampaignLens`)

---

## Evaluation

The system tracks live session metrics accessible via `/api/eval`:

- **Total queries** and breakdown by persona (supporter vs critic)
- **Routing decisions**: how many queries went to RAG vs knowledge graph
- **Confidence**: average confidence score across all responses, uncertain response rate
- **Latency**: average and p95 response time in milliseconds
- **Low-confidence questions**: last 5 questions where confidence < 0.4

The Judge LLM provides per-exchange qualitative evaluation: scoring each response 0–1, giving specific feedback, and producing a verdict on which was stronger.
