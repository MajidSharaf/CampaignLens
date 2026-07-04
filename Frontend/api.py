import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import subprocess
import threading
import httpx
import json as _json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from NEO4j.router import run_router
from analysis import load_analysis
from debate import run_debate, run_judge, stream_debate

load_dotenv()

app = FastAPI(title="CampaignLens API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory evaluation tracker
# Accumulates per-request stats across the session.
# Logged to MLflow per run; aggregated here for the dashboard.
# ---------------------------------------------------------------------------

_stats = {
    "total_queries": 0,
    "supporter_queries": 0,
    "critic_queries": 0,
    "rag_routes": 0,
    "graph_routes": 0,
    "uncertain_count": 0,
    "similarity_sum": 0.0,
    "response_times_ms": [],
    "low_similarity_questions": [],   # questions where similarity < 0.4
}

# ---------------------------------------------------------------------------
# Judge history — auto-populated after every chatbot response
# ---------------------------------------------------------------------------

_judge_latest: dict = {}    # {"supporter": {question, response}, "critic": {question, response}}
_judge_history: list = []   # list of verdict dicts, newest last

def _auto_judge(s_question: str, s_resp: str, c_question: str, c_resp: str):
    """Called in a background thread after every response, uses latest from each side."""
    try:
        verdict = run_judge(question=s_question, supporter_response=s_resp, critic_response=c_resp)
        _judge_history.append({
            "timestamp": int(time.time()),
            "question": s_question,
            "critic_question": c_question,
            "supporter_response": s_resp,
            "critic_response": c_resp,
            **verdict,
        })
    except Exception as e:
        print(f"[judge] auto-judge failed: {e}")


def _record(persona: str, result: dict, elapsed_ms: float):
    _stats["total_queries"] += 1
    _stats[f"{persona}_queries"] += 1
    _stats[f"{result.get('route','rag')}_routes"] += 1
    if result.get("uncertain"):
        _stats["uncertain_count"] += 1
    sim = float(result.get("similarity", 0.0))
    _stats["similarity_sum"] += sim
    _stats["response_times_ms"].append(elapsed_ms)
    if sim < 0.4 and result.get("question"):
        _stats["low_similarity_questions"].append(result["question"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PipelineRunRequest(BaseModel):
    urls: list[str]

class ChatRequest(BaseModel):
    question: str

class DebateRequest(BaseModel):
    topic: str
    rounds: int = 3

class JudgeRequest(BaseModel):
    question: str
    supporter_response: str
    critic_response: str

class JudgeLogRequest(BaseModel):
    question: str
    persona: str   # "supporter" or "critic"
    response: str

class TopicLabelRequest(BaseModel):
    words: list[str]
    sentiment: str = ""  # "pro" or "anti", optional context

class ChatResponse(BaseModel):
    persona: str
    question: str
    response: str
    similarity: float
    uncertain: bool
    sources: list[str]
    source_texts: list[str]
    route: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re as _re

def _extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    for pattern in [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]:
        m = _re.search(pattern, s)
        if m:
            return m.group(1)
    if _re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s
    raise ValueError(f"Cannot parse video ID from: {s}")

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/setup")
def setup_page():
    return FileResponse("static/setup.html")

@app.get("/progress")
def progress_page():
    return FileResponse("static/progress.html")

# ---------------------------------------------------------------------------
# API — health
# ---------------------------------------------------------------------------

def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def _weaviate_collections():
    try:
        import weaviate
        client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "weaviate"),
            port=int(os.getenv("WEAVIATE_PORT", 8080)),
        )
        pos = client.collections.exists("PositiveComments")
        neg = client.collections.exists("NegativeComments")
        client.close()
        return {"positive_exists": pos, "negative_exists": neg}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/health")
def api_health():
    ollama_url   = os.getenv("OLLAMA_URL", "http://ollama:11434")
    weaviate_url = f"http://{os.getenv('WEAVIATE_HOST','weaviate')}:{os.getenv('WEAVIATE_PORT',8080)}/v1/.well-known/ready"
    mlflow_url   = "http://mlflow:5001"
    neo4j_url    = "http://neo4j:7474"

    weaviate_up = _probe(weaviate_url)
    ollama_up   = _probe(ollama_url)
    neo4j_up    = _probe(neo4j_url)
    mlflow_up   = _probe(mlflow_url)

    return {
        "timestamp": int(time.time()),
        "services": {
            "weaviate": {"status": "online" if weaviate_up else "offline", "port": 8080},
            "ollama":   {"status": "online" if ollama_up   else "offline", "port": 11434},
            "neo4j":    {"status": "online" if neo4j_up    else "offline", "port": 7474},
            "mlflow":   {"status": "online" if mlflow_up   else "offline", "port": 5001},
            "chatbot":  {"status": "online", "port": 8000},
        },
        "collections": _weaviate_collections() if weaviate_up else {},
    }

# ---------------------------------------------------------------------------
# API — evaluation metrics
# ---------------------------------------------------------------------------

@app.get("/api/eval")
def api_eval():
    t = _stats["total_queries"]
    times = _stats["response_times_ms"]

    avg_sim = round(_stats["similarity_sum"] / t, 3) if t else 0.0
    uncertain_rate = round(_stats["uncertain_count"] / t * 100, 1) if t else 0.0
    rag_pct  = round(_stats["rag_routes"]   / t * 100, 1) if t else 0.0
    graph_pct= round(_stats["graph_routes"] / t * 100, 1) if t else 0.0
    avg_ms   = round(sum(times) / len(times), 0) if times else 0
    p95_ms   = round(sorted(times)[int(len(times) * 0.95)], 0) if len(times) >= 2 else avg_ms

    return {
        "total_queries": t,
        "by_persona": {
            "supporter": _stats["supporter_queries"],
            "critic":    _stats["critic_queries"],
        },
        "routing": {
            "rag":   {"count": _stats["rag_routes"],   "pct": rag_pct},
            "graph": {"count": _stats["graph_routes"],  "pct": graph_pct},
        },
        "similarity": {
            "avg":           avg_sim,
            "uncertain_count": _stats["uncertain_count"],
            "uncertain_rate_pct": uncertain_rate,
        },
        "latency_ms": {
            "avg": avg_ms,
            "p95": p95_ms,
        },
        "low_similarity_questions": _stats["low_similarity_questions"][-5:],
    }

# ---------------------------------------------------------------------------
# API — pipeline
# ---------------------------------------------------------------------------

_pipeline = {"status": "idle", "stage": "", "error": ""}

def _run_pipeline(urls: list[str]):
    pipeline_dir = "/app/pipeline_data"
    dataset_dir = "/dataset"
    weaviate_dir = "/app/weaviate_app"
    env = {**os.environ}
    try:
        _pipeline.update({"status": "running", "stage": "scraping", "error": ""})
        r = subprocess.run(
            ["python", "0_scraper.py", "--urls", ",".join(urls)],
            cwd=pipeline_dir, capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

        _pipeline["stage"] = "preprocessing"
        r = subprocess.run(
            ["python", "1_preprocessing.py", "--input", "comments.csv", "--output", "processed_comments.csv"],
            cwd=pipeline_dir, capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

        _pipeline["stage"] = "analyzing"
        r = subprocess.run(
            ["python", "2_analysis.py", "--input", "processed_comments.csv", "--output", "analysis_results.csv"],
            cwd=pipeline_dir, capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

        # Copy CSV to /dataset so embedding.py can read it
        _pipeline["stage"] = "embedding"
        os.makedirs(dataset_dir, exist_ok=True)
        shutil.copy(f"{pipeline_dir}/analysis_results.csv", f"{dataset_dir}/analysis_results.csv")

        r = subprocess.run(
            ["python", "embedding.py"],
            cwd=weaviate_dir, capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

        _pipeline["stage"] = "vectorizing"
        r = subprocess.run(
            ["python", "vectorize.py"],
            cwd=weaviate_dir, capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

        _pipeline.update({"status": "complete", "stage": "done", "last_run": int(time.time())})
    except Exception as e:
        _pipeline.update({"status": "error", "stage": "failed", "error": str(e)[:300]})


@app.post("/api/pipeline/validate")
def api_pipeline_validate(req: PipelineRunRequest):
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="YOUTUBE_API_KEY not configured in .env.")
    try:
        from googleapiclient.discovery import build as _yt_build
        youtube = _yt_build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"YouTube client error: {e}")

    errors: dict[str, str] = {}
    valid_ids: list[str] = []
    for raw in req.urls:
        try:
            vid = _extract_video_id(raw)
        except ValueError:
            errors[raw] = "Not a valid YouTube URL or video ID."
            continue
        try:
            resp = youtube.videos().list(part="id,status,contentDetails", id=vid).execute()
            items = resp.get("items", [])
            if not items:
                errors[raw] = "Video not found — it may be private, deleted, or the ID is wrong."
            else:
                privacy = items[0].get("status", {}).get("privacyStatus", "")
                if privacy == "private":
                    errors[raw] = "Video is private."
                else:
                    valid_ids.append(vid)
        except Exception as e:
            errors[raw] = f"YouTube API error: {str(e)[:120]}"

    return {"valid": len(errors) == 0, "valid_ids": valid_ids, "errors": errors}


@app.post("/api/pipeline/run")
def api_pipeline_run(req: PipelineRunRequest):
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
    if _pipeline["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running.")
    threading.Thread(target=_run_pipeline, args=(req.urls,), daemon=True).start()
    return {"status": "started"}


@app.get("/api/pipeline/status")
def api_pipeline_status():
    return _pipeline


# ---------------------------------------------------------------------------
# API — analysis
# ---------------------------------------------------------------------------

@app.get("/api/analysis")
def api_analysis():
    data = load_analysis()
    if data is None:
        raise HTTPException(status_code=503, detail="Pipeline data not available. Check PIPELINE_DATA_PATH.")
    return data

# ---------------------------------------------------------------------------
# API — chat
# ---------------------------------------------------------------------------

@app.post("/chat/supporter", response_model=ChatResponse)
def chat_supporter(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    t0 = time.time()
    result = run_router(question=req.question, persona="supporter")
    elapsed = (time.time() - t0) * 1000
    result["question"] = req.question
    _record("supporter", result, elapsed)
    return ChatResponse(
        persona="supporter", question=req.question,
        response=result.get("response", ""),
        similarity=float(result.get("similarity", 0.0)),
        uncertain=bool(result.get("uncertain", False)),
        sources=result.get("sources", []),
        source_texts=result.get("source_texts", []),
        route=result.get("route", "rag"),
    )


@app.post("/chat/critic", response_model=ChatResponse)
def chat_critic(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    t0 = time.time()
    result = run_router(question=req.question, persona="critic")
    elapsed = (time.time() - t0) * 1000
    result["question"] = req.question
    _record("critic", result, elapsed)
    return ChatResponse(
        persona="critic", question=req.question,
        response=result.get("response", ""),
        similarity=float(result.get("similarity", 0.0)),
        uncertain=bool(result.get("uncertain", False)),
        sources=result.get("sources", []),
        source_texts=result.get("source_texts", []),
        route=result.get("route", "rag"),
    )


@app.post("/api/judge/auto")
def api_judge_auto(req: JudgeLogRequest):
    """Log this side's latest response. Judge runs whenever both sides have responded at least once."""
    if req.persona not in ("supporter", "critic"):
        raise HTTPException(status_code=400, detail="persona must be supporter or critic")
    _judge_latest[req.persona] = {"question": req.question.strip(), "response": req.response}
    s = _judge_latest.get("supporter")
    c = _judge_latest.get("critic")
    if s and c:
        threading.Thread(target=_auto_judge, args=(s["question"], s["response"], c["question"], c["response"]), daemon=True).start()
        return {"status": "judging"}
    return {"status": "waiting"}

@app.get("/api/judge/history")
def api_judge_history():
    return list(reversed(_judge_history))  # newest first

@app.get("/boxing")
def boxing_page():
    return FileResponse("static/boxing.html")

@app.post("/api/debate")
def api_debate(req: DebateRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if not (1 <= req.rounds <= 5):
        raise HTTPException(status_code=400, detail="Rounds must be between 1 and 5.")
    return run_debate(topic=req.topic, rounds=req.rounds)

@app.post("/api/debate/stream")
def api_debate_stream(req: DebateRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if not (1 <= req.rounds <= 5):
        raise HTTPException(status_code=400, detail="Rounds must be between 1 and 5.")
    def generate():
        for entry in stream_debate(topic=req.topic, rounds=req.rounds):
            yield f"data: {_json.dumps(entry)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/judge")
def api_judge(req: JudgeRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return run_judge(
        question=req.question,
        supporter_response=req.supporter_response,
        critic_response=req.critic_response,
    )


@app.post("/api/label-topic")
def api_label_topic(req: TopicLabelRequest):
    """Use the LLM to generate a short label for a topic given its top words."""
    from debate import lm
    import dspy

    words_str = ", ".join(req.words[:6])
    prompt = (
        f"Words: {words_str}\n"
        f"Write a 2-3 word topic label for these words. Examples: 'Trade Policy', 'Religious Support', 'Election Fraud'. "
        f"Reply with ONLY the label."
    )
    with dspy.context(lm=lm):
        result = lm(messages=[{"role": "user", "content": prompt}])
    raw = (result[0] if isinstance(result, list) else str(result)).strip().strip('"').strip("'")
    # take only the first line in case model adds explanation
    label = raw.split("\n")[0].strip()[:40]
    return {"label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
