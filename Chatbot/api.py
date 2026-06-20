import warnings
warnings.filterwarnings("ignore")

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from NEO4j.router import run_router

load_dotenv()

app = FastAPI(
    title="CampaignLens Chatbot API",
    description="Supporter and Critic chatbots grounded in real YouTube comments.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    persona: str
    question: str
    response: str
    confidence: float
    uncertain: bool
    sources: list[str]
    route: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "CampaignLens API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat/supporter", response_model=ChatResponse)
def chat_supporter(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_router(question=req.question, persona="supporter")

    return ChatResponse(
        persona="supporter",
        question=req.question,
        response=result["response"],
        confidence=result["confidence"],
        uncertain=result["uncertain"],
        sources=result["sources"],
        route=result["route"]
    )


@app.post("/chat/critic", response_model=ChatResponse)
def chat_critic(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_router(question=req.question, persona="critic")

    return ChatResponse(
        persona="critic",
        question=req.question,
        response=result["response"],
        confidence=result["confidence"],
        uncertain=result["uncertain"],
        sources=result["sources"],
        route=result["route"]
    )


# ---------------------------------------------------------------------------
# Run directly: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
