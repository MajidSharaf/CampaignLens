import os
import sys
import mlflow
import weaviate

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# chatbot.py and retrieval.py live one level up in Chatbot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import SupporterChatbot, CriticChatbot, run_chatbot
from NEO4j.query import run_graph_query
from retrieval import get_weaviate_client

load_dotenv()

# ---------------------------------------------------------------------------
# MLflow — same pattern as lab files
# ---------------------------------------------------------------------------

mlflow.set_experiment("CampaignLens_Router")
mlflow.dspy.autolog()

# ---------------------------------------------------------------------------
# LLM for classification only
# ChatOllama used here (not DSPy) because the lab classifier pattern
# uses llm.invoke() directly — keeps it simple for a routing decision.
# Same model, temperature=0 so routing is deterministic.
# ---------------------------------------------------------------------------

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
    base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    temperature=0
)

# ---------------------------------------------------------------------------
# Weaviate collections — opened once, reused across requests
# ---------------------------------------------------------------------------

_client = None
_positive = None
_negative = None

def get_collections():
    global _client, _positive, _negative
    if _client is None:
        _client = get_weaviate_client()
        _positive = _client.collections.get("PositiveComments")
        _negative = _client.collections.get("NegativeComments")
    return _positive, _negative

# ---------------------------------------------------------------------------
# Chatbot instances — one of each, reused
# ---------------------------------------------------------------------------

supporter = SupporterChatbot()
critic = CriticChatbot()

# ---------------------------------------------------------------------------
# State
# Mirrors the AgentState pattern from 3_LangGraph_MultiAgent.py.
# Carries the question and persona through the graph,
# and collects the final result.
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    question: str
    persona: str        # "supporter" or "critic" — set by the API before invoking
    route: str          # "rag" or "graph" — set by the classifier node
    response: str
    confidence: float
    sources: list
    uncertain: bool

# ---------------------------------------------------------------------------
# Node 1 — Classifier
# Decides whether the question is better answered by RAG (comment retrieval)
# or graph query (entity relationships in Neo4j).
# Follows classify_email pattern from 3_LangGraph_MultiAgent.py.
#
# RAG questions: "what do supporters think about X", "how do people feel about Y"
# Graph questions: "what is the relationship between X and Y", "how is X connected to Z"
# ---------------------------------------------------------------------------

def classify_question(state: RouterState):
    prompt = f"""You are a question classifier for a political sentiment analysis system.

Question: {state['question']}

Classify into ONE category only:
- rag: the question asks about opinions, feelings, or what people think
- graph: the question asks about relationships, connections, or entities

Return only the category name: rag or graph"""

    result = llm.invoke(prompt)
    route = result.content.strip().lower()

    # fallback to rag if model returns something unexpected
    if "graph" not in route:
        route = "rag"

    return {"route": route}


# ---------------------------------------------------------------------------
# Node 2a — RAG node
# Calls the right chatbot based on persona.
# Follows the agent node pattern from 3_LangGraph_MultiAgent.py.
# ---------------------------------------------------------------------------

def rag_node(state: RouterState):
    positive, negative = get_collections()

    collection = positive if state["persona"] == "supporter" else negative

    result = run_chatbot(
        chatbot=supporter if state["persona"] == "supporter" else critic,
        collection=collection,
        question=state["question"]
    )

    return {
        "response": result["response"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "uncertain": result["uncertain"]
    }


# ---------------------------------------------------------------------------
# Node 2b — Graph node
# Calls Neo4j query regardless of persona — the graph has all comments.
# ---------------------------------------------------------------------------

def graph_node(state: RouterState):
    result = run_graph_query(state["question"])

    return {
        "response": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "uncertain": result["uncertain"]
    }


# ---------------------------------------------------------------------------
# Router function
# Maps the classified route to the next node.
# Same pattern as route_email in 3_LangGraph_MultiAgent.py.
# ---------------------------------------------------------------------------

def route_question(state: RouterState):
    if state["route"] == "graph":
        return "graph_node"
    return "rag_node"


# ---------------------------------------------------------------------------
# Build graph
# START -> classifier -> conditional edge -> rag_node or graph_node -> END
# ---------------------------------------------------------------------------

builder = StateGraph(RouterState)

builder.add_node("classifier", classify_question)
builder.add_node("rag_node", rag_node)
builder.add_node("graph_node", graph_node)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier",
    route_question,
    {
        "rag_node": "rag_node",
        "graph_node": "graph_node"
    }
)
builder.add_edge("rag_node", END)
builder.add_edge("graph_node", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Main function — what api.py calls
# Takes question + persona, runs the graph, returns result dict.
# ---------------------------------------------------------------------------

def run_router(question: str, persona: str):
    """
    Args:
        question: user's question string
        persona:  "supporter" or "critic"

    Returns:
        dict with keys: response, confidence, sources, uncertain, route
    """
    with mlflow.start_run(run_name=f"router_{persona}", nested=True):
        result = graph.invoke({
            "question": question,
            "persona": persona,
            "route": "",
            "response": "",
            "confidence": 0.0,
            "sources": [],
            "uncertain": False
        })

    return {
        "response": result["response"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "uncertain": result["uncertain"],
        "route": result["route"]
    }


# ---------------------------------------------------------------------------
# Run directly to test
# python router.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("What do supporters think about Trump's economy?", "supporter"),
        ("What do critics say about immigration?", "critic"),
        ("What is the relationship between Trump and Russia in the comments?", "supporter"),
    ]

    for question, persona in test_cases:
        print(f"\nPersona:  {persona}")
        print(f"Question: {question}")
        result = run_router(question, persona)
        print(f"Route:      {result['route']}")
        print(f"Response:   {result['response']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources'][:3]}")