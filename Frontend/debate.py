"""
debate.py — Boxing debate engine and LLM judge.

Debate: Supporter and Critic take turns responding to each other,
each grounded in their RAG context. Each round produces a confidence score.

Judge: A separate LLM evaluates a Q&A pair (question + both chatbot responses)
and returns a structured verdict.
"""

import os
import dspy
from dotenv import load_dotenv
from retrieval import retrieve, get_weaviate_client

load_dotenv()

lm = dspy.LM(
    f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen2.5:0.5b')}",
    api_base=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    api_key=None,
    temperature=0.7,
    cache=False,
)
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

class DebateRoundSignature(dspy.Signature):
    """You are in a live debate. The critic attacks first, the supporter defends.
    If you are the critic: open with a sharp attack on the topic, grounded in the comments.
    If you are the supporter: directly counter what the critic just said, then defend your position.
    Use only the retrieved comments as evidence. Be aggressive and opinionated — 2-3 sentences max."""
    topic = dspy.InputField(desc="the debate topic")
    persona = dspy.InputField(desc="your role: 'critic' (attacker) or 'supporter' (defender)")
    opponent_last = dspy.InputField(desc="what the opponent just said (empty for critic's opening attack)")
    context = dspy.InputField(desc="retrieved comments supporting your position")
    response = dspy.OutputField(desc="your sharp, opinionated debate statement grounded in the context")
    confidence = dspy.OutputField(desc="float 0-1: how well the context supports your statement")


class JudgeSignature(dspy.Signature):
    """You are an impartial judge evaluating two chatbot responses to the same question.
    Assess factual grounding, clarity, and how well each response addresses the question.
    Be objective and specific."""
    question = dspy.InputField(desc="the user's question")
    supporter_response = dspy.InputField(desc="response from the supporter chatbot")
    critic_response = dspy.InputField(desc="response from the critic chatbot")
    supporter_score = dspy.OutputField(desc="float 0-1: quality score for the supporter response")
    critic_score = dspy.OutputField(desc="float 0-1: quality score for the critic response")
    supporter_feedback = dspy.OutputField(desc="one sentence: what the supporter response did well or poorly")
    critic_feedback = dspy.OutputField(desc="one sentence: what the critic response did well or poorly")
    verdict = dspy.OutputField(desc="one sentence: which response was stronger and why, or if they were equal")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class DebateRound(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(DebateRoundSignature)

    def forward(self, topic, persona, opponent_last, context):
        return self.generate(
            topic=topic,
            persona=persona,
            opponent_last=opponent_last,
            context=context,
        )


class Judge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(JudgeSignature)

    def forward(self, question, supporter_response, critic_response):
        return self.evaluate(
            question=question,
            supporter_response=supporter_response,
            critic_response=critic_response,
        )


_debate_module = DebateRound()
_judge_module = Judge()

# ---------------------------------------------------------------------------
# Weaviate collections (lazy)
# ---------------------------------------------------------------------------

_client = None
_positive = None
_negative = None

def _get_collections():
    global _client, _positive, _negative
    if _client is None:
        _client = get_weaviate_client()
        _positive = _client.collections.get("PositiveComments")
        _negative = _client.collections.get("NegativeComments")
    return _positive, _negative


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_debate(topic: str, rounds: int = 3):
    """Generator that yields one result dict per turn (critic then supporter)."""
    positive, negative = _get_collections()
    supporter_last = ""
    critic_last = ""

    for i in range(rounds):
        # Critic attacks
        context, sources, _ = retrieve(negative, f"{topic} {supporter_last}".strip(), k=10, top_k=5)
        with dspy.context(lm=lm):
            c_out = _debate_module(topic=topic, persona="critic", opponent_last=supporter_last,
                                   context=context or "No relevant comments found.")
        try:
            c_conf = round(float(c_out.confidence), 2)
        except (ValueError, TypeError):
            c_conf = 0.0
        critic_last = c_out.response
        yield {"round": i + 1, "persona": "critic", "response": c_out.response, "confidence": c_conf, "sources": sources}

        # Supporter defends
        context, sources, _ = retrieve(positive, f"{topic} {critic_last}".strip(), k=10, top_k=5)
        with dspy.context(lm=lm):
            s_out = _debate_module(topic=topic, persona="supporter", opponent_last=critic_last,
                                   context=context or "No relevant comments found.")
        try:
            s_conf = round(float(s_out.confidence), 2)
        except (ValueError, TypeError):
            s_conf = 0.0
        supporter_last = s_out.response
        yield {"round": i + 1, "persona": "supporter", "response": s_out.response, "confidence": s_conf, "sources": sources}


def run_debate(topic: str, rounds: int = 3) -> dict:
    """
    Run a multi-round debate between the supporter and critic.

    Returns:
        {
            "topic": str,
            "rounds": [
                {
                    "round": int,
                    "persona": "supporter" | "critic",
                    "response": str,
                    "confidence": float,
                    "sources": list[str]
                },
                ...
            ]
        }
    """
    positive, negative = _get_collections()

    results = []
    supporter_last = ""
    critic_last = ""

    for i in range(rounds):
        # --- Critic attacks first ---
        context, sources, _ = retrieve(negative, f"{topic} {supporter_last}".strip(), k=10, top_k=5)

        with dspy.context(lm=lm):
            c_out = _debate_module(
                topic=topic,
                persona="critic",
                opponent_last=supporter_last,  # empty on round 1 → opening attack
                context=context or "No relevant comments found.",
            )

        try:
            c_conf = round(float(c_out.confidence), 2)
        except (ValueError, TypeError):
            c_conf = 0.0

        critic_last = c_out.response
        results.append({
            "round": i + 1,
            "persona": "critic",
            "response": c_out.response,
            "confidence": c_conf,
            "sources": sources,
        })

        # --- Supporter defends ---
        context, sources, _ = retrieve(positive, f"{topic} {critic_last}".strip(), k=10, top_k=5)

        with dspy.context(lm=lm):
            s_out = _debate_module(
                topic=topic,
                persona="supporter",
                opponent_last=critic_last,
                context=context or "No relevant comments found.",
            )

        try:
            s_conf = round(float(s_out.confidence), 2)
        except (ValueError, TypeError):
            s_conf = 0.0

        supporter_last = s_out.response
        results.append({
            "round": i + 1,
            "persona": "supporter",
            "response": s_out.response,
            "confidence": s_conf,
            "sources": sources,
        })

    return {"topic": topic, "rounds": results}


def run_judge(question: str, supporter_response: str, critic_response: str) -> dict:
    """
    Judge a Q&A pair from the main chat.

    Returns:
        {
            "supporter_score": float,
            "critic_score": float,
            "supporter_feedback": str,
            "critic_feedback": str,
            "verdict": str
        }
    """
    with dspy.context(lm=lm):
        result = _judge_module(
            question=question,
            supporter_response=supporter_response,
            critic_response=critic_response,
        )

    def _parse_score(v):
        try:
            return round(min(1.0, max(0.0, float(v))), 2)
        except (ValueError, TypeError):
            return 0.0

    return {
        "supporter_score": _parse_score(result.supporter_score),
        "critic_score": _parse_score(result.critic_score),
        "supporter_feedback": result.supporter_feedback,
        "critic_feedback": result.critic_feedback,
        "verdict": result.verdict,
    }
