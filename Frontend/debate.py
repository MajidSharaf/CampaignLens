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
    """You are in a live debate grounded ONLY in real YouTube comments provided below.
    Do NOT use outside knowledge — every claim must come directly from the comments in context.
    Paraphrase or quote what real people said. If you are the critic: attack using what critics said.
    If you are the supporter: defend using what supporters said and rebut the opponent.
    Be direct and opinionated. 2-3 sentences max."""
    topic = dspy.InputField(desc="the debate topic")
    persona = dspy.InputField(desc="your role: 'critic' (attacker) or 'supporter' (defender)")
    opponent_last = dspy.InputField(desc="what the opponent just said (empty for critic's opening attack)")
    context = dspy.InputField(desc="real YouTube comments you MUST use as your only evidence")
    response = dspy.OutputField(desc="debate statement using only what real commenters said")
    confidence = dspy.OutputField(desc="float 0-1: how well the context supports your statement")


def _debate_turn(persona: str, topic: str, opponent_last: str, context: str) -> tuple[str, float]:
    """Call the LLM directly with a simple prompt to avoid structured output parsing issues."""
    role = "Trump critic" if persona == "critic" else "Trump supporter"
    rebut = f'\nThe opponent just said: "{opponent_last}"\nDirectly counter that, then make your point.' if opponent_last else ""
    prompt = (
        f"You are a {role} in a debate about: {topic}\n"
        f"Use ONLY these real YouTube comments as evidence:\n{context}\n"
        f"{rebut}\n"
        f"Write 2-3 sharp sentences from the {role}'s perspective, grounded in those comments. "
        f"Reply with ONLY your statement, nothing else."
    )
    with dspy.context(lm=lm):
        result = lm(messages=[{"role": "user", "content": prompt}])
    text = (result[0] if isinstance(result, list) else str(result)).strip()
    return text, 0.8


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
        c_response, c_conf = _debate_turn("critic", topic, supporter_last, context or "No relevant comments found.")
        critic_last = c_response
        yield {"round": i + 1, "persona": "critic", "response": c_response, "confidence": c_conf, "sources": sources}

        # Supporter defends
        context, sources, _ = retrieve(positive, f"{topic} {critic_last}".strip(), k=10, top_k=5)
        s_response, s_conf = _debate_turn("supporter", topic, critic_last, context or "No relevant comments found.")
        supporter_last = s_response
        yield {"round": i + 1, "persona": "supporter", "response": s_response, "confidence": s_conf, "sources": sources}


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

        c_response, c_conf = _debate_turn("critic", topic, supporter_last, context or "No relevant comments found.")
        critic_last = c_response
        results.append({"round": i + 1, "persona": "critic", "response": c_response, "confidence": c_conf, "sources": sources})

        # --- Supporter defends ---
        context, sources, _ = retrieve(positive, f"{topic} {critic_last}".strip(), k=10, top_k=5)
        s_response, s_conf = _debate_turn("supporter", topic, critic_last, context or "No relevant comments found.")
        supporter_last = s_response
        results.append({"round": i + 1, "persona": "supporter", "response": s_response, "confidence": s_conf, "sources": sources})

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
