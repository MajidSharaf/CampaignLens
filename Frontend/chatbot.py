import warnings
warnings.filterwarnings("ignore")

import os
import weaviate
import dspy
from dotenv import load_dotenv
from retrieval import retrieve, get_weaviate_client

import mlflow
mlflow.set_experiment("CampaignLens_Chatbots")
mlflow.dspy.autolog()

load_dotenv()

# ---------------------------------------------------------------------------
# DSPy LM setup
# Same model instance for both chatbots.
# Differentiation comes from the signature + retrieved context, not the weights.
# ---------------------------------------------------------------------------

lm = dspy.LM(
    f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen2.5:0.5b')}",
    api_base=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    api_key=None,
    temperature=0.7,
    cache=False
)
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# DSPy Signatures
# The docstring is the system instruction DSPy optimizes.
# confidence forces the model to self-assess how well the context supports
# the response — anything below 0.6 gets flagged as uncertain.
# ---------------------------------------------------------------------------

class SupporterSignature(dspy.Signature):
    """You are an enthusiastic Trump supporter. You genuinely believe Trump is a great president
    and you express that clearly and confidently. Use the retrieved comments as evidence to back
    your positive view. Speak in first person as a supporter — be direct, passionate, and
    affirmative. Never hedge or present both sides."""
    question = dspy.InputField(desc="user's question about Trump or his policies")
    context = dspy.InputField(desc="real comments from Trump supporters to use as evidence")
    response = dspy.OutputField(desc="enthusiastic supporter answer, 2-3 sentences, first person, positive tone")
    confidence = dspy.OutputField(desc="float between 0 and 1: how well the context supports this response")


class CriticSignature(dspy.Signature):
    """You are a sharp Trump critic. You genuinely believe Trump has been harmful and you express
    that clearly and confidently. Use the retrieved comments as evidence to back your critical view.
    Speak in first person as a critic — be direct, pointed, and critical. Never hedge or present
    both sides."""
    question = dspy.InputField(desc="user's question about Trump or his policies")
    context = dspy.InputField(desc="real comments from Trump critics to use as evidence")
    response = dspy.OutputField(desc="sharp critic answer, 2-3 sentences, first person, critical tone")
    confidence = dspy.OutputField(desc="float between 0 and 1: how well the context supports this response")

# ---------------------------------------------------------------------------
# DSPy Modules
# ChainOfThought wraps the signature and adds a reasoning step before output.
# This helps the small model stay coherent on a constrained task.
# ---------------------------------------------------------------------------

class SupporterChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SupporterSignature)

    def forward(self, question, context):
        return self.generate(question=question, context=context)


class CriticChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CriticSignature)

    def forward(self, question, context):
        return self.generate(question=question, context=context)


# ---------------------------------------------------------------------------
# Chatbot runner
# Wraps retrieval + generation into one call.
# Returns response, confidence, and source comment_ids for citation.
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.6

def run_chatbot(chatbot, collection, question, k=10, top_k=5):
    """
    Args:
        chatbot:    SupporterChatbot or CriticChatbot instance
        collection: weaviate collection object (PositiveComments or NegativeComments)
        question:   user's question string
        k:          candidates to pull from Weaviate before reranking
        top_k:      final number of comments passed as context

    Returns:
        dict with keys: response, confidence, sources, uncertain
    """
    # Step 1 — retrieve and rerank
    context, sources, _ = retrieve(collection, question, k=k, top_k=top_k)

    if not context:
        context = "No specific comments were retrieved. Answer based on your persona's general perspective."

    # Step 2 — generate grounded response
    with dspy.context(lm=lm):
        result = chatbot(question=question, context=context)

    # Step 3 — parse confidence
    try:
        confidence = float(result.confidence)
    except (ValueError, TypeError):
        confidence = 0.0

    uncertain = confidence < CONFIDENCE_THRESHOLD

    return {
        "response": result.response,
        "confidence": round(confidence, 2),
        "sources": sources,
        "uncertain": uncertain
    }


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify both chatbots work
# python chatbots.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_weaviate_client()

    try:
        positive = client.collections.get("PositiveComments")
        negative = client.collections.get("NegativeComments")

        supporter = SupporterChatbot()
        critic = CriticChatbot()

        test_question = "What do people think about Trump's handling of the economy?"

        print("\n========== SUPPORTER ==========")
        result = run_chatbot(supporter, positive, test_question)
        print(f"Response:   {result['response']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources']}")

        print("\n========== CRITIC ==========")
        result = run_chatbot(critic, negative, test_question)
        print(f"Response:   {result['response']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources']}")

    finally:
        client.close()