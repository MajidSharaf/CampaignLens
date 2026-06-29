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
    """You are roleplaying as a Trump supporter based ONLY on the YouTube comments provided.
    Speak in first person as if you wrote those comments. Use 'I' and 'we'.
    Only use what is in the comments. Do not add outside knowledge."""
    question = dspy.InputField(desc="question about Trump or his policies")
    context = dspy.InputField(desc="real YouTube comments from Trump supporters — your ONLY source")
    response = dspy.OutputField(desc="2-3 sentences in first person, drawn entirely from the provided comments. Start with 'I' or 'We'.")
    confidence = dspy.OutputField(desc="float between 0 and 1: how well the context supports this response")


class CriticSignature(dspy.Signature):
    """You are roleplaying as a person who disagrees with Trump based ONLY on the YouTube comments provided.
    Speak in first person as if you wrote those comments. Use 'I' and 'we'.
    Only use what is in the comments. Do not add outside knowledge."""
    question = dspy.InputField(desc="question about Trump or his policies")
    context = dspy.InputField(desc="real YouTube comments from people who disagree with Trump — your ONLY source")
    response = dspy.OutputField(desc="2-3 sentences in first person, drawn entirely from the provided comments. Start with 'I' or 'We'.")
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
    context, sources, reranked = retrieve(collection, question, k=k, top_k=top_k)

    if not context:
        return {
            "response": "No relevant comments were found in the dataset for this question.",
            "confidence": 0.0,
            "sources": [],
            "uncertain": True
        }

    # Step 2 — generate grounded response
    with dspy.context(lm=lm):
        result = chatbot(question=question, context=context)

    # Step 3 — parse confidence
    try:
        confidence = float(result.confidence)
    except (ValueError, TypeError):
        confidence = 0.0

    uncertain = confidence < CONFIDENCE_THRESHOLD

    source_texts = [text for text, score, cid in reranked]

    return {
        "response": result.response,
        "confidence": round(confidence, 2),
        "sources": sources,
        "source_texts": source_texts,
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