import warnings
warnings.filterwarnings("ignore")

import os
import weaviate
import dspy
from dotenv import load_dotenv
from retrieval import retrieve, get_weaviate_client, rerank_score_to_similarity

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
# Similarity is derived from the reranker's relevance score (see run_chatbot),
# not self-reported by the generation model.
# ---------------------------------------------------------------------------

class SupporterSignature(dspy.Signature):
    """Rewrite the provided texts as a single first-person response. Use only what is in the texts."""
    context = dspy.InputField(desc="texts to rewrite")
    response = dspy.OutputField(desc="2-3 sentences in first person combining the texts. Start with I or We.")


class CriticSignature(dspy.Signature):
    """Rewrite the provided texts as a single first-person response. Use only what is in the texts."""
    context = dspy.InputField(desc="texts to rewrite")
    response = dspy.OutputField(desc="2-3 sentences in first person combining the texts. Start with I or We.")

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
        return self.generate(context=context)


class CriticChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CriticSignature)

    def forward(self, question, context):
        return self.generate(context=context)


# ---------------------------------------------------------------------------
# Chatbot runner
# Wraps retrieval + generation into one call.
# Returns response, similarity, and source comment_ids for citation.
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.6

def run_chatbot(chatbot, collection, question, k=10, top_k=5):
    """
    Args:
        chatbot:    SupporterChatbot or CriticChatbot instance
        collection: weaviate collection object (PositiveComments or NegativeComments)
        question:   user's question string
        k:          candidates to pull from Weaviate before reranking
        top_k:      final number of comments passed as context

    Returns:
        dict with keys: response, similarity, sources, uncertain
    """
    # Step 1 — retrieve and rerank
    context, sources, reranked = retrieve(collection, question, k=k, top_k=top_k)

    if not context:
        return {
            "response": "No relevant comments were found in the dataset for this question.",
            "similarity": 0.0,
            "sources": [],
            "uncertain": True
        }

    # Step 2 — generate grounded response
    with dspy.context(lm=lm):
        result = chatbot(question=question, context=context)

    # Step 3 — similarity = average reranker relevance of the comments actually
    # used as context, not a self-reported number from the generation model.
    avg_score = sum(score for _, score, _ in reranked) / len(reranked)
    similarity = rerank_score_to_similarity(avg_score)

    uncertain = similarity < SIMILARITY_THRESHOLD

    source_texts = [text for text, score, cid in reranked]

    return {
        "response": result.response,
        "similarity": round(similarity, 2),
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
        print(f"Similarity: {result['similarity']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources']}")

        print("\n========== CRITIC ==========")
        result = run_chatbot(critic, negative, test_question)
        print(f"Response:   {result['response']}")
        print(f"Similarity: {result['similarity']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources']}")

    finally:
        client.close()