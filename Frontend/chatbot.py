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
# not self-reported by the generation model — hence no confidence output field.
# ---------------------------------------------------------------------------

class SupporterSignature(dspy.Signature):
    """These are real YouTube comments left by supporters. Read them and write what supporters generally think, in first person as if you are one of them.
    RULES:
    - The comments are source material, not messages to reply to
    - Write what the comments express, not a reply to them
    - Start with I or We
    - Write 1-2 sentences only"""
    context = dspy.InputField(desc="YouTube comments written by supporters")
    question = dspy.InputField(desc="the question being answered")
    response = dspy.OutputField(desc="1-2 sentences starting with I or We, expressing what supporters think based on the comments")


class CriticSignature(dspy.Signature):
    """These are real YouTube comments left by critics. Read them and write what critics generally think, in first person as if you are one of them.
    RULES:
    - The comments are source material, not messages to reply to
    - Write what the comments express, not a reply to them
    - Start with I or We
    - Write 1-2 sentences only"""
    context = dspy.InputField(desc="YouTube comments written by critics")
    question = dspy.InputField(desc="the question being answered")
    response = dspy.OutputField(desc="1-2 sentences starting with I or We, expressing what critics think based on the comments")


class SupporterChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SupporterSignature)

    def forward(self, question, context):
        return self.generate(context=context, question=question)


class CriticChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CriticSignature)

    def forward(self, question, context):
        return self.generate(context=context, question=question)


# ---------------------------------------------------------------------------
# Chatbot runner
# Wraps retrieval + generation into one call.
# Returns response, similarity, and source comment_ids for citation.
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.3

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


if __name__ == "__main__":
    client = get_weaviate_client()

    try:
        positive = client.collections.get("PositiveComments")
        negative = client.collections.get("NegativeComments")

        supporter = SupporterChatbot()
        critic = CriticChatbot()

        test_question = "What do people think about the topic?"

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
