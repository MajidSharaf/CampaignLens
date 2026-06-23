import weaviate
from weaviate.classes import query as wq
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url=os.getenv("OLLAMA_URL", "http://ollama:11434")
)

def get_weaviate_client():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "weaviate"),
        port=int(os.getenv("WEAVIATE_PORT", 8080))
    )

# ---------------------------------------------------------------------------
# Hybrid retrieval
# Combines vector search (semantic) and BM25 (keyword) in one query.
# alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25, alpha=0.5 -> balanced.
# Returns two things: scored tuples for reranking, plain texts for context.
# ---------------------------------------------------------------------------

def hybridRetrieval(collection, question, k=10, alpha=0.5):
    """
    Args:
        collection: weaviate collection object (PositiveComments or NegativeComments)
        question:   user's question string
        k:          number of results to retrieve before reranking
        alpha:      0.0 = BM25 only, 1.0 = vector only, 0.5 = balanced

    Returns:
        scored: list of (comment_text, score, comment_id) tuples
        texts:  list of comment_text strings only
    """
    try:
        response = collection.query.hybrid(
            query=question,
            alpha=alpha,
            limit=k,
            return_metadata=wq.MetadataQuery(score=True),
        )
        scored = [
            (
                obj.properties["cleaned_text"],
                obj.metadata.score,
                obj.properties["comment_id"]
            )
            for obj in response.objects
        ]
        texts = [obj.properties["cleaned_text"] for obj in response.objects]
        return scored, texts

    except Exception as e:
        print(f"[hybridRetrieval] error: {e}")
        return [], []


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# After hybrid retrieval returns k=10 candidates, rerank them.
# The cross-encoder reads (question, comment) pairs together and scores
# how relevant each comment actually is to the question.
# Much more accurate than embedding similarity alone but slower,
# so run it on a small candidate set (10), not the full collection.
# ---------------------------------------------------------------------------

RERANKER = None  # lazy load so import doesn't slow startup

def rerank(question, scored_results, top_k=5):
    """
    Args:
        question:       user's question string
        scored_results: list of (text, score, comment_id) from hybridRetrieval
        top_k:          how many to return after reranking

    Returns:
        list of (text, rerank_score, comment_id) sorted best first
    """
    global RERANKER
    if RERANKER is None:
        RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    if not scored_results:
        return []

    texts = [r[0] for r in scored_results]
    comment_ids = [r[2] for r in scored_results]

    pairs = [[question, text] for text in texts]
    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(texts, scores, comment_ids),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# Single function the chatbot calls.
# Returns top_k comments as context string + source comment_ids for citation.
# ---------------------------------------------------------------------------

def retrieve(collection, question, k=10, top_k=5, alpha=0.5):
    """
    Args:
        collection: weaviate collection object
        question:   user's question
        k:          candidates to pull from Weaviate before reranking
        top_k:      final number of comments to pass as context
        alpha:      hybrid search balance

    Returns:
        context:     single string joining top_k comments, ready to pass to LLM
        sources:     list of comment_ids for citation
        reranked:    full ranked list of (text, score, comment_id) if needed
    """
    scored, _ = hybridRetrieval(collection, question, k=k, alpha=alpha)

    if not scored:
        return "", [], []

    reranked = rerank(question, scored, top_k=top_k)

    context = "\n\n".join([f"- {text}" for text, score, cid in reranked])
    sources = [cid for text, score, cid in reranked]

    return context, sources, reranked


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify retrieval is working
# python retrieval.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_weaviate_client()

    try:
        positive = client.collections.get("PositiveComments")
        negative = client.collections.get("NegativeComments")

        test_question = "What do people think about Trump's economy?"

        print("\n--- Supporter retrieval ---")
        context, sources, reranked = retrieve(positive, test_question)
        print(f"Sources: {sources}")
        print(f"Context:\n{context}")

        print("\n--- Critic retrieval ---")
        context, sources, reranked = retrieve(negative, test_question)
        print(f"Sources: {sources}")
        print(f"Context:\n{context}")

    finally:
        client.close()