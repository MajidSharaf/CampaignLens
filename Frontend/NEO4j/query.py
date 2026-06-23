import os
import dspy
from dotenv import load_dotenv
from .knowledge_graph import get_context_for_entity

import mlflow
mlflow.set_experiment("CampaignLens_Chatbots")
mlflow.langchain.autolog()

load_dotenv()

# ---------------------------------------------------------------------------
# DSPy LM — same instance as chatbots.py
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
# Entity extractor
# The user asks "what is the relationship between Trump and the economy?"
# We need to pull "trump" and "economy" out of that before hitting Neo4j.
# Uses DSPy so the small model has clear instructions for the task.
# Taken from the MultiHop pattern in 5_Retrieval.py (getKeywords signature).
# ---------------------------------------------------------------------------

class ExtractEntity(dspy.Signature):
    """Extract the single most important entity or person name from the question.
    Return only the entity as a lowercase single word or short phrase.
    Examples: 'trump', 'economy', 'immigration', 'biden'."""
    question = dspy.InputField(desc="user question about a politician or topic")
    entity = dspy.OutputField(desc="the key entity to look up in the knowledge graph")


# ---------------------------------------------------------------------------
# Graph answer generator
# Takes the raw triplets from Neo4j and synthesizes a coherent answer.
# Same pattern as getAnswer in 7_Generator.py but scoped to graph context.
# ---------------------------------------------------------------------------

class GraphAnswer(dspy.Signature):
    """Answer the question using only the knowledge graph relationships provided.
    The relationships show how entities connect based on what commenters said.
    Do not invent facts not present in the context."""
    question = dspy.InputField(desc="user question")
    context = dspy.InputField(desc="knowledge graph triplets showing entity relationships")
    answer = dspy.OutputField(desc="answer grounded in the graph relationships")
    confidence = dspy.OutputField(desc="float 0-1: how well the graph context supports this answer")


class Neo4jQuery(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_entity = dspy.Predict(ExtractEntity)
        self.generate = dspy.ChainOfThought(GraphAnswer)

    def forward(self, question):
        # step 1 — extract entity from question
        with dspy.context(lm=lm):
            entity_result = self.extract_entity(question=question)
        entity = entity_result.entity.strip().lower()

        # step 2 — query Neo4j for that entity
        context, sources = get_context_for_entity(entity, limit=10)

        if not context:
            return {
                "answer": f"No graph relationships found for '{entity}'.",
                "confidence": 0.0,
                "entity": entity,
                "sources": [],
                "uncertain": True
            }

        # step 3 — generate grounded answer
        with dspy.context(lm=lm):
            result = self.generate(question=question, context=context)

        try:
            confidence = float(result.confidence)
        except (ValueError, TypeError):
            confidence = 0.0

        return {
            "answer": result.answer,
            "confidence": round(confidence, 2),
            "entity": entity,
            "sources": sources,
            "uncertain": confidence < 0.6
        }


# ---------------------------------------------------------------------------
# Module instance — imported by router.py
# ---------------------------------------------------------------------------

neo4j_query = Neo4jQuery()


def run_graph_query(question):
    """
    Single function the router calls.
    Returns same shape as run_chatbot() in chatbots.py so the router
    can treat both tools identically.

    Returns:
        dict with keys: answer, confidence, entity, sources, uncertain
    """
    return neo4j_query(question=question)


# ---------------------------------------------------------------------------
# Run directly to test
# python neo4j_query.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "What is the relationship between Trump and the economy?",
        "What do people say about Trump's political ideology?",
        "How is Trump connected to Russia in the comments?"
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        result = run_graph_query(q)
        print(f"Entity:     {result['entity']}")
        print(f"Answer:     {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Uncertain:  {result['uncertain']}")
        print(f"Sources:    {result['sources'][:3]}")