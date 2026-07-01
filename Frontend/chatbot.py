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

lm = dspy.LM(
    f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen2.5:0.5b')}",
    api_base=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    api_key=None,
    temperature=0.7,
    cache=False
)
dspy.configure(lm=lm)

class SupporterSignature(dspy.Signature):
    """You are a chatbot summarizing what supporters say in YouTube comments about the topic.
    Read the provided comments and write a short natural summary of the main sentiment.
    RULES:
    - Never copy or quote comments directly
    - Write 1-2 sentences only"""
    context = dspy.InputField(desc="YouTube comments from supporters")
    question = dspy.InputField(desc="the question")
    response = dspy.OutputField(desc="1-2 sentences naturally summarizing supporter sentiment")
    confidence = dspy.OutputField(desc="float 0-1")


class CriticSignature(dspy.Signature):
    """You are a chatbot summarizing what critics say in YouTube comments about the topic.
    Read the provided comments and write a short natural summary of the main sentiment.
    RULES:
    - Never copy or quote comments directly
    - Write 1-2 sentences only"""
    context = dspy.InputField(desc="YouTube comments from critics")
    question = dspy.InputField(desc="the question")
    response = dspy.OutputField(desc="1-2 sentences naturally summarizing critic sentiment")
    confidence = dspy.OutputField(desc="float 0-1")


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


CONFIDENCE_THRESHOLD = 0.6

def run_chatbot(chatbot, collection, question, k=10, top_k=5):
    context, sources, reranked = retrieve(collection, question, k=k, top_k=top_k)

    if not context:
        return {
            "response": "No relevant comments were found in the dataset for this question.",
            "confidence": 0.0,
            "sources": [],
            "uncertain": True
        }

    with dspy.context(lm=lm):
        result = chatbot(question=question, context=context)

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
