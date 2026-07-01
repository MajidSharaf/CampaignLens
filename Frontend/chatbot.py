cat > /app/chatbot.py << 'ENDOFFILE'
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
    """You are a political analyst summarizing what Trump SUPPORTERS say in YouTube comments.
    Based only on the provided comments, summarize what supporters think, feel, or believe about the topic.
    Do NOT speak in first person. Do NOT quote comments directly. Do NOT sound like you are Trump himself.
    Synthesize the views analytically, as if reporting on public opinion."""
    context = dspy.InputField(desc="YouTube comments from Trump supporters")
    question = dspy.InputField(desc="the question being asked about these comments")
    response = dspy.OutputField(desc="2-3 sentences summarizing what supporters think. Start with Supporters or Trump supporters or Many commenters.")
    confidence = dspy.OutputField(desc="float 0-1 reflecting how well the comments answer the question")


class CriticSignature(dspy.Signature):
    """You are a political analyst summarizing what Trump CRITICS say in YouTube comments.
    Based only on the provided comments, summarize what critics think, feel, or believe about the topic.
    Do NOT speak in first person. Do NOT quote comments directly. Do NOT sound like you are Trump himself.
    Synthesize the views analytically, as if reporting on public opinion."""
    context = dspy.InputField(desc="YouTube comments from Trump critics")
    question = dspy.InputField(desc="the question being asked about these comments")
    response = dspy.OutputField(desc="2-3 sentences summarizing what critics think. Start with Critics or Trump critics or Many commenters.")
    confidence = dspy.OutputField(desc="float 0-1 reflecting how well the comments answer the question")


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
ENDOFFILE
