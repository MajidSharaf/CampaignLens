"""
Insert ~15 fake comments into PositiveComments and NegativeComments
to unblock chatbot testing before the teammate finishes stage 3.

Usage (from inside the app container):
    python mock_insert.py

Run once. Re-running will duplicate entries — delete the collections first
if you need a clean slate:
    python -c "import weaviate; c=weaviate.connect_to_local('weaviate',8080); \
               c.collections.delete('PositiveComments'); \
               c.collections.delete('NegativeComments'); c.close()"
"""

import os
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

POSITIVE_COMMENTS = [
    ("mock_pos_001", "Trump has done more for the economy than any president in decades. Jobs are back!"),
    ("mock_pos_002", "Finally a president who puts America first. No more bad trade deals."),
    ("mock_pos_003", "The stock market was incredible under Trump. My 401k grew like never before."),
    ("mock_pos_004", "Trump stood up to China when nobody else would. That took real courage."),
    ("mock_pos_005", "Lower taxes meant more money in my pocket. That is real policy that works."),
    ("mock_pos_006", "Trump rebuilt the military and made the world respect us again."),
    ("mock_pos_007", "He kept his promises on immigration. The border was more secure under his watch."),
    ("mock_pos_008", "Operation Warp Speed saved millions of lives. History will judge him well."),
    ("mock_pos_009", "Energy independence was a game changer. Gas prices were low and America was winning."),
    ("mock_pos_010", "Trump speaks the truth that no politician has had the guts to say."),
]

NEGATIVE_COMMENTS = [
    ("mock_neg_001", "Trump's handling of COVID was a disaster. He downplayed it while people died."),
    ("mock_neg_002", "The January 6th attack on democracy is something I will never forgive him for."),
    ("mock_neg_003", "The tax cuts only benefited the ultra wealthy. Working people got nothing."),
    ("mock_neg_004", "His rhetoric has divided the country in ways that will take generations to heal."),
    ("mock_neg_005", "Pulling out of the Paris climate agreement was irresponsible and short sighted."),
    ("mock_neg_006", "He turned the justice department into a personal attack machine against his enemies."),
    ("mock_neg_007", "The trade war with China hurt American farmers and consumers, not China."),
    ("mock_neg_008", "Four years of lies and misinformation from the highest office in the land."),
    ("mock_neg_009", "His immigration policies separated families and violated basic human rights."),
    ("mock_neg_010", "He has no respect for democratic institutions or the rule of law."),
]


def get_embedding(text: str, embeddings: OllamaEmbeddings) -> list[float]:
    return embeddings.embed_query(text)


def ensure_collection(client, name: str):
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            properties=[
                wvc.config.Property(name="comment_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="cleaned_text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="sentiment", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )
        print(f"[mock_insert] Created collection: {name}")
    else:
        print(f"[mock_insert] Collection already exists: {name}")


def insert_batch(collection, comments: list[tuple], sentiment: str, embeddings: OllamaEmbeddings):
    with collection.batch.dynamic() as batch:
        for comment_id, text in comments:
            vector = get_embedding(text, embeddings)
            batch.add_object(
                properties={
                    "comment_id": comment_id,
                    "cleaned_text": text,
                    "sentiment": sentiment,
                },
                vector=vector,
            )
    print(f"[mock_insert] Inserted {len(comments)} comments into {collection.name}")


def main():
    client = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "weaviate"),
        port=int(os.getenv("WEAVIATE_PORT", 8080)),
    )

    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
        base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    )

    try:
        ensure_collection(client, "PositiveComments")
        ensure_collection(client, "NegativeComments")

        positive = client.collections.get("PositiveComments")
        negative = client.collections.get("NegativeComments")

        print("[mock_insert] Embedding and inserting positive comments...")
        insert_batch(positive, POSITIVE_COMMENTS, "positive", embeddings)

        print("[mock_insert] Embedding and inserting negative comments...")
        insert_batch(negative, NEGATIVE_COMMENTS, "negative", embeddings)

        print("[mock_insert] Done. Run chatbot tests now.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
