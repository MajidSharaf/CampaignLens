import warnings
import numpy as np
import pandas as pd
import weaviate
from langchain_ollama import OllamaEmbeddings
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes import query
import os

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Connect to Weaviate
# ---------------------------------------------------------------------------

client = weaviate.connect_to_local(
    host=os.getenv("WEAVIATE_HOST", "weaviate"),
    port=8080,
    grpc_port=50051,
)

# ---------------------------------------------------------------------------
# Load embedded parquet files (from embedding.py)
# ---------------------------------------------------------------------------

Comments_Positive = pd.read_parquet('/dataset/positive_embeddings.parquet')
Comments_Negative = pd.read_parquet('/dataset/negative_embeddings.parquet')

# ---------------------------------------------------------------------------
# Properties
# comment_id changed from INT to TEXT - Stage 1 generates it as a 16-char
# sha256 hex string (sha256(video_id+author+updated_at)[:16]), not a number.
# vader_score replaced with sentiment_label (Pro/Anti/Neutral, RoBERTa only).
# Also added video_id and topic_id, since retrieval/router and the eventual
# dashboard can filter on these.
# ---------------------------------------------------------------------------

def getProperties():
    return [
        Property(name="comment_id",
                 data_type=DataType.TEXT,
                 description='Comment ID (sha256 hash, traces back to source comment)',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
        Property(name="cleaned_text",
                 data_type=DataType.TEXT,
                 description='Cleaned comment text',
                 indexFilterable=True,
                 indexSearchable=True),
        Property(name="sentiment_label",
                 data_type=DataType.TEXT,
                 description='RoBERTa sentiment: Pro, Anti, or Neutral',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
        Property(name="video_id",
                 data_type=DataType.TEXT,
                 description='Source YouTube video ID',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
        Property(name="topic_id",
                 data_type=DataType.INT,
                 description='LDA topic ID within this sentiment group',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
    ]


# ---------------------------------------------------------------------------
# Create collection
# Names changed to PositiveComments / NegativeComments - matches what
# router.py and chatbot.py already call (Comments_Positive / Comments_Negative
# would not have matched, this was a silent mismatch in the old version).
# ---------------------------------------------------------------------------

def createCollection(name):
    if client.collections.exists(name):
        client.collections.delete(name)

    client.collections.create(
        name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef=128,
                ef_construction=128,
                max_connections=64,
            ),
        ),
        properties=getProperties(),
        inverted_index_config=Configure.inverted_index(
            bm25_k1=1.2,
            bm25_b=0.75,
        )
    )


# ---------------------------------------------------------------------------
# Load data into collection
# ---------------------------------------------------------------------------

def loadDataIntoCollection(df, collection_name, batch_size=10):
    collection = client.collections.use(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for _, row in df.iterrows():
            object_ = {
                "comment_id": str(row["comment_id"]),
                "cleaned_text": row["cleaned_text"],
                "sentiment_label": row["sentiment_label"],
                "video_id": str(row.get("video_id", "")),
                "topic_id": int(row["topic_id"]) if pd.notna(row.get("topic_id")) else -1,
            }
            vector = row["embedding"]
            # Flatten (1, dim) → (dim,) in case embedding was stored as 2D
            vec = np.array(vector)
            if vec.ndim == 2:
                vec = vec[0]
            batch.add_object(properties=object_, vector=vec.tolist())


# ---------------------------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------------------------

def semanticRetrieval(collection, vector, k=5, threshold=0.5):
    response = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        certainty=threshold,
        return_metadata=query.MetadataQuery(certainty=True),
    )
    return [(obj.properties["cleaned_text"],
             obj.metadata.certainty) for obj in response.objects]


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

createCollection("PositiveComments")
loadDataIntoCollection(Comments_Positive, "PositiveComments")

createCollection("NegativeComments")
loadDataIntoCollection(Comments_Negative, "NegativeComments")

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

embeddings = OllamaEmbeddings(model="embeddinggemma", base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"))
test_vector = embeddings.embed_query("this product is amazing")
collection = client.collections.use("PositiveComments")
results = semanticRetrieval(collection, test_vector)
print(results)

client.close()