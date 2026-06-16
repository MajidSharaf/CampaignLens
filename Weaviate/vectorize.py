import warnings
import pandas as pd
import weaviate
from langchain_ollama import OllamaEmbeddings
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes import query
import os 

warnings.filterwarnings('ignore')

# Connect to Weaviate
client = weaviate.connect_to_custom(
    http_host=os.getenv("WEAVIATE_HOST", "localhost"),
    http_port=8080,
    grpc_host=os.getenv("WEAVIATE_HOST", "localhost"),
    grpc_port=50051,
)

# Load embedded parquet files
Comments_Positive = pd.read_parquet('/dataset/positive_embeddings.parquet')
Comments_Negative = pd.read_parquet('/dataset/negative_embeddings.parquet')

# Define properties
def getProperties():
    return [
        Property(name="comment_id",
                 data_type=DataType.INT,
                 description='Comment ID',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
        Property(name="cleaned_text",
                 data_type=DataType.TEXT,
                 description='Cleaned comment text',
                 indexFilterable=True,
                 indexSearchable=True),
        Property(name="vader_score",
                 data_type=DataType.INT,
                 description='Vader sentiment score',
                 indexFilterable=True,
                 indexSearchable=False,
                 vectorize_property_name=False,
                 skip_vectorization=True),
    ]

# Create collection
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

# Load data into collection from parquet
def loadDataIntoCollection(df, collection_name, batch_size=10):
    collection = client.collections.use(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for _, row in df.iterrows():
            object_ = {
                "comment_id": int(row["comment_id"]),
                "cleaned_text": row["cleaned_text"],
                "vader_score": int(row["Vader"]),
            }
            batch.add_object(properties=object_, vector=row["embedding"])

# Test retrieval
def semanticRetrieval(collection, vector, k=5, threshold=0.5):
    response = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        certainty=threshold,
        return_metadata=query.MetadataQuery(certainty=True),
    )
    return [(obj.properties["cleaned_text"],
             obj.metadata.certainty) for obj in response.objects]

# Run pipeline
createCollection("Comments_Positive")
loadDataIntoCollection(Comments_Positive, "Comments_Positive")

createCollection("Comments_Negative")
loadDataIntoCollection(Comments_Negative, "Comments_Negative")

# Sanity check
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="embeddinggemma")
test_vector = embeddings.embed_query("this product is amazing")
collection = client.collections.use("Comments_Positive")
results = semanticRetrieval(collection, test_vector)
print(results)

client.close()