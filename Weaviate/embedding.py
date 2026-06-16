import warnings
import pandas as pd
import dspy
from langchain_ollama import OllamaEmbeddings

warnings.filterwarnings('ignore')

# Load data
Processed_Comments = pd.read_csv('/dataset/sentiment_results.csv')
Comments_Positive = Processed_Comments[Processed_Comments['Vader'] == 1]
Comments_Negative = Processed_Comments[Processed_Comments['Vader'] == -1]

# Define embedder
embedder_dspy = dspy.Embedder("ollama/embeddinggemma")

def getEmbedding(text):
    return embedder_dspy(text)

def embed_dataframe(df, text_column):
    df = df.copy()
    df['embedding'] = df[text_column].apply(lambda x: getEmbedding(x))
    return df

# Embed
Comments_Positive = embed_dataframe(Comments_Positive, 'cleaned_text')
Comments_Negative = embed_dataframe(Comments_Negative, 'cleaned_text')

# Save to parquet
Comments_Positive.to_parquet('/dataset/positive_embeddings.parquet')
Comments_Negative.to_parquet('/dataset/negative_embeddings.parquet')

print(f"Positive: {len(Comments_Positive)} rows saved")
print(f"Negative: {len(Comments_Negative)} rows saved")