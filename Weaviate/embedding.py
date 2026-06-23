import os
import warnings
import pandas as pd
import dspy

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load data
# Reads from Stage 2's output (analysis_results.csv), not the old
# sentiment_results.csv. Splits on sentiment_label (Pro/Anti/Neutral) -
# RoBERTa only, no VADER, per team decision. Neutral comments are skipped,
# matching the original Positive/Negative split pattern.
# ---------------------------------------------------------------------------

Processed_Comments = pd.read_csv('/dataset/analysis_results.csv')
Comments_Positive = Processed_Comments[Processed_Comments['sentiment_label'] == 'Pro']
Comments_Negative = Processed_Comments[Processed_Comments['sentiment_label'] == 'Anti']

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
embedder_dspy = dspy.Embedder("ollama/embeddinggemma", api_base=ollama_host)

def getEmbedding(text):
    return embedder_dspy(text)

def embed_dataframe(df, text_column):
    df = df.copy()
    df['embedding'] = df[text_column].apply(lambda x: getEmbedding(x))
    return df

# ---------------------------------------------------------------------------
# Embed and save
# ---------------------------------------------------------------------------

Comments_Positive = embed_dataframe(Comments_Positive, 'cleaned_text')
Comments_Negative = embed_dataframe(Comments_Negative, 'cleaned_text')

Comments_Positive.to_parquet('/dataset/positive_embeddings.parquet')
Comments_Negative.to_parquet('/dataset/negative_embeddings.parquet')

print(f"Positive (Pro): {len(Comments_Positive)} rows saved")
print(f"Negative (Anti): {len(Comments_Negative)} rows saved")