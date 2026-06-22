"""
Stage 2 - Analysis
Reads processed_comments.csv and runs sentiment classification, NER, and
topic modeling on each comment. All results are joined on comment_id into
one output file.

Sentiment: RoBERTa only (cardiffnlp/twitter-roberta-base-sentiment), no VADER,
no fallback - team decision, this deviates from the original handover spec
which described a confidence-based VADER fallback.

Topic modeling: LDA run separately on Pro comments and Anti comments, so we
get "what supporters talk about" vs "what critics talk about" rather than
one combined model.

Input:  processed_comments.csv
Output: analysis_results.csv
"""

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy
from transformers import pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tqdm.pandas()


# ---------- sentiment (RoBERTa only) ----------

def loadSentimentModel():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
        max_length=512,
    )


def getRoBERTaSentiment(text, model):
    result = model(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "LABEL_2":
        sentiment_label = "Pro"
    elif label == "LABEL_0":
        sentiment_label = "Anti"
    else:
        sentiment_label = "Neutral"

    return pd.Series({
        "roberta_label": label,
        "roberta_confidence": score,
        "sentiment_label": sentiment_label,
    })


# ---------- NER (spaCy) ----------

def loadNerModel():
    return spacy.load("en_core_web_sm")


def getEntities(text, nlp):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return json.dumps(entities)


# ---------- topic modeling (LDA, per sentiment split) ----------

def runLda(comments, n_topics=10):
    """
    Fits LDA on a list/Series of comments. Returns:
    - a dict mapping comment index (within this subset) -> (topic_id, top_keywords)
    - the fitted lda + vectorizer (for saving as .pkl, dashboard needs them later)
    """
    if len(comments) < n_topics:
        n_topics = max(2, len(comments) // 2) if len(comments) > 1 else 1

    vect = CountVectorizer(max_features=1000, max_df=0.5)
    X = vect.fit_transform(comments)

    lda = LatentDirichletAllocation(
        n_components=n_topics, learning_method="batch", max_iter=25, random_state=0
    )
    doc_topics = lda.fit_transform(X)

    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names_out())

    topic_keywords = {}
    for i in range(n_topics):
        top_words = feature_names[sorting[i, :5]]
        topic_keywords[i] = ", ".join(top_words)

    topic_assignments = doc_topics.argmax(axis=1)

    return topic_assignments, topic_keywords, lda, vect


# ---------- main pipeline ----------

def runAnalysis(input_path="processed_comments.csv", output_path="analysis_results.csv",
                 lda_pro_path="lda_pro.pkl", lda_anti_path="lda_anti.pkl"):

    df = pd.read_csv(input_path)
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    # ---------- sentiment ----------
    print("Running sentiment analysis (RoBERTa)...")
    sentiment_model = loadSentimentModel()
    sentiment_results = df["cleaned_text"].progress_apply(
        lambda t: getRoBERTaSentiment(t, sentiment_model)
    )
    df = pd.concat([df, sentiment_results], axis=1)
    print(df["sentiment_label"].value_counts())

    # ---------- NER ----------
    print("\nRunning NER (spaCy)...")
    nlp = loadNerModel()
    df["entities"] = df["cleaned_text"].progress_apply(lambda t: getEntities(t, nlp))

    # ---------- topic modeling, split by sentiment ----------
    print("\nRunning topic modeling (LDA, Pro vs Anti)...")

    df["topic_id"] = -1
    df["topic_keywords"] = ""

    pro_mask = df["sentiment_label"] == "Pro"
    anti_mask = df["sentiment_label"] == "Anti"

    if pro_mask.sum() > 1:
        pro_assignments, pro_keywords, lda_pro, vect_pro = runLda(df.loc[pro_mask, "cleaned_text"])
        df.loc[pro_mask, "topic_id"] = pro_assignments
        df.loc[pro_mask, "topic_keywords"] = [pro_keywords[t] for t in pro_assignments]

        import pickle
        with open(lda_pro_path, "wb") as f:
            pickle.dump({"model": lda_pro, "vectorizer": vect_pro, "keywords": pro_keywords}, f)
        print(f"  Pro topics saved to {lda_pro_path}")

    if anti_mask.sum() > 1:
        anti_assignments, anti_keywords, lda_anti, vect_anti = runLda(df.loc[anti_mask, "cleaned_text"])
        df.loc[anti_mask, "topic_id"] = anti_assignments
        df.loc[anti_mask, "topic_keywords"] = [anti_keywords[t] for t in anti_assignments]

        import pickle
        with open(lda_anti_path, "wb") as f:
            pickle.dump({"model": lda_anti, "vectorizer": vect_anti, "keywords": anti_keywords}, f)
        print(f"  Anti topics saved to {lda_anti_path}")

    # Neutral comments are left with topic_id = -1 / no keywords, per spec
    # (spec only requires Pro vs Anti topic modeling)

    # ---------- final column order per spec ----------
    final_cols = [
        "comment_id", "video_id", "author", "updated_at", "like_count", "cleaned_text",
        "roberta_label", "roberta_confidence", "sentiment_label",
        "entities", "topic_id", "topic_keywords",
    ]
    df = df[final_cols]

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    runAnalysis()