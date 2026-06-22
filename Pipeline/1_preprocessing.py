"""
Stage 1 - Preprocessing
Reads comments.csv, cleans text, generates comment_id for every row.
comment_id is created here and never changes - everything downstream joins on it.

Input:  comments.csv
Output: processed_comments.csv
"""

import hashlib
import json
import re
import pandas as pd

try:
    from autocorrect import Speller
except ImportError:
    raise ImportError("Run: pip install autocorrect")


# ---------- comment_id ----------

def makeCommentId(video_id, author, updated_at):
    raw = f"{video_id}{author}{updated_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------- slang dictionary ----------

def loadSlangDict(path="slang_dict.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expandSlang(text, slang_dict):
    tokens = text.split()
    expanded = []
    for tok in tokens:
        stripped = re.sub(r"[^\w']", "", tok).lower()
        if stripped in slang_dict:
            expanded.append(slang_dict[stripped])
        else:
            expanded.append(tok)
    return " ".join(expanded)


# ---------- cleaning steps ----------

def stripUrls(text):
    return re.sub(r"http\S+|www\S+", "", text)


def stripMentions(text):
    return re.sub(r"@\w+", "", text)


def stripExcessivePunctuation(text):
    return re.sub(r"([!?.,])\1{2,}", r"\1", text)


def cleanWhitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def correctSpelling(text, speller):
    return speller(text)


def cleanText(text, slang_dict, speller):
    text = stripUrls(text)
    text = stripMentions(text)
    text = stripExcessivePunctuation(text)
    text = expandSlang(text, slang_dict)
    text = correctSpelling(text, speller)
    text = cleanWhitespace(text)
    return text


# ---------- main pipeline ----------

def runPreprocessing(input_path="../Datasets/Versions/comments.csv", output_path="processed_comments.csv",
                      slang_path="slang_dict.json"):

    df = pd.read_csv(input_path)

    required_cols = ["video_id", "author", "updated_at", "like_count", "text"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"comments.csv is missing required columns: {missing}")

    df = df[required_cols].copy()

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    df["comment_id"] = df.apply(
        lambda r: makeCommentId(r["video_id"], r["author"], r["updated_at"]), axis=1
    )

    df = df.sort_values("like_count", ascending=False)
    df = df.drop_duplicates(subset=["author", "text"], keep="first")

    slang_dict = loadSlangDict(slang_path)
    speller = Speller(lang="en")

    df["cleaned_text"] = df["text"].apply(lambda t: cleanText(t, slang_dict, speller))

    df = df[df["cleaned_text"].str.len() >= 5]

    df = df[["comment_id", "video_id", "author", "updated_at", "like_count", "cleaned_text"]]
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    runPreprocessing()