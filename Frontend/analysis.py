import os
import json
import re
import collections
import pandas as pd
from functools import lru_cache

DATA_PATH = os.getenv("PIPELINE_DATA_PATH", "/app/pipeline_data")

EMOJI_PATTERN = re.compile(r'^[a-z_]+(?:_with_[a-z_]+)+$|^[a-z]+_[a-z]+_[a-z]+_[a-z_]+$')
SKIP_TOKENS = {"one", "two", "three", "first", "second", "third", "last", "many", "few", "thumbs_up", "thumbs_down"}
KEEP_TYPES = {"GPE", "PERSON", "ORG", "NORP"}


def _is_noise(text: str) -> bool:
    t = text.lower().strip()
    return t in SKIP_TOKENS or len(t) <= 1 or bool(EMOJI_PATTERN.match(t))


@lru_cache(maxsize=1)
def load_analysis():
    path = os.path.join(DATA_PATH, "analysis_results.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    total = len(df)

    # ── Sentiment ──────────────────────────────────────────────────────────
    # Pipeline outputs RoBERTa only: Pro / Anti / Neutral
    vc = df["sentiment_label"].value_counts().to_dict()
    sentiment = {
        "roberta": {
            "pro":     int(vc.get("Pro", 0)),
            "anti":    int(vc.get("Anti", 0)),
            "neutral": int(vc.get("Neutral", 0)),
        }
    }

    # ── NER ───────────────────────────────────────────────────────────────
    entity_counts: dict[str, dict] = {}
    type_counts: dict[str, int] = collections.Counter()

    for val in df["entities"].dropna():
        try:
            ents = json.loads(val)
        except Exception:
            continue
        for ent in ents:
            text = ent.get("text", "").lower().strip()
            label = ent.get("label", "")
            if _is_noise(text) or label not in KEEP_TYPES:
                continue
            if text not in entity_counts:
                entity_counts[text] = {"count": 0, "label": label}
            entity_counts[text]["count"] += 1
            type_counts[label] += 1

    top_entities = sorted(
        [{"text": k, "count": v["count"], "label": v["label"]}
         for k, v in entity_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:30]

    ner = {
        "top_entities": top_entities,
        "by_type": {k: int(v) for k, v in type_counts.most_common()},
    }

    # ── Topics ────────────────────────────────────────────────────────────
    # LDA was run separately per sentiment group. The CSV has topic_id and
    # topic_keywords per comment. Reconstruct topic summaries from the CSV
    # so we avoid re-running LDA at runtime.
    topics = {"pro": [], "anti": []}

    for group_label, key in [("Pro", "pro"), ("Anti", "anti")]:
        group = df[df["sentiment_label"] == group_label].copy()
        if group.empty or "topic_id" not in group.columns:
            continue

        group_valid = group[group["topic_id"] >= 0]
        if group_valid.empty:
            continue

        topic_counts = group_valid.groupby(["topic_id", "topic_keywords"]).size().reset_index(name="count")
        total_group = topic_counts["count"].sum()

        for _, row in topic_counts.sort_values("count", ascending=False).iterrows():
            words = [w.strip() for w in str(row["topic_keywords"]).split(",") if w.strip()]
            topics[key].append({
                "id": int(row["topic_id"]),
                "words": words,
                "count": int(row["count"]),
                "weight_pct": round(int(row["count"]) / total_group * 100, 1),
            })

    # ── Keywords ──────────────────────────────────────────────────────────
    keyword_counts: dict[str, int] = collections.Counter()

    for val in df["keywords"].dropna():
        try:
            kws = json.loads(val)
        except Exception:
            continue
        for kw in kws:
            kw = kw.lower().strip()
            if kw and len(kw) > 2:
                keyword_counts[kw] += 1

    # Top 40 overall, plus top 20 per sentiment group
    top_keywords = [{"text": k, "count": v}
                    for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:40]]

    def top_keywords_for(label):
        counts: dict[str, int] = collections.Counter()
        group = df[df["sentiment_label"] == label]
        for val in group["keywords"].dropna():
            try:
                kws = json.loads(val)
            except Exception:
                continue
            for kw in kws:
                kw = kw.lower().strip()
                if kw and len(kw) > 2:
                    counts[kw] += 1
        return [{"text": k, "count": v}
                for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]]

    keywords = {
        "overall": top_keywords,
        "pro":     top_keywords_for("Pro"),
        "anti":    top_keywords_for("Anti"),
    }

    # ── Sentiment examples ────────────────────────────────────────────────
    def sample_comments(label, n=5):
        group = df[df["sentiment_label"] == label]["cleaned_text"]
        # prefer mid-length comments (not too short, not too long)
        filtered = group[group.str.len().between(30, 200)].drop_duplicates()
        if len(filtered) >= n:
            return filtered.sample(n, random_state=42).tolist()
        return filtered.head(n).tolist()

    sentiment["examples"] = {
        "pro":     sample_comments("Pro"),
        "anti":    sample_comments("Anti"),
        "neutral": sample_comments("Neutral"),
    }

    return {
        "total_comments": total,
        "sentiment": sentiment,
        "ner": ner,
        "keywords": keywords,
        "topics": topics,
    }
