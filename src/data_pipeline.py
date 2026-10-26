"""Local ticket-data inspection and topic discovery. These operations do not call Gemini."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import (
    DEMO_TICKETS_FILE,
    TICKETS_RAW_FILE,
    TICKETS_CLEAN_FILE,
    TOPIC_PROFILE_FILE,
    PROCESSED_DIR,
)


def load_tickets() -> tuple[pd.DataFrame, str]:
    """Load real tickets_raw.csv when present; otherwise use the included demo sample.

    Accepted schemas are Document/Topic_group or text/category. Returns a normalized
    DataFrame and a label describing the source used.
    """
    if TICKETS_RAW_FILE.exists():
        path = TICKETS_RAW_FILE
        source = "Real dataset: data/tickets_raw.csv"
    else:
        path = DEMO_TICKETS_FILE
        source = "Included demo sample (replace with your real data/tickets_raw.csv)"

    if not path.exists():
        raise FileNotFoundError(f"Ticket dataset not found: {path}")

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    rename = {}
    if "Document" in df.columns:
        rename["Document"] = "text"
    if "Topic_group" in df.columns:
        rename["Topic_group"] = "category"
    df = df.rename(columns=rename)

    missing = {"text", "category"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset must contain columns Document and Topic_group, or text and category. "
            f"Missing normalized columns: {sorted(missing)}. Found: {df.columns.tolist()}"
        )

    before = len(df)
    df = df[["text", "category"]].copy()
    df["text"] = df["text"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df = df.dropna(subset=["text", "category"])
    df = df[(df["text"] != "") & (df["category"] != "")].copy()
    df = df.drop_duplicates(subset=["text", "category"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["category"] = df["category"].astype(str)
    df.attrs["source_rows"] = before
    return df, source


def _keywords_for_texts(texts: list[str], n: int = 12) -> list[dict[str, Any]]:
    if not texts:
        return []
    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2 if len(texts) >= 10 else 1,
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
        )
        matrix = vectorizer.fit_transform(texts)
        counts = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        order = counts.argsort()[::-1][:n]
        return [
            {"term": str(terms[i]), "count": int(counts[i])}
            for i in order if counts[i] > 0
        ]
    except ValueError:
        words = Counter()
        for text in texts:
            words.update(re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower()))
        stop = {"the", "and", "for", "with", "this", "that", "from", "have", "has", "not", "are", "was", "but", "you", "your"}
        return [{"term": w, "count": c} for w, c in words.most_common() if w not in stop][:n]


def profile_tickets() -> dict[str, Any]:
    """Create a compact profile and category-specific phrase suggestions locally."""
    df, source = load_tickets()
    categories: dict[str, Any] = {}

    # Fit each category separately; these terms describe recurring ticket text,
    # not verified resolutions.
    for category, group in df.groupby("category", sort=True):
        texts = group["text"].tolist()
        sample_count = min(3, len(texts))
        samples = group["text"].sample(n=sample_count, random_state=42).tolist() if sample_count else []
        categories[str(category)] = {
            "count": int(len(group)),
            "keywords": _keywords_for_texts(texts, n=12),
            "examples": [" ".join(str(x).split())[:300] for x in samples],
        }

    length_stats = df["text"].str.len().describe().round(1).to_dict()
    profile = {
        "source": source,
        "rows": int(len(df)),
        "categories_count": int(df["category"].nunique()),
        "duplicate_texts_after_cleaning": int(df["text"].duplicated().sum()),
        "text_length_stats": {str(k): float(v) for k, v in length_stats.items()},
        "categories": categories,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TICKETS_CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TICKETS_CLEAN_FILE, index=False)
    TOPIC_PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def load_saved_profile() -> dict[str, Any] | None:
    if not TOPIC_PROFILE_FILE.exists():
        return None
    try:
        return json.loads(TOPIC_PROFILE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
