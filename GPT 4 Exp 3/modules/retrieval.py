"""
modules/retrieval.py
--------------------
Module 2: Historical Email Retrieval using Sentence Transformers & Cosine Similarity.
Includes a TF-IDF semantic/lexical fallback so results are always displayed reliably.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "email_history.csv")


def load_dataset():
    """Load the historical emails CSV dataset."""
    file_path = os.path.normpath(DATA_FILE)
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["customer_email"]).reset_index(drop=True)
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_sentence_transformer():
    """Load SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"SentenceTransformer not loaded: {e}")
        return None


def retrieve_similar_emails(current_email: str, top_k: int = 3) -> dict:
    """
    Retrieve top similar historical emails using Sentence Transformers.
    If Sentence Transformers is still downloading or fails, falls back gracefully to TF-IDF cosine similarity.
    """
    df = load_dataset()
    if df is None or df.empty:
        return {
            "results": [],
            "error": "Could not read data/email_history.csv. Please verify the file exists.",
        }

    emails = df["customer_email"].tolist()

    # Strategy 1: Sentence Transformers (Preferred)
    model = get_sentence_transformer()
    if model is not None:
        try:
            history_embeddings = model.encode(emails, convert_to_numpy=True)
            current_emb = model.encode([current_email], convert_to_numpy=True)
            scores = cosine_similarity(current_emb, history_embeddings)[0]

            top_indices = np.argsort(scores)[::-1][: min(top_k, len(df))]
            results = []
            for idx in top_indices:
                row = df.iloc[idx]
                results.append({
                    "customer_email": str(row.get("customer_email", "")),
                    "subject": str(row.get("subject", "")),
                    "intent": str(row.get("intent", "")),
                    "previous_reply": str(row.get("previous_reply", "")),
                    "similarity_score": float(round(scores[idx], 4)),
                })
            return {"results": results, "error": None}
        except Exception as e:
            print(f"Sentence transformer encoding failed: {e}")

    # Strategy 2: TF-IDF Fallback (Fast, guaranteed zero-download)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [current_email] + emails
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        top_indices = np.argsort(scores)[::-1][: min(top_k, len(df))]
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            results.append({
                "customer_email": str(row.get("customer_email", "")),
                "subject": str(row.get("subject", "")),
                "intent": str(row.get("intent", "")),
                "previous_reply": str(row.get("previous_reply", "")),
                "similarity_score": float(round(scores[idx], 4)),
            })
        return {"results": results, "error": None}
    except Exception as e:
        return {"results": [], "error": f"Retrieval failed: {str(e)}"}
