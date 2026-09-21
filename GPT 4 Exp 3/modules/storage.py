"""
modules/storage.py
------------------
Module 6: Local CSV Storage for generated replies and conversation metadata.
"""

import os
import csv
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "generated_replies.csv")

HEADERS = [
    "timestamp",
    "customer_name",
    "subject",
    "customer_email",
    "detected_intent",
    "similar_historical_email",
    "similarity_score",
    "generated_reply",
    "selected_tone",
    "tone_improved_reply",
    "conversation_summary",
]


def save_reply(
    customer_name: str,
    subject: str,
    customer_email: str,
    detected_intent: str,
    similar_historical_email: str,
    similarity_score: float,
    generated_reply: str,
    selected_tone: str,
    tone_improved_reply: str,
    conversation_summary: str,
) -> dict:
    """Save record to data/generated_replies.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.normpath(OUTPUT_FILE)
    exists = os.path.isfile(file_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name,
        "subject": subject,
        "customer_email": customer_email,
        "detected_intent": detected_intent,
        "similar_historical_email": similar_historical_email,
        "similarity_score": similarity_score,
        "generated_reply": generated_reply,
        "selected_tone": selected_tone,
        "tone_improved_reply": tone_improved_reply,
        "conversation_summary": conversation_summary,
    }

    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return {"success": True, "filepath": file_path, "error": None}
    except Exception as e:
        return {"success": False, "filepath": file_path, "error": str(e)}
