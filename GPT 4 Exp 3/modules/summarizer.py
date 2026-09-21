"""
modules/summarizer.py
---------------------
Module 5: Conversation Summarizer (~50 words) using Ollama.
"""

from utils.ollama_client import generate


def summarize_conversation(
    customer_name: str,
    subject: str,
    customer_email: str,
    detected_intent: str,
    generated_reply: str,
    model: str = "llama3.2",
) -> dict:
    """Produce a concise ~50-word summary containing customer issue, context, and next step."""
    prompt = f"""Summarize this customer support exchange concisely in approximately 50 words.

CONVERSATION:
Customer: {customer_name}
Subject: {subject}
Intent: {detected_intent}
Customer Email: \"\"\"{customer_email}\"\"\"
Support Reply: \"\"\"{generated_reply}\"\"\"

ESSENTIAL SUMMARY CRITERIA:
1. Identify the core problem.
2. Note key contextual details.
3. State the action or next step being taken.
Output ONLY the summary paragraph.
"""

    result = generate(prompt, model=model)
    if result.startswith("ERROR:"):
        return {"summary": "", "error": result}
    return {"summary": result.strip(), "error": None}
