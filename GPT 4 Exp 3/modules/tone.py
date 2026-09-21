"""
modules/tone.py
---------------
Module 4: Tone Improvement Module using Ollama.
"""

from utils.ollama_client import generate

TONE_PROFILES = {
    "Professional": "Clear, objective, respectful, polished business language.",
    "Friendly": "Warm, upbeat, courteous, using approachable conversational phrasing.",
    "Formal": "Strictly professional, traditional executive phrasing with no contractions.",
    "Empathetic": "Compassionate, validating customer frustration and showing sincere care.",
}

AVAILABLE_TONES = list(TONE_PROFILES.keys())


def improve_tone(
    original_reply: str,
    tone: str,
    customer_name: str,
    model: str = "llama3.2",
) -> dict:
    """Rewrite customer reply according to the requested tone while retaining key facts."""
    if tone not in TONE_PROFILES:
        return {"improved_reply": original_reply, "error": f"Invalid tone '{tone}'"}

    guidelines = TONE_PROFILES[tone]
    prompt = f"""You are an email tone specialist.
Rewrite the customer email below to have a {tone} tone.
Tone Guidelines: {guidelines}

ORIGINAL EMAIL:
\"\"\"{original_reply}\"\"\"

REQUIREMENTS:
- Retain all key facts, commitments, and meanings from the original email.
- Do NOT add false promises or unsupported policies.
- Keep greeting (Dear {customer_name},) and closing (Best regards,\\nCustomer Support Team).
- Output ONLY the updated email text.
"""

    result = generate(prompt, model=model)
    if result.startswith("ERROR:"):
        return {"improved_reply": "", "error": result}
    return {"improved_reply": result.strip(), "error": None}
