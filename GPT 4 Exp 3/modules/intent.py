"""
modules/intent.py
-----------------
Module 1: Intent Detection using Ollama with controlled intent categories.
"""

from utils.ollama_client import generate

VALID_INTENTS = [
    "Delivery Delay",
    "Order Cancellation",
    "Damaged Product",
    "Refund Request",
    "Wrong Product",
    "Payment Issue",
    "Account Problem",
    "Product Information",
    "Shipping Status",
    "Return Request",
    "Technical Issue",
    "Missing Item",
    "Replacement Request",
    "Invoice Request",
    "General Complaint",
    "Other",
]


def detect_intent(customer_email: str, subject: str, model: str = "llama3.2") -> dict:
    """
    Classify customer's email intent using Ollama.
    """
    intents_formatted = "\n".join(f"- {item}" for item in VALID_INTENTS)

    prompt = f"""You are an expert customer service intent classifier.
Analyze the email below and classify it into EXACTLY ONE intent category from this list:
{intents_formatted}

SUBJECT: {subject}
EMAIL CONTENT:
\"\"\"{customer_email}\"\"\"

RESPONSE FORMAT (Respond in exactly two lines):
INTENT: <exact category name from the list>
REASON: <one short clear sentence explaining why>
"""

    raw_response = generate(prompt, model=model)

    if raw_response.startswith("ERROR:"):
        return {
            "intent": "Unknown",
            "explanation": "",
            "error": raw_response,
        }

    detected = "Other"
    reason = "Classified based on email text analysis."

    for line in raw_response.splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            parsed_intent = line.split(":", 1)[1].strip()
            for v in VALID_INTENTS:
                if v.lower() in parsed_intent.lower() or parsed_intent.lower() in v.lower():
                    detected = v
                    break
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {
        "intent": detected,
        "explanation": reason,
        "error": None,
    }
