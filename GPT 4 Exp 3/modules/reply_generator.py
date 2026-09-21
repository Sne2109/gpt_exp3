"""
modules/reply_generator.py
--------------------------
Module 3: Context-Aware Customer Support Reply Generation using Ollama.
"""

from utils.ollama_client import generate

SUPPORT_SYSTEM_PROMPT = """You are an empathetic, concise, and professional customer-service representative.
Rules:
- Address customer by name politely.
- Use the retrieved historical customer conversation as inspiration and context, but adapt it to the customer's exact scenario.
- Never mention AI, LLM, Ollama, vector search, or database retrieval.
- Never invent policies or promise unrealistic deadlines unless stated in context.
- Sign off as: Customer Support Team.
"""


def generate_reply(
    customer_name: str,
    subject: str,
    customer_email: str,
    detected_intent: str,
    similar_email: str,
    similar_reply: str,
    model: str = "llama3.2",
) -> dict:
    """Generate context-aware professional customer support reply."""
    prompt = f"""Write a professional customer support email reply.

CUSTOMER INFO:
Name: {customer_name}
Subject: {subject}
Intent: {detected_intent}

CUSTOMER EMAIL:
\"\"\"{customer_email}\"\"\"

SIMILAR HISTORICAL INTERACTION (CONTEXT):
Past Email: \"{similar_email}\"
Past Resolution: \"{similar_reply}\"

INSTRUCTIONS:
- Start with: Dear {customer_name},
- Directly acknowledge their issue with empathy.
- Provide a clear, actionable resolution matching the situation.
- Conclude with:
Best regards,
Customer Support Team
- Output ONLY the reply email body.
"""

    result = generate(prompt, model=model, system_prompt=SUPPORT_SYSTEM_PROMPT)
    if result.startswith("ERROR:"):
        return {"reply": "", "error": result}
    return {"reply": result.strip(), "error": None}
