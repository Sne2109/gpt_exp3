"""
utils/ollama_client.py
----------------------
Reusable Ollama client communicating locally with the Ollama server (default http://localhost:11434).
"""

import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running() -> bool:
    """Check if the local Ollama daemon is reachable."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> list:
    """Retrieve installed Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def generate(prompt: str, model: str = "llama3.2", system_prompt: str = None) -> str:
    """
    Send prompt to local Ollama instance and get non-streamed response.
    """
    if not is_ollama_running():
        return (
            "ERROR: Ollama is not running. "
            "Please start Ollama (run: ollama serve) and ensure the selected model is installed."
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        elif response.status_code == 404:
            return f"ERROR: Model '{model}' not found in Ollama. Run: ollama pull {model}"
        else:
            return f"ERROR: Ollama responded with status code {response.status_code}."
    except requests.exceptions.Timeout:
        return "ERROR: Ollama request timed out. The model might be loading into memory."
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Ollama at http://localhost:11434."
    except Exception as e:
        return f"ERROR: Unexpected error with Ollama: {str(e)}"
