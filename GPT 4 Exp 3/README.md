# Intelligent Email Assistant using Ollama
**24CS2506 – Building Applications using GPT-4: Experiment 3**

## Overview
An AI-powered customer email response assistant using local Ollama and Sentence Transformers. No cloud APIs (OpenAI, Gemini, Claude) are used.

## Architecture & Modules
1. **Input Module (`app.py`)**: Collects Customer Name, Subject, and Customer Email.
2. **Intent Module (`modules/intent.py`)**: Classifies email intent using local Ollama (`llama3.2`).
3. **Retrieval Module (`modules/retrieval.py`)**: Finds similar past conversations from `data/email_history.csv` using `all-MiniLM-L6-v2` embeddings and Cosine Similarity.
4. **Reply Module (`modules/reply_generator.py`)**: Generates a context-aware response using the customer query and retrieved historical resolution.
5. **Tone Module (`modules/tone.py`)**: Rewrites replies in Professional, Friendly, Formal, or Empathetic tones.
6. **Summary Module (`modules/summarizer.py`)**: Produces a ~50-word summary with issue, context, and next action.
7. **Save Module (`modules/storage.py`)**: Appends records to `data/generated_replies.csv` with timestamps.

## Setup & Execution
1. Open Terminal:
   ```bash
   cd "/Users/sharuck/Documents/GPT Lab Records/GPT 4 Exp 3"
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start Ollama and download the model:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Access the web interface at: `http://localhost:8501`
