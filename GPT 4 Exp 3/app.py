"""
app.py
------
Streamlit User Interface for Intelligent Email Assistant (Experiment 3).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils.ollama_client import is_ollama_running, get_available_models
from modules.intent import detect_intent
from modules.retrieval import retrieve_similar_emails
from modules.reply_generator import generate_reply
from modules.tone import improve_tone, AVAILABLE_TONES
from modules.summarizer import summarize_conversation
from modules.storage import save_reply

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Email Assistant",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .badge-intent {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-score {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .footer-bar {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ollama Settings")
    installed = get_available_models()
    model_choices = installed if installed else ["llama3.2", "llama3.1", "mistral"]
    selected_model = st.selectbox("Model", options=model_choices, index=0)
    top_k = st.slider("Similarity results", min_value=1, max_value=5, value=3)

    st.markdown("---")
    running = is_ollama_running()
    if running:
        st.success("✅ Ollama is running")
    else:
        st.error("❌ Ollama is not running")
        st.caption("Start Ollama in terminal using: `ollama serve`")

    st.markdown("---")
    st.subheader("About")
    st.info(
        "This application uses Ollama and local AI models to understand customer emails, "
        "retrieve similar historical conversations, generate replies, improve tone, "
        "summarize conversations, and save responses locally."
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Intelligent Email Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered customer email response assistant using Ollama</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
for state_key in ["intent_data", "retrieval_data", "reply_data", "tone_data", "summary_data", "saved_data"]:
    if state_key not in st.session_state:
        st.session_state[state_key] = None

if "chosen_tone" not in st.session_state:
    st.session_state["chosen_tone"] = AVAILABLE_TONES[0]

# ─────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("Customer Name:", placeholder="e.g. John", key="input_name")
    with col2:
        subject = st.text_input("Subject:", placeholder="e.g. Delivery Delay", key="input_subject")

    customer_email = st.text_area(
        "Customer Email:",
        placeholder="e.g. My order has not been delivered yet. or My laptop screen is broken.",
        height=140,
        key="input_email",
    )

    generate_btn = st.button("Generate Reply", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────
if generate_btn:
    if not customer_name.strip() or not subject.strip() or not customer_email.strip():
        st.warning("⚠️ Please fill in all fields (Customer Name, Subject, and Customer Email).")
    elif not is_ollama_running():
        st.error(f"❌ Ollama is not running. Please run `ollama serve` and ensure model `{selected_model}` is pulled.")
    else:
        # Step 1: Intent Detection
        with st.spinner("Detecting customer intent..."):
            st.session_state["intent_data"] = detect_intent(customer_email, subject, model=selected_model)

        # Step 2: Retrieval
        with st.spinner("Retrieving similar past interactions via Sentence Transformers..."):
            st.session_state["retrieval_data"] = retrieve_similar_emails(customer_email, top_k=top_k)

        # Step 3: Context-aware Reply Generation
        top_past_email = ""
        top_past_reply = ""
        if st.session_state["retrieval_data"] and st.session_state["retrieval_data"].get("results"):
            top_past_email = st.session_state["retrieval_data"]["results"][0].get("customer_email", "")
            top_past_reply = st.session_state["retrieval_data"]["results"][0].get("previous_reply", "")

        with st.spinner("Generating AI customer support reply via Ollama..."):
            st.session_state["reply_data"] = generate_reply(
                customer_name=customer_name,
                subject=subject,
                customer_email=customer_email,
                detected_intent=st.session_state["intent_data"].get("intent", "Other"),
                similar_email=top_past_email,
                similar_reply=top_past_reply,
                model=selected_model,
            )

        # Step 4: Summary Generation
        with st.spinner("Summarizing conversation..."):
            st.session_state["summary_data"] = summarize_conversation(
                customer_name=customer_name,
                subject=subject,
                customer_email=customer_email,
                detected_intent=st.session_state["intent_data"].get("intent", "Other"),
                generated_reply=st.session_state["reply_data"].get("reply", ""),
                model=selected_model,
            )

        st.session_state["tone_data"] = None
        st.session_state["saved_data"] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────
if st.session_state["intent_data"]:
    st.markdown("---")

    # Section 1: Detected Intent
    st.subheader("Section 1 — Detected Intent")
    intent_res = st.session_state["intent_data"]
    if intent_res.get("error"):
        st.error(intent_res["error"])
    else:
        col_i1, col_i2 = st.columns([1, 3])
        with col_i1:
            st.markdown(f'<span class="badge-intent">{intent_res["intent"]}</span>', unsafe_allow_html=True)
        with col_i2:
            st.write(f"**Explanation:** {intent_res['explanation']}")

    # Section 2: Similar Historical Conversation
    st.markdown("---")
    st.subheader("Section 2 — Similar Historical Conversation")
    retrieval_res = st.session_state["retrieval_data"]
    if retrieval_res and retrieval_res.get("results"):
        top_match = retrieval_res["results"][0]
        st.markdown(
            f'<span class="badge-score">Cosine Similarity: {top_match["similarity_score"]:.4f}</span>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Similar Previous Customer Email:**")
            st.info(top_match["customer_email"])
        with c2:
            st.markdown("**Previous Reply:**")
            st.success(top_match["previous_reply"])

        if len(retrieval_res["results"]) > 1:
            with st.expander(f"View {len(retrieval_res['results']) - 1} other similar interactions"):
                for idx, extra in enumerate(retrieval_res["results"][1:], start=2):
                    st.markdown(f"**Match #{idx}** | Score: `{extra['similarity_score']:.4f}` | Intent: `{extra['intent']}`")
                    st.write(f"*Email:* {extra['customer_email']}")
                    st.write(f"*Reply:* {extra['previous_reply']}")
                    st.divider()
    elif retrieval_res and retrieval_res.get("error"):
        st.error(f"❌ Retrieval Note: {retrieval_res['error']}")
    else:
        st.info("No matching previous conversations found.")

    # Section 3: AI Generated Reply
    st.markdown("---")
    st.subheader("Section 3 — AI Generated Reply")
    reply_res = st.session_state["reply_data"]
    if reply_res:
        if reply_res.get("error"):
            st.error(reply_res["error"])
        else:
            st.text_area("Generated Reply:", value=reply_res["reply"], height=200, key="gen_reply_area", disabled=True)

    # Section 4: Tone Module
    if reply_res and not reply_res.get("error"):
        st.markdown("---")
        st.subheader("Section 4 — Tone Improvement")
        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            chosen_tone = st.selectbox(
                "Select Tone:",
                options=AVAILABLE_TONES,
                index=AVAILABLE_TONES.index(st.session_state["chosen_tone"]),
            )
            st.session_state["chosen_tone"] = chosen_tone
        with t_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            improve_btn = st.button("Improve Tone", use_container_width=True)

        if improve_btn:
            with st.spinner(f"Improving tone to {chosen_tone}..."):
                st.session_state["tone_data"] = improve_tone(
                    original_reply=reply_res["reply"],
                    tone=chosen_tone,
                    customer_name=customer_name,
                    model=selected_model,
                )
                st.rerun()

        if st.session_state["tone_data"]:
            if st.session_state["tone_data"].get("error"):
                st.error(st.session_state["tone_data"]["error"])
            else:
                st.markdown(f"**Improved Reply ({st.session_state['chosen_tone']} Tone):**")
                st.text_area(
                    "Improved Tone Reply:",
                    value=st.session_state["tone_data"]["improved_reply"],
                    height=200,
                    key="improved_reply_area",
                    disabled=True,
                )

    # Section 5: Summary Module
    if st.session_state["summary_data"]:
        st.markdown("---")
        st.subheader("Section 5 — Conversation Summary")
        sum_res = st.session_state["summary_data"]
        if sum_res.get("error"):
            st.error(sum_res["error"])
        else:
            st.info(f"📝 {sum_res['summary']}")

    # Section 6: Save Module
    if reply_res and not reply_res.get("error"):
        st.markdown("---")
        st.subheader("Section 6 — Save Generated Reply")
        save_btn = st.button("Save Response", type="secondary", use_container_width=True)
        if save_btn:
            hist_mail = ""
            sim_score = 0.0
            if retrieval_res and retrieval_res.get("results"):
                hist_mail = retrieval_res["results"][0].get("customer_email", "")
                sim_score = retrieval_res["results"][0].get("similarity_score", 0.0)

            tone_out = ""
            if st.session_state["tone_data"] and not st.session_state["tone_data"].get("error"):
                tone_out = st.session_state["tone_data"].get("improved_reply", "")

            sum_out = ""
            if st.session_state["summary_data"] and not st.session_state["summary_data"].get("error"):
                sum_out = st.session_state["summary_data"].get("summary", "")

            saved = save_reply(
                customer_name=customer_name,
                subject=subject,
                customer_email=customer_email,
                detected_intent=intent_res.get("intent", ""),
                similar_historical_email=hist_mail,
                similarity_score=sim_score,
                generated_reply=reply_res.get("reply", ""),
                selected_tone=st.session_state["chosen_tone"],
                tone_improved_reply=tone_out,
                conversation_summary=sum_out,
            )
            if saved.get("success"):
                st.success(f"✅ Response saved successfully to: `{saved['filepath']}`")
            else:
                st.error(f"❌ Failed to save: {saved.get('error')}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="footer-bar">Powered by Ollama • Local AI • Intelligent Email Assistant</div>', unsafe_allow_html=True)
