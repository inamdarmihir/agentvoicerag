"""
AgentVoiceRAG – Agentic Voice RAG with Microsoft VibeVoice and Qdrant.

Architecture
------------
  Voice input  ──►  VibeVoice-ASR (transcription)
                        │
  URL Ingestion ──►  Qdrant (vector search)
                        │
                    LLM (response generation)
                        │
                    gTTS (text-to-speech response)
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.ingestion import get_encoder, get_ingested_sources, ingest_url
from src.llm import generate_response
from src.retrieval import build_context, retrieve
from src.voice import VibeVoiceASR, text_to_speech

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgentVoiceRAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("🗄️ Qdrant")
    qdrant_mode = st.radio(
        "Mode",
        ["In-Memory (local)", "Server"],
        index=0,
        help="Use in-memory Qdrant for quick demos, or connect to a running server.",
    )
    qdrant_url = ""
    qdrant_api_key = ""
    if qdrant_mode == "Server":
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=os.getenv("QDRANT_API_KEY", ""),
            type="password",
        )

    st.subheader("🤖 LLM")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for GPT-powered answers.  Leave blank to see raw context.",
    )
    llm_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )

    st.subheader("🎙️ ASR")
    asr_model = st.text_input(
        "VibeVoice-ASR model",
        value=os.getenv("VIBEVOICE_ASR_MODEL", "microsoft/VibeVoice-ASR"),
        help=(
            "HuggingFace model ID for speech recognition.\n"
            "Default: microsoft/VibeVoice-ASR (7 B, requires GPU).\n"
            "Lightweight alternative: openai/whisper-base"
        ),
    )

    st.subheader("🔍 Retrieval")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.caption(
        "**AgentVoiceRAG** · Built with "
        "[VibeVoice](https://github.com/microsoft/VibeVoice) & "
        "[Qdrant](https://qdrant.tech)"
    )


# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_qdrant(mode: str, url: str, api_key: str) -> QdrantClient:
    if mode == "Server":
        kwargs: dict = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        return QdrantClient(**kwargs)
    return QdrantClient(":memory:")


@st.cache_resource
def _get_encoder():
    return get_encoder()


@st.cache_resource
def _get_asr(model_id: str) -> VibeVoiceASR:
    return VibeVoiceASR(model_id=model_id)


qdrant_client = _get_qdrant(qdrant_mode, qdrant_url, qdrant_api_key)
encoder = _get_encoder()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🎙️ AgentVoiceRAG")
st.markdown(
    "**Agentic Voice RAG** powered by "
    "[Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) "
    "and [Qdrant](https://qdrant.tech).  "
    "Ingest web pages, then ask questions with your voice!"
)

tab_ingest, tab_chat = st.tabs(["📥 Ingest URLs", "🎙️ Voice Chat"])

# ===========================================================================
# TAB 1 – URL Ingestion
# ===========================================================================
with tab_ingest:
    st.header("📥 Ingest URLs")
    st.markdown(
        "Enter one or more URLs (one per line).  The pages will be scraped, "
        "chunked, embedded with **all-MiniLM-L6-v2**, and stored in Qdrant."
    )

    url_input = st.text_area(
        "URLs to ingest",
        placeholder="https://example.com\nhttps://another-page.com",
        height=120,
    )

    if st.button("🚀 Ingest", type="primary", use_container_width=True):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            progress = st.progress(0, text="Starting …")
            results = []
            for i, url in enumerate(urls):
                try:
                    with st.spinner(f"Scraping {url} …"):
                        n_chunks, _ = ingest_url(url, qdrant_client, encoder)
                    results.append((url, n_chunks, None))
                    logger.info("Ingested %s → %d chunks", url, n_chunks)
                except Exception as exc:
                    results.append((url, 0, str(exc)))
                    logger.error("Ingestion failed for %s: %s", url, exc)
                progress.progress((i + 1) / len(urls), text=f"Processed {i + 1}/{len(urls)}")

            progress.empty()
            st.subheader("Results")
            for url, n, err in results:
                if err:
                    st.error(f"❌ **{url}** – {err}")
                else:
                    st.success(f"✅ **{url}** – {n} chunks stored")

    # Show already-ingested sources
    sources = get_ingested_sources(qdrant_client)
    if sources:
        with st.expander(f"📚 Ingested sources ({len(sources)})", expanded=False):
            for s in sources:
                st.write(f"• {s}")

# ===========================================================================
# TAB 2 – Voice Chat
# ===========================================================================
with tab_chat:
    st.header("🎙️ Voice Chat")
    st.markdown(
        "Record your question using the microphone below.  "
        "VibeVoice-ASR will transcribe it, then the RAG pipeline will "
        "retrieve relevant context from your ingested pages and generate an answer."
    )

    # --- ASR model loading notice ---
    with st.expander("ℹ️ ASR model information", expanded=False):
        st.info(
            f"**Selected ASR model:** `{asr_model}`\n\n"
            "The VibeVoice-ASR (7 B) model requires a CUDA-enabled GPU for "
            "reasonable inference speed.  On CPU it will work but may be slow.  "
            "You can switch to `openai/whisper-base` in the sidebar for a lighter "
            "alternative."
        )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎤 Record your question")
        audio_input = st.audio_input(
            "Click the microphone icon to record",
            key="voice_input",
        )

        if audio_input is not None:
            st.audio(audio_input, format="audio/wav")

            if st.button("🔍 Transcribe & Answer", type="primary", use_container_width=True):
                # 1. Transcribe ------------------------------------------------
                with st.spinner("🎙️ Transcribing with VibeVoice-ASR …"):
                    try:
                        asr = _get_asr(asr_model)
                        audio_bytes = audio_input.read()
                        transcript = asr.transcribe(audio_bytes)
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
                        st.stop()

                st.session_state["last_transcript"] = transcript
                st.session_state["last_answer"] = None
                st.session_state["last_tts"] = None

                st.success(f"**Transcript:** {transcript}")

                if not transcript.strip():
                    st.warning("No speech detected.  Please try again.")
                    st.stop()

                # 2. Retrieve context ------------------------------------------
                with st.spinner("🔎 Retrieving relevant context from Qdrant …"):
                    chunks = retrieve(transcript, qdrant_client, encoder, top_k=top_k)

                if not chunks:
                    st.warning(
                        "No relevant content found.  "
                        "Please ingest some URLs first (see the Ingest tab)."
                    )
                    st.stop()

                context = build_context(chunks)

                # 3. Generate answer -------------------------------------------
                with st.spinner("🤖 Generating answer …"):
                    answer = generate_response(
                        query=transcript,
                        context=context,
                        openai_api_key=openai_api_key or None,
                        model=llm_model,
                    )

                st.session_state["last_answer"] = answer

                # 4. TTS -------------------------------------------------------
                with st.spinner("🔊 Generating speech …"):
                    try:
                        tts_bytes = text_to_speech(answer)
                        st.session_state["last_tts"] = tts_bytes
                    except Exception as exc:
                        logger.warning("TTS failed: %s", exc)
                        st.session_state["last_tts"] = None

    with col_right:
        st.subheader("💬 Answer")

        transcript = st.session_state.get("last_transcript")
        answer = st.session_state.get("last_answer")
        tts_bytes = st.session_state.get("last_tts")

        if transcript:
            st.markdown(f"**🗣️ You asked:** {transcript}")

        if answer:
            st.markdown(answer)

            if tts_bytes:
                st.audio(tts_bytes, format="audio/mp3", autoplay=True)

    # ---- Chat history ----
    st.markdown("---")
    st.subheader("📜 Conversation history")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    transcript = st.session_state.get("last_transcript")
    answer = st.session_state.get("last_answer")

    if transcript and answer:
        entry = {"user": transcript, "assistant": answer}
        if (
            not st.session_state["history"]
            or st.session_state["history"][-1] != entry
        ):
            st.session_state["history"].append(entry)

    if st.session_state["history"]:
        for turn in reversed(st.session_state["history"]):
            with st.chat_message("user"):
                st.write(turn["user"])
            with st.chat_message("assistant"):
                st.write(turn["assistant"])
    else:
        st.info("Your conversation will appear here after you ask your first question.")

    # Text fallback (typed query)
    st.markdown("---")
    st.subheader("⌨️  Type a question (text fallback)")
    text_query = st.text_input("Ask a question:", key="text_query")
    if st.button("Ask", key="ask_text", use_container_width=True):
        if not text_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🔎 Retrieving …"):
                chunks = retrieve(text_query, qdrant_client, encoder, top_k=top_k)
            if not chunks:
                st.warning("No relevant content found.  Please ingest URLs first.")
            else:
                context = build_context(chunks)
                with st.spinner("🤖 Generating answer …"):
                    answer = generate_response(
                        query=text_query,
                        context=context,
                        openai_api_key=openai_api_key or None,
                        model=llm_model,
                    )
                st.markdown(answer)
                st.session_state["history"].append(
                    {"user": text_query, "assistant": answer}
                )
