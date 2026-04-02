# 🎙️ AgentVoiceRAG

> **Agentic Voice RAG** – ask questions about any web page using your voice.
> Powered by [Microsoft VibeVoice-ASR](https://github.com/microsoft/VibeVoice) for speech recognition and [Qdrant](https://qdrant.tech) as the vector database.

---

## Architecture

```
 Microphone 🎤
      │
      ▼
 VibeVoice-ASR ──── (transcription)
      │
      ▼
 Qdrant Vector DB ─ (semantic search over ingested pages)
      │
      ▼
 OpenAI LLM ──────── (RAG answer generation)
      │
      ▼
 gTTS 🔊 ──────────── (text-to-speech response)
```

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | [Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) (7 B, via 🤗 Transformers ≥ 5.3.0) |
| Vector Database | [Qdrant](https://qdrant.tech) (in-memory or server) |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers |
| LLM | OpenAI GPT-4o / GPT-4o-mini / GPT-3.5-turbo |
| Text-to-Speech | [gTTS](https://pypi.org/project/gTTS/) |
| UI | [Streamlit](https://streamlit.io) |

---

## Features

- �� **URL ingestion** – scrape any public web page, chunk the text, embed it, and store it in Qdrant
- 🎤 **Voice input** – record questions directly in the browser; VibeVoice-ASR handles up to 60 minutes of audio in a single pass
- 🔍 **Semantic search** – top-K cosine similarity retrieval from Qdrant
- 🤖 **RAG answers** – GPT-4o generates grounded answers from retrieved context
- 🔊 **Voice output** – answers are read back using gTTS
- ⌨️ **Text fallback** – type questions when a microphone is unavailable

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/inamdarmihir/agentvoicerag.git
cd agentvoicerag
pip install -r requirements.txt
```

> **Note**: VibeVoice-ASR is a 7 B parameter model.  A CUDA-capable GPU is
> strongly recommended.  On CPU it will still work but inference will be slow.
> Set `VIBEVOICE_ASR_MODEL=openai/whisper-base` in `.env` for a lightweight
> CPU-friendly alternative.

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional – needed for GPT answers)
```

### 3. (Optional) Start a Qdrant server with Docker

```bash
docker compose up -d
```

The UI also ships an **in-memory** Qdrant mode that requires no extra setup.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Ingest URLs

1. Click the **📥 Ingest URLs** tab.
2. Paste one or more URLs (one per line) into the text area.
3. Click **🚀 Ingest**.

The app scrapes each page, splits the text into 512-token chunks with 64-token
overlap, generates embeddings, and upserts them into Qdrant.

### Ask questions with your voice

1. Click the **🎙️ Voice Chat** tab.
2. Click the microphone icon and record your question.
3. Click **🔍 Transcribe & Answer**.

The pipeline will:
1. Transcribe your audio with VibeVoice-ASR.
2. Embed the transcript and retrieve the top-K most relevant chunks from Qdrant.
3. Send the context + question to OpenAI and stream back an answer.
4. Read the answer aloud using gTTS.

### Ask questions by typing

Use the **⌨️ Type a question** section at the bottom of the Voice Chat tab as a
text-only fallback.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key for GPT answers; if absent, raw context is shown |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (Server mode only) |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant API key for Qdrant Cloud |
| `VIBEVOICE_ASR_MODEL` | `microsoft/VibeVoice-ASR` | HuggingFace model ID for speech recognition |

All settings can also be changed at runtime from the **⚙️ Settings** sidebar.

---

## Project Structure

```
agentvoicerag/
├── app.py                  # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # URL scraping, chunking, Qdrant upsert
│   ├── retrieval.py        # Qdrant semantic search
│   ├── voice.py            # VibeVoice-ASR wrapper + gTTS
│   └── llm.py              # OpenAI response generation
├── docker-compose.yml      # Qdrant server
├── requirements.txt
├── .env.example
└── README.md
```

---

## VibeVoice-ASR

[VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) is a 7 B
parameter unified speech-to-text model from Microsoft Research that:

- Processes **up to 60 minutes** of audio in a single pass
- Performs **speaker diarization** (Who), **timestamping** (When), and **transcription** (What) jointly
- Supports **50+ languages** and code-switching
- Supports **custom hotwords** for domain-specific accuracy

It is available natively in 🤗 Transformers ≥ 5.3.0:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="microsoft/VibeVoice-ASR")
result = pipe("path/to/audio.wav")
print(result["text"])
```

---

## Qdrant

[Qdrant](https://qdrant.tech) is a high-performance vector similarity search
engine.  This project supports two modes:

| Mode | Setup | Use case |
|---|---|---|
| **In-Memory** | No setup required | Quick demos, development |
| **Server** | `docker compose up -d` | Persistent storage, production |

---

## License

MIT License – see [LICENSE](LICENSE) for details.

VibeVoice-ASR is © Microsoft and licensed under the
[MIT License](https://huggingface.co/microsoft/VibeVoice-ASR).  
Please review Microsoft's
[responsible AI guidelines](https://www.microsoft.com/en-us/ai/responsible-ai)
before deploying in production.
