# 🎙️ AgentVoiceRAG

> **Agentic Voice RAG** – ask questions about any web page using your voice.
> Powered by [Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) for speech recognition, [Qdrant](https://qdrant.tech) as the vector database, and quantized [HuggingFace LLMs](https://huggingface.co/models) for answer generation — no OpenAI API key required.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────┐
 │                        USER INTERFACE                        │
 │                     Streamlit (app.py)                       │
 └───────────────┬────────────────────────┬────────────────────┘
                 │                        │
         Voice Input 🎤           URL Ingestion 📥
                 │                        │
                 ▼                        ▼
   ┌─────────────────────┐    ┌──────────────────────────┐
   │  Microsoft           │    │  Web Scraper             │
   │  VibeVoice-ASR       │    │  (requests + BeautifulSoup│
   │  (ASR / STT)         │    │   + lxml)                │
   └────────┬────────────┘    └────────────┬─────────────┘
            │ transcript                   │ raw text
            │                             ▼
            │                 ┌──────────────────────────┐
            │                 │  Text Chunker             │
            │                 │  RecursiveCharacterText   │
            │                 │  Splitter (512 tok,       │
            │                 │  64 tok overlap)          │
            │                 └────────────┬─────────────┘
            │                             │ chunks
            │                             ▼
            │                 ┌──────────────────────────┐
            │                 │  Embeddings               │
            │                 │  all-MiniLM-L6-v2         │
            │                 │  (sentence-transformers)  │
            │                 └────────────┬─────────────┘
            │                             │ 384-dim vectors
            │                             ▼
            │                 ┌──────────────────────────┐
            │                 │  Qdrant Vector DB         │
            │◄── query ──────►│  (in-memory or server)   │
            │   embedding     │  cosine similarity search │
            │                 └────────────┬─────────────┘
            │                             │ top-K chunks
            ▼                             ▼
   ┌──────────────────────────────────────────────────────┐
   │          HuggingFace Quantized LLM                    │
   │  Default: Qwen/Qwen2.5-1.5B-Instruct                 │
   │  Quantization: none | 4bit (NF4) | 8bit (LLM.int8()) │
   └───────────────────────────┬──────────────────────────┘
                               │ answer text
                               ▼
                   ┌───────────────────────┐
                   │  gTTS                 │
                   │  Text-to-Speech 🔊    │
                   └───────────────────────┘
```

### Component Summary

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | [Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) (7 B, via 🤗 Transformers ≥ 5.3.0) |
| Vector Database | [Qdrant](https://qdrant.tech) (in-memory or server mode) |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers (384-dim) |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` (512 tokens, 64 overlap) |
| LLM | HuggingFace causal LM — default `Qwen/Qwen2.5-1.5B-Instruct` (no API key required) |
| Quantization | BitsAndBytes NF4 (4-bit) or LLM.int8() (8-bit) — CUDA only |
| Text-to-Speech | [gTTS](https://pypi.org/project/gTTS/) |
| UI | [Streamlit](https://streamlit.io) |

---

## Features

- 📥 **URL ingestion** – scrape any public web page, chunk the text, embed it, and store it in Qdrant
- 🎤 **Voice input** – record questions directly in the browser; VibeVoice-ASR handles up to 60 minutes of audio in a single pass
- 🔍 **Semantic search** – top-K cosine similarity retrieval from Qdrant
- 🤖 **Local RAG answers** – quantized HuggingFace LLM generates grounded answers from retrieved context — no external API needed
- 🔊 **Voice output** – answers are read back using gTTS
- ⌨️ **Text fallback** – type questions when a microphone is unavailable
- 📜 **Conversation history** – full Q&A history displayed in the chat tab

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/inamdarmihir/agentvoicerag.git
cd agentvoicerag
pip install -r requirements.txt
```

> **Note**: VibeVoice-ASR is a 7 B parameter model. A CUDA-capable GPU is
> strongly recommended. On CPU it will still work but inference will be slow.
> Set `VIBEVOICE_ASR_MODEL=openai/whisper-base` in `.env` for a lightweight
> CPU-friendly alternative.

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — all settings have sensible defaults; no API key is required
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
overlap, generates embeddings with `all-MiniLM-L6-v2`, and upserts them into Qdrant.

### Ask questions with your voice

1. Click the **🎙️ Voice Chat** tab.
2. Click the microphone icon and record your question.
3. Click **🔍 Transcribe & Answer**.

The pipeline will:
1. Transcribe your audio with VibeVoice-ASR.
2. Embed the transcript and retrieve the top-K most relevant chunks from Qdrant.
3. Feed the context + question to the HuggingFace LLM and return an answer.
4. Read the answer aloud using gTTS.

### Ask questions by typing

Use the **⌨️ Type a question** section at the bottom of the Voice Chat tab as a
text-only fallback.

---

## Configuration

All settings can also be changed at runtime from the **⚙️ Settings** sidebar.

| Environment Variable | Default | Description |
|---|---|---|
| `HF_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model repo ID for answer generation |
| `HF_LLM_QUANTIZATION` | `none` | Quantization level: `none`, `4bit`, or `8bit` (CUDA only) |
| `HF_TOKEN` | _(empty)_ | HuggingFace access token — required only for gated models (Llama, Gemma, …) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (Server mode only) |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant API key for Qdrant Cloud |
| `VIBEVOICE_ASR_MODEL` | `microsoft/VibeVoice-ASR` | HuggingFace model ID for speech recognition |

---

## Project Structure

```
agentvoicerag/
├── app.py                  # Streamlit UI & pipeline orchestration
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # URL scraping, chunking, embedding, Qdrant upsert
│   ├── retrieval.py        # Qdrant cosine-similarity search & context builder
│   ├── voice.py            # VibeVoice-ASR wrapper (STT) + gTTS (TTS)
│   └── llm.py              # HFQuantizedLLM — BitsAndBytes 4-bit/8-bit inference
├── docker-compose.yml      # Qdrant server
├── requirements.txt
├── .env.example
└── README.md
```

---

## HuggingFace LLM

The LLM layer (`src/llm.py`) loads any HuggingFace causal language model via the
`transformers` pipeline. No OpenAI API key is required.

### Supported free models (no token needed)

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | Runs on any CPU |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | **Default** — fast on CPU |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B | Very small footprint |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8 B | Strong quality |
| `google/gemma-2-2b-it` | 2.0 B | Google Gemma 2 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7.0 B | GPU recommended |

### Gated models (HF token required)

| Model | Size |
|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | 3.0 B |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8.0 B |

### Quantization

| Mode | Requirement | Description |
|---|---|---|
| `none` | CPU / MPS / CUDA | Full precision (fp32 on CPU/MPS, fp16 on CUDA) |
| `4bit` | CUDA + bitsandbytes | NF4 double-quantization — ~4× memory reduction |
| `8bit` | CUDA + bitsandbytes | LLM.int8() — ~2× memory reduction |

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

> **Lightweight alternative**: set `VIBEVOICE_ASR_MODEL=openai/whisper-base` for
> CPU-friendly transcription at the cost of some accuracy.

---

## Qdrant

[Qdrant](https://qdrant.tech) is a high-performance vector similarity search
engine. This project supports two modes:

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
