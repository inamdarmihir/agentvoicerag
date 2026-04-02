"""
LLM response generation using quantized HuggingFace models.

Supported quantization methods
--------------------------------
none  – full precision (fp32 on CPU / MPS, fp16 on CUDA)
4bit  – BitsAndBytes NF4 double-quantization  (CUDA only)
8bit  – BitsAndBytes LLM.int8()               (CUDA only)

Popular free models (no HF token required)
-------------------------------------------
Qwen/Qwen2.5-0.5B-Instruct          0.5 B  – runs on any CPU
Qwen/Qwen2.5-1.5B-Instruct          1.5 B  – fast on CPU  (default)
TinyLlama/TinyLlama-1.1B-Chat-v1.0  1.1 B  – very small
microsoft/Phi-3-mini-4k-instruct     3.8 B  – strong quality
google/gemma-2-2b-it                 2.0 B  – Google Gemma 2
mistralai/Mistral-7B-Instruct-v0.3   7.0 B  – GPU recommended

Gated models (HF token required)
---------------------------------
meta-llama/Llama-3.2-3B-Instruct     3.0 B
meta-llama/Meta-Llama-3-8B-Instruct  8.0 B
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions exclusively based "
    "on the provided context retrieved from ingested web pages. "
    "If the context does not contain enough information to answer the question, "
    "say so clearly and suggest the user ingest more relevant URLs. "
    "Be concise and factual."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFQuantizedLLM:
    """
    Loads a HuggingFace causal-LM with optional BitsAndBytes quantization.

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID, e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``.
    quantization : str or None
        ``"4bit"`` (NF4 double-quant) or ``"8bit"`` (LLM.int8()).
        Both require CUDA + bitsandbytes.  On CPU/MPS this is silently
        ignored and full precision is used instead.
    hf_token : str or None
        HuggingFace access token for gated models (Llama, Gemma, …).
    max_new_tokens : int
        Maximum tokens to generate per response.
    temperature : float
        Sampling temperature.  0.0 → greedy decoding.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        quantization: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline as hf_pipeline,
        )

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        device = _best_device()
        token = hf_token or os.getenv("HF_TOKEN", "") or None

        logger.info(
            "Loading LLM '%s'  quantization=%s  device=%s",
            model_id,
            quantization,
            device,
        )

        # BitsAndBytes quantization is CUDA-only
        if quantization in ("4bit", "8bit") and device != "cuda":
            logger.warning(
                "BitsAndBytes quantization requires CUDA — falling back to "
                "full precision on %s.",
                device,
            )
            quantization = None

        model_kwargs: dict = {}
        if token:
            model_kwargs["token"] = token

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"

        else:
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Move to device manually only when device_map was not used
        if "device_map" not in model_kwargs and device != "cpu":
            model = model.to(device)

        pipe_kwargs: dict = {"model": model, "tokenizer": tokenizer}
        if "device_map" in model_kwargs:
            pipe_kwargs["device_map"] = "auto"
        else:
            pipe_kwargs["device"] = device

        self._pipe = hf_pipeline("text-generation", **pipe_kwargs)
        self.quantization = quantization

        logger.info("LLM ready: %s  (effective quantization=%s)", model_id, quantization)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, query: str, context: str) -> str:
        """Run the RAG prompt through the model and return the answer text."""
        prompt = self._build_prompt(query, context)
        tokenizer = self._pipe.tokenizer
        eos_id = tokenizer.eos_token_id if tokenizer is not None else None
        raw = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=eos_id,
            return_full_text=False,
        )
        # pipeline returns list[dict] with "generated_text" key
        outputs: list = raw if isinstance(raw, list) else list(raw)  # type: ignore[arg-type]
        answer: str = str(outputs[0].get("generated_text", "")).strip()
        logger.info("LLM response (%d chars).", len(answer))
        return answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context: str) -> str:
        """Format the chat messages using the model's own chat template when available."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Context (retrieved from ingested web pages):\n\n"
                    f"{context}\n\n"
                    "---\n\n"
                    f"Question: {query}"
                ),
            },
        ]
        tokenizer = self._pipe.tokenizer  # type: ignore[union-attr]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            result = tokenizer.apply_chat_template(  # type: ignore[union-attr,reportOptionalMemberAccess]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(result)
        # Plain fallback for models without a registered chat template
        return (
            f"### System\n{SYSTEM_PROMPT}\n\n"
            f"### User\nContext:\n\n{context}\n\n---\n\n"
            f"Question: {query}\n\n"
            "### Assistant\n"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(
    query: str,
    context: str,
    llm: Optional[HFQuantizedLLM] = None,
) -> str:
    """
    Generate a RAG answer for *query* grounded in *context*.

    Uses *llm* (HFQuantizedLLM) when provided; falls back to a raw-context
    display when no model is loaded.
    """
    if llm is not None:
        return llm.generate(query, context)
    return _fallback_response(query, context)


def _fallback_response(query: str, context: str) -> str:
    if not context.strip():
        return (
            "No relevant context found. "
            "Please ingest at least one URL and try again."
        )
    return (
        "**[No LLM loaded – showing raw retrieved context]**\n\n"
        f"**Question:** {query}\n\n"
        f"**Retrieved context:**\n\n{context}"
    )
