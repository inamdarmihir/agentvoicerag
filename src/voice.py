"""
Voice I/O:
  - ASR  : Microsoft VibeVoice-ASR  (transformers >= 5.3.0)
  - TTS  : gTTS (lightweight, no GPU needed)

VibeVoice-ASR is a 7 B-parameter model trained for up to 60-minute
long-form audio.  It is available on Hugging Face as
``microsoft/VibeVoice-ASR`` and is included in the Transformers library
starting at version 5.3.0 (released 2026-03-06).

On machines without a GPU, loading and running the full 7 B model may be
slow.  Set ``VIBEVOICE_ASR_MODEL`` to a different HF model ID or ``whisper``
to use OpenAI Whisper (``openai/whisper-base``) as a lighter alternative.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ASR_MODEL = os.getenv("VIBEVOICE_ASR_MODEL", "microsoft/VibeVoice-ASR")
FALLBACK_ASR_MODEL = "openai/whisper-base"
TTS_LANG = "en"


# ---------------------------------------------------------------------------
# ASR – VibeVoice-ASR
# ---------------------------------------------------------------------------

class VibeVoiceASR:
    """Wrapper around the VibeVoice-ASR HuggingFace pipeline."""

    def __init__(self, model_id: str = DEFAULT_ASR_MODEL):
        from transformers import pipeline as hf_pipeline  # lazy import

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(
            "Loading ASR model '%s' on %s …",
            model_id,
            "CUDA" if device == 0 else "CPU",
        )
        try:
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=dtype,
                device=device,
            )
            self._model_id = model_id
        except Exception as exc:
            logger.warning(
                "Failed to load '%s' (%s). Falling back to '%s'.",
                model_id,
                exc,
                FALLBACK_ASR_MODEL,
            )
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=FALLBACK_ASR_MODEL,
                device=device,
            )
            self._model_id = FALLBACK_ASR_MODEL

    @property
    def model_id(self) -> str:
        return self._model_id

    def transcribe(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """
        Transcribe raw audio bytes (WAV / MP3 / OGG / …) to text.

        Parameters
        ----------
        audio_bytes:
            Raw audio file content (e.g. from ``st.audio_input()``).
        sample_rate:
            Optional hint.  If None the sample rate is read from the file.
        """
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = audio_array.astype(np.float32)

        result = self._pipe(
            {"array": audio_array, "sampling_rate": sr},
            return_timestamps=False,
        )
        text: str = result.get("text", "").strip()  # type: ignore[union-attr]
        logger.info("Transcription (%d chars): %s …", len(text), text[:80])
        return text


# ---------------------------------------------------------------------------
# TTS – gTTS
# ---------------------------------------------------------------------------

def text_to_speech(text: str, lang: str = TTS_LANG) -> bytes:
    """
    Convert *text* to MP3 audio bytes using gTTS.

    Returns raw MP3 bytes that can be passed to ``st.audio()``.
    """
    from gtts import gTTS  # lazy import

    if not text.strip():
        return b""

    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
