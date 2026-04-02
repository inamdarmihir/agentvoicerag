"""
LLM response generation using OpenAI chat completions.

Falls back to a simple context-only summary when no API key is available.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions exclusively based "
    "on the provided context retrieved from ingested web pages. "
    "If the context does not contain enough information to answer the question, "
    "say so clearly and suggest the user ingest more relevant URLs. "
    "Be concise and factual."
)


def generate_response(
    query: str,
    context: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a RAG answer for *query* grounded in *context*.

    Uses OpenAI chat completions when an API key is available;
    otherwise returns a plain-text fallback that surfaces the raw context.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

    if api_key:
        return _openai_response(query, context, api_key, model)
    else:
        return _fallback_response(query, context)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_response(
    query: str,
    context: str,
    api_key: str,
    model: str,
) -> str:
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context (retrieved from ingested web pages):\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"Question: {query}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    answer: str = resp.choices[0].message.content.strip()
    logger.info("LLM response (%d chars).", len(answer))
    return answer


def _fallback_response(query: str, context: str) -> str:
    """Return a simple answer when no OpenAI key is configured."""
    if not context.strip():
        return (
            "No relevant context found. "
            "Please ingest at least one URL and try again."
        )
    return (
        "**[No OpenAI API key configured – showing raw retrieved context]**\n\n"
        f"**Question:** {query}\n\n"
        f"**Retrieved context:**\n\n{context}"
    )
