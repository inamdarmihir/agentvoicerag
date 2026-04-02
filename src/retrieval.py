"""
RAG retrieval: query Qdrant and return the most relevant text chunks.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.ingestion import COLLECTION_NAME, ensure_collection

logger = logging.getLogger(__name__)

RetrievedChunk = Tuple[str, str, float]  # (text, source_url, score)


def retrieve(
    query: str,
    client: QdrantClient,
    encoder: SentenceTransformer,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[RetrievedChunk]:
    """
    Embed *query*, search Qdrant and return the top-k chunks with metadata.

    Returns a list of (text, source_url, similarity_score) tuples.
    """
    ensure_collection(client, dim=encoder.get_sentence_embedding_dimension())

    query_embedding = encoder.encode(
        query, normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
    )

    chunks: List[RetrievedChunk] = [
        (
            hit.payload.get("text", ""),
            hit.payload.get("source", ""),
            hit.score,
        )
        for hit in results.points
        if hit.payload
    ]
    logger.info("Retrieved %d chunks for query: %s", len(chunks), query[:60])
    return chunks


def build_context(chunks: List[RetrievedChunk]) -> str:
    """Combine retrieved chunks into a single context string for the LLM."""
    parts = []
    for i, (text, source, score) in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {source}\n{text}")
    return "\n\n---\n\n".join(parts)
