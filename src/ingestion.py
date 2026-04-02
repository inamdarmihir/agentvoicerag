"""
URL ingestion: scrape → chunk → embed → store in Qdrant.
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "agentvoicerag"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def get_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a sentence-transformer encoder (cached by Streamlit)."""
    return SentenceTransformer(model_name)


def ensure_collection(client: QdrantClient, dim: int = EMBEDDING_DIM) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)


def scrape_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return clean text content."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AgentVoiceRAG/1.0; "
            "+https://github.com/inamdarmihir/agentvoicerag)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text


def ingest_url(
    url: str,
    client: QdrantClient,
    encoder: SentenceTransformer,
) -> Tuple[int, str]:
    """
    Scrape *url*, chunk the text, embed each chunk, and upsert into Qdrant.

    Returns (number_of_chunks_stored, page_title).
    """
    ensure_collection(client, dim=encoder.get_sentence_embedding_dimension())

    raw_text = scrape_url(url)
    if not raw_text.strip():
        raise ValueError(f"No text content found at {url!r}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks: List[str] = splitter.split_text(raw_text)
    if not chunks:
        raise ValueError(f"Text could not be chunked from {url!r}")

    embeddings = encoder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chunk, "source": url},
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Ingested %d chunks from %s", len(chunks), url)
    return len(chunks), url


def get_ingested_sources(client: QdrantClient) -> List[str]:
    """Return a deduplicated list of all source URLs stored in Qdrant."""
    try:
        ensure_collection(client)
        scroll_result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=10_000,
        )
        sources = sorted({p.payload.get("source", "") for p in scroll_result if p.payload})
        return [s for s in sources if s]
    except Exception:
        return []
