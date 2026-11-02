"""Retrieval: embed the question, fetch the most relevant chunks from Qdrant.

Supports filtering by document source, so the UI can answer from CSV files
only, PDF files only, a single file, or everything in the index.
"""

from sentence_transformers import SentenceTransformer

from qdrant_client.models import FieldCondition, Filter, MatchAny

from .config import RAGConfig
from .ingest import get_client


def retrieve(
    question: str,
    cfg: RAGConfig | None = None,
    model=None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Returns top-k chunks: [{text, source, score}].

    sources: optional list of source filenames to restrict retrieval to
    (e.g. ["genai_concepts.csv"]). None means search the whole index.
    """
    cfg = cfg or RAGConfig()
    if model is None:
        model = SentenceTransformer(cfg.embedding_model)
    client = get_client(cfg)

    if not client.collection_exists(cfg.collection_name):
        raise RuntimeError(
            f"Collection '{cfg.collection_name}' does not exist. Run ingest first."
        )

    query_vector = model.encode(
        (cfg.query_instruction or "") + question
    ).tolist()
    query_filter = None
    if sources:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchAny(any=sources))]
        )
    hits = client.query_points(
        collection_name=cfg.collection_name,
        query=query_vector,
        limit=cfg.top_k,
        query_filter=query_filter,
    ).points

    return [
        {
            "text": h.payload["text"],
            "source": h.payload["source"],
            "score": round(h.score, 4),
        }
        for h in hits
    ]


def list_sources(cfg: RAGConfig | None = None) -> list[str]:
    """Distinct source filenames currently in the index, sorted."""
    cfg = cfg or RAGConfig()
    client = get_client(cfg)
    if not client.collection_exists(cfg.collection_name):
        return []
    points, _ = client.scroll(
        collection_name=cfg.collection_name,
        limit=1000,
        with_payload=["source"],
        with_vectors=False,
    )
    return sorted({p.payload["source"] for p in points if "source" in p.payload})
