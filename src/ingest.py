"""Document ingestion: load files -> chunk -> embed -> store in Qdrant."""

import csv
import hashlib
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import RAGConfig


def load_documents(docs_dir: str) -> list[dict]:
    """Load only .csv files. Returns [{source, text}]."""
    docs = []
    for path in sorted(Path(docs_dir).rglob("*.csv")):
        if not path.is_file():
            continue
        # one chunk per row: "column: value" lines
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.DictReader(f))
        text = "\n\n".join(
            "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            for row in rows
        )
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    # normalize whitespace, keep paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # paragraph itself too long -> hard split with overlap
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap :]
            current = para
    if current:
        chunks.append(current)
    return chunks


def get_client(cfg: RAGConfig) -> QdrantClient:
    return QdrantClient(path=cfg.qdrant_path)


def ensure_collection(client: QdrantClient, cfg: RAGConfig, vector_size: int) -> None:
    """Create the collection if missing. If it exists but was built with a
    different embedding size (e.g. after switching models), drop and rebuild
    it — mixed-dimension vectors can never be queried together."""
    if not client.collection_exists(cfg.collection_name):
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return
    try:
        vectors = client.get_collection(cfg.collection_name).config.params.vectors
        existing_size = vectors.size if hasattr(vectors, "size") else vectors["size"]
    except Exception:
        existing_size = None
    if existing_size is not None and existing_size != vector_size:
        print(
            f"Collection '{cfg.collection_name}' has {existing_size}-dim vectors "
            f"but the current model produces {vector_size}-dim vectors. "
            "Dropping and rebuilding it."
        )
        client.delete_collection(cfg.collection_name)
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def ingest(docs_dir: str, cfg: RAGConfig | None = None, model=None) -> int:
    """Ingest every document in docs_dir. Returns number of chunks stored."""
    cfg = cfg or RAGConfig()
    docs = load_documents(docs_dir)
    if not docs:
        print(f"No documents found in {docs_dir}")
        return 0

    if model is None:
        print(f"Loading embedding model '{cfg.embedding_model}'...")
        model = SentenceTransformer(cfg.embedding_model)
    client = get_client(cfg)
    ensure_collection(client, cfg, model.get_embedding_dimension())

    items = []
    for doc in docs:
        for i, chunk in enumerate(
            chunk_text(doc["text"], cfg.chunk_size, cfg.chunk_overlap)
        ):
            items.append((doc["source"], i, chunk))
    if not items:
        print("No chunks to ingest")
        return 0

    # Batch embedding: one encode call for all chunks instead of one per
    # chunk. Much faster, and lets the model use the hardware properly.
    print(f"Embedding {len(items)} chunks in batches...")
    vectors = model.encode(
        [chunk for _, _, chunk in items],
        batch_size=32,
        show_progress_bar=True,
    ).tolist()

    points = []
    for (source, i, chunk), vector in zip(items, vectors):
        # stable id -> re-ingesting the same chunk overwrites instead of duplicating
        point_id = hashlib.md5(f"{source}#{i}#{chunk[:64]}".encode()).hexdigest()
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"source": source, "chunk_index": i, "text": chunk},
            )
        )

    client.upsert(collection_name=cfg.collection_name, points=points)
    print(f"Ingested {len(docs)} document(s) -> {len(points)} chunks into '{cfg.collection_name}'")
    return len(points)
