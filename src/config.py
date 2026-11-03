"""Central configuration for the RAG pipeline.

All settings live here, in code — not in the UI. Override anything with
environment variables or a `.env` file (see .env.example), e.g.:
    GROQ_API_KEY=... python app.py
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Always load the project's own .env (next to src/), no matter which
# folder the app is launched from. Real environment variables still win.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class RAGConfig:
    # Embeddings (sentence-transformers model, runs locally).
    # BGE-base is a big step up from MiniLM for retrieval quality; first run
    # downloads ~440MB once. Override with EMBEDDING_MODEL if needed.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # BGE models want queries prefixed with an instruction (documents are NOT
    # prefixed). Set QUERY_INSTRUCTION="" to disable, or to another string.
    # For non-BGE models this defaults to empty.
    query_instruction: str | None = os.getenv("QUERY_INSTRUCTION")

    # Qdrant runs in local file mode - no server needed
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")
    collection_name: str = os.getenv("COLLECTION_NAME", "doc_qa")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))

    # LLM (Groq - free tier)
    groq_api_key: str = ""
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    def __post_init__(self):
        if self.query_instruction is None:
            if "bge" in self.embedding_model.lower():
                self.query_instruction = (
                    "Represent this sentence for searching relevant passages: "
                )
            else:
                self.query_instruction = ""

        # Check environment variable first, then Streamlit Cloud secrets
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not self.groq_api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    self.groq_api_key = str(st.secrets["GROQ_API_KEY"])
            except Exception:
                pass
