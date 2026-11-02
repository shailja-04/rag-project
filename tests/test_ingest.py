"""Unit tests for document ingestion and CSV loading."""

import csv
from pathlib import Path
from src.ingest import load_documents, chunk_text


def test_load_documents_only_csv(tmp_path: Path):
    """Ensure load_documents only reads .csv files and ignores others."""
    # Create sample CSV
    csv_file = tmp_path / "sample.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "definition"])
        writer.writerow(["Python", "A popular programming language"])

    # Create dummy txt and pdf files that should be ignored
    (tmp_path / "notes.txt").write_text("Hello world")
    (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4 dummy")

    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["source"] == "sample.csv"
    assert "Python" in docs[0]["text"]
    assert "A popular programming language" in docs[0]["text"]


def test_chunk_text_preserves_boundaries():
    """Ensure chunk_text splits paragraphs properly within chunk_size limits."""
    text = "Paragraph 1: Short text.\n\nParagraph 2: Another short text."
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert "Paragraph 1" in chunks[0]
    assert "Paragraph 2" in chunks[0]


def test_chunk_text_hard_split():
    """Ensure long paragraphs exceeding chunk_size are broken down."""
    long_para = "A" * 300
    chunks = chunk_text(long_para, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
