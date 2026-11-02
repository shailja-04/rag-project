"""Unit tests for query rewriting and fallback behavior."""

from src.answer import rewrite_query
from src.config import RAGConfig


def test_rewrite_query_standalone():
    """If there is no conversation history, query is returned as-is."""
    cfg = RAGConfig()
    query = "What is a List in Python?"
    assert rewrite_query(query, history=[], cfg=cfg) == query


def test_rewrite_query_greetings_bypass():
    """Common greetings should bypass LLM rewrite and return as-is."""
    cfg = RAGConfig()
    history = [{"role": "user", "content": "What is Python?"}]
    for greeting in ["hi", "hello", "how are you", "thanks", "thank you"]:
        res = rewrite_query(greeting, history=history, cfg=cfg)
        assert res == greeting


def test_rewrite_query_empty_fallback():
    """Ensure empty or whitespace input doesn't crash."""
    cfg = RAGConfig()
    assert rewrite_query("", history=[], cfg=cfg) == ""
