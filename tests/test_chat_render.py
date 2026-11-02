"""Unit tests for chat HTML rendering, tables, and latency badges."""

from src.chat_render import message_html, latency_badge_html, source_card_html


def test_user_message_html():
    """User message should render user avatar and escaped content."""
    html = message_html("user", "Hello <world> & test")
    assert "avatar user" in html
    assert "&lt;world&gt; &amp; test" in html


def test_assistant_table_rendering():
    """Assistant message with markdown table must render an HTML table."""
    markdown_table = """
| Feature | List | Tuple |
| :--- | :--- | :--- |
| Mutability | Mutable | Immutable |
"""
    html = message_html("assistant", markdown_table)
    assert "<table>" in html
    assert ">Feature</th>" in html
    assert ">Mutable</td>" in html


def test_latency_badge_html():
    """Ensure latency values are properly formatted in badge."""
    badge = latency_badge_html({"retrieval_ms": 45.2, "llm_ms": 350.8})
    assert "396.0ms" in badge  # Total
    assert "45.2ms" in badge
    assert "350.8ms" in badge


def test_source_card_confidence_levels():
    """Ensure high, medium, and low scores get the right color-coded classes."""
    card_high = source_card_html(1, "test.csv", 0.75, "Sample text")
    assert "High Confidence" in card_high
    assert "confidence-bar-fill high" in card_high

    card_med = source_card_html(2, "test.csv", 0.45, "Sample text")
    assert "Moderate" in card_med
    assert "confidence-bar-fill medium" in card_med

    card_low = source_card_html(3, "test.csv", 0.25, "Sample text")
    assert "Low Confidence" in card_low
    assert "confidence-bar-fill low" in card_low
