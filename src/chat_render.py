"""HTML rendering for chat messages, styled like a dark RAG chatbot UI.

User messages get a pink avatar, assistant messages a yellow robot avatar,
both on full-width dark cards. Kept in its own module so it can be tested
without running Streamlit.
"""

import html

import markdown

CSS = """
<style>
/* Page title, like the reference UI */
h1.app-title { font-size: 2.6rem; font-weight: 800; margin-bottom: 0.4rem; }
/* Chat cards */
.msg { display: flex; gap: 12px; background: #1c1f26; border-radius: 14px;
       padding: 16px 18px; margin-bottom: 14px; }
.msg .avatar { width: 36px; height: 36px; min-width: 36px; border-radius: 9px;
               display: flex; align-items: center; justify-content: center;
               font-size: 20px; line-height: 1; }
.msg .avatar.user { background: #e06a6a; }
.msg .avatar.bot { background: #f2c14e; }
.msg .text { color: #f1f1f1; line-height: 1.6; overflow-wrap: anywhere; width: 100%; }
.msg .text p { margin: 0 0 0.6em 0; }
.msg .text p:last-child { margin-bottom: 0; }
.msg .text code { background: #2a2e38; padding: 2px 6px; border-radius: 5px; font-family: monospace; font-size: 0.9em; }
.msg .text pre { background: #16181d; border: 1px solid #2e3340; padding: 12px; border-radius: 8px;
                overflow-x: auto; position: relative; }
/* Code blocks and copy button */
.msg .text pre {
    background: #15181f;
    border: 1px solid #2e3340;
    padding: 12px 14px;
    border-radius: 8px;
    overflow-x: auto;
    position: relative;
    margin: 10px 0;
}
.msg .text pre code {
    background: none;
    padding: 0;
    font-size: 0.9em;
    color: #e2e8f0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.copy-code-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: #262a34;
    color: #b0b8c4;
    border: 1px solid #3b4252;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
}
.copy-code-btn:hover {
    background: #363c4a;
    color: #f2f2f2;
}

/* Markdown Tables styling */
.msg .text table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 16px 0;
    font-size: 0.92rem;
    background: #15181f;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #2e3340;
}
.msg .text th {
    background-color: #242833;
    color: #f2c14e;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid #3b4252;
}
.msg .text td {
    padding: 9px 14px;
    border-bottom: 1px solid #222631;
    color: #e2e8f0;
}
.msg .text tr:last-child td {
    border-bottom: none;
}
.msg .text tr:hover td {
    background-color: #1c202a;
}

/* Latency breakdown badge */
.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #15181f;
    border: 1px solid #2e3340;
    color: #94a3b8;
    font-size: 0.78rem;
    font-family: ui-monospace, monospace;
    padding: 3px 10px;
    border-radius: 6px;
    margin-top: 6px;
}
.meta-badge span.highlight {
    color: #f2c14e;
    font-weight: 600;
}

/* Source Card with Visual Confidence Bar */
.source-card {
    background: #15181f;
    border: 1px solid #2a2e39;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.source-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
}
.confidence-container {
    display: flex;
    align-items: center;
    gap: 8px;
}
.confidence-bar-bg {
    width: 70px;
    height: 6px;
    background: #252830;
    border-radius: 3px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.confidence-bar-fill.high { background: #10b981; }      /* Green for high */
.confidence-bar-fill.medium { background: #f59e0b; }    /* Orange for medium */
.confidence-bar-fill.low { background: #ef4444; }       /* Red for low */
.confidence-label {
    font-size: 0.78rem;
    font-family: ui-monospace, monospace;
    font-weight: 600;
}
.confidence-label.high { color: #10b981; }
.confidence-label.medium { color: #f59e0b; }
.confidence-label.low { color: #ef4444; }

/* Starter chips styling */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 16px 0;
}

/* Slimmer input area, matching the reference layout */
div[data-testid="stChatInput"] { padding-bottom: 8px; }
</style>
<script>
function attachCopyButtons() {
    document.querySelectorAll('.msg .text pre').forEach((pre) => {
        if (!pre.querySelector('.copy-code-btn')) {
            const btn = document.createElement('button');
            btn.className = 'copy-code-btn';
            btn.innerText = '📋 Copy';
            btn.onclick = () => {
                const code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
                navigator.clipboard.writeText(code).then(() => {
                    btn.innerText = '✅ Copied!';
                    setTimeout(() => { btn.innerText = '📋 Copy'; }, 2000);
                });
            };
            pre.appendChild(btn);
        }
    });
}
// Attach on load and observer for streamed content
attachCopyButtons();
const observer = new MutationObserver(attachCopyButtons);
observer.observe(document.body, { childList: true, subtree: true });
</script>

"""


def latency_badge_html(latency: dict) -> str:
    """HTML for latency breakdown (Retrieval + LLM)."""
    retrieval_ms = latency.get("retrieval_ms", 0)
    llm_ms = latency.get("llm_ms", 0)
    total_ms = round(retrieval_ms + llm_ms, 1)
    return (
        f'<div class="meta-badge">'
        f'⚡ Total: <span class="highlight">{total_ms}ms</span> '
        f'(Retrieval: <span class="highlight">{retrieval_ms}ms</span> | '
        f'LLM: <span class="highlight">{llm_ms}ms</span>)'
        f'</div>'
    )


def source_card_html(index: int, source: str, score: float, text: str) -> str:
    """HTML for a source card with visual confidence indicator and progress bar."""
    # Score classification: cosine similarity usually ranges between 0.3 - 0.9 in dense search
    percent = min(max(int(score * 100), 10), 100)
    if score >= 0.55:
        level = "high"
        label = "High Confidence"
    elif score >= 0.40:
        level = "medium"
        label = "Moderate"
    else:
        level = "low"
        label = "Low Confidence"

    snippet = html.escape(text[:350] + ("…" if len(text) > 350 else "")).replace("\n", " ")

    return f"""
    <div class="source-card">
        <div class="source-header">
            <span class="source-name">[{index}] 📄 {html.escape(source)}</span>
            <div class="confidence-container">
                <span class="confidence-label {level}">{label} ({score})</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill {level}" style="width: {percent}%;"></div>
                </div>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.45;">
            {snippet}
        </div>
    </div>
    """


def message_html(role: str, content: str) -> str:
    """One chat card as HTML. Assistant content is rendered as Markdown with tables support."""
    if role == "user":
        avatar, body = '<div class="avatar user">🙂</div>', html.escape(content).replace(
            "\n", "<br>"
        )
        cls = "user"
    else:
        avatar, body = '<div class="avatar bot">🤖</div>', markdown.markdown(
            content, extensions=["fenced_code", "tables"]
        )
        cls = "bot"
    return f'<div class="msg {cls}">{avatar}<div class="text">{body}</div></div>'


