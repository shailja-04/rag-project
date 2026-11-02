"""Answering: stuff retrieved chunks into a prompt and ask the LLM, with citations.

Supports multi-turn conversation (history) and streaming. Follow-up questions
like "what about its pricing?" are first rewritten into standalone questions
so retrieval works on them.
"""

from .config import RAGConfig
from .retrieve import retrieve

SYSTEM_PROMPT = """You answer questions based on the provided document context.
Rules:
- For greetings or pleasantries (e.g. 'hi', 'how are you'), respond politely and mention that you are a helpful assistant ready to answer questions about the documents.
- If a factual question cannot be answered using the provided context, say "I don't know based on the provided documents."
- Cite every factual claim like [1], [2] matching the source numbers.
- Keep answers clear, structured, and easy to read.
- When explaining differences or comparing items, ALWAYS format the key differences as a clean Markdown table with headers (e.g., | Feature | Item A | Item B |). Ensure each table row is on its own separate line.
- When providing code examples, ALWAYS use fenced markdown code blocks with the language tag (e.g. ```python ... ```).
- If conversation history is provided, use it to resolve pronouns and references.
- Synthesize information from multiple context chunks when answering broader questions, summaries, or comparisons.
"""

REWRITE_PROMPT = """Look at the conversation history and the follow-up question.
If the follow-up question contains pronouns like 'it', 'they', 'this', 'that', or refers to a concept from earlier turns, rewrite it into a clear, standalone question by replacing the pronouns with the specific subjects being discussed.
If the question is already clear, complete, or a greeting, return the question as-is.
Reply with ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}
Standalone question:"""


def _groq_client(cfg: RAGConfig):
    if not cfg.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and export GROQ_API_KEY=..."
        )
    from groq import Groq

    return Groq(api_key=cfg.groq_api_key)


def rewrite_query(question: str, history: list[dict], cfg: RAGConfig) -> str:
    """Turn a follow-up into a standalone question using chat history."""
    if not history:
        return question.strip()
    
    # Greetings / basic conversational turns don't need rewriting
    greetings = {"hi", "hello", "hey", "how are you", "how r u", "thanks", "thank you", "bye"}
    if question.strip().lower() in greetings:
        return question.strip()

    try:
        client = _groq_client(cfg)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-6:]
        )
        completion = client.chat.completions.create(
            model=cfg.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        history=history_text, question=question
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = completion.choices[0].message.content.strip().strip('"').strip()
        # Fallback to original question if the model returns empty or whitespace
        return rewritten if rewritten else question.strip()
    except Exception:
        return question.strip()


def build_messages(question: str, chunks: list[dict], history: list[dict] | None) -> list[dict]:
    context = "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    )
    return messages


def answer(
    question: str,
    cfg: RAGConfig | None = None,
    history: list[dict] | None = None,
    stream: bool = False,
    model=None,
    source_filter: list[str] | None = None,
):
    """Ask the RAG pipeline with latency profiling.

    Returns {answer, sources, chunks_used, rewritten_query, latency} when stream=False.
    Returns (generator, chunks, sources, rewritten, latency_dict) when stream=True.
    """
    import time
    cfg = cfg or RAGConfig()
    history = history or []

    rewritten = rewrite_query(question, history, cfg)
    
    t_start_retrieval = time.perf_counter()
    chunks = retrieve(rewritten, cfg, model=model, sources=source_filter)
    t_retrieval = (time.perf_counter() - t_start_retrieval) * 1000

    latency = {
        "retrieval_ms": round(t_retrieval, 1),
        "llm_ms": 0.0,
    }

    if not chunks:
        if stream:
            return iter(["No relevant documents found."]), [], [], rewritten, latency
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks_used": 0,
            "rewritten_query": rewritten,
            "latency": latency,
        }

    client = _groq_client(cfg)
    messages = build_messages(question, chunks, history)

    t_start_llm = time.perf_counter()
    completion = client.chat.completions.create(
        model=cfg.groq_model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=600,
        stream=stream,
    )

    sources = [{"source": c["source"], "score": c["score"]} for c in chunks]
    if stream:
        def gen():
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            latency["llm_ms"] = round((time.perf_counter() - t_start_llm) * 1000, 1)

        return gen(), chunks, sources, rewritten, latency

    latency["llm_ms"] = round((time.perf_counter() - t_start_llm) * 1000, 1)
    return {
        "answer": completion.choices[0].message.content,
        "sources": sources,
        "chunks_used": len(chunks),
        "rewritten_query": rewritten,
        "latency": latency,
    }

