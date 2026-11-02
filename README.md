# RAG Chatbot

Ask questions over your own documents — with a dark chatbot UI. A complete,
production-style RAG pipeline in Python:

**PDF/CSV** (TXT/MD also supported) → chunking → **BGE-base embeddings**
→ **Qdrant** vector search → **Groq** LLM answer with citations.
(CSV rows become one chunk per row.)

## What's beyond fresher level

- **One main file** — `app.py` is the only main file and the only
  interface. It loads the `.env` file and the documents into the vector
  DB at startup, then starts the chat UI. Just run `python app.py`.
- **Optimized startup** — the fingerprint check needs no model, so when
  the index is already fresh, `python app.py` starts in seconds without
  reloading the embedding model. The model loads only when a rebuild is
  actually needed.
- **Batched embedding** — ingest embeds all chunks in batches of 32 with
  a single encode call, instead of one call per chunk.
- **Conversational RAG** — multi-turn chat memory; follow-ups like *"what about
  its pricing?"* are rewritten into standalone questions before retrieval
- **Streaming answers** with live citations and expandable source chunks
- **Better embeddings** — BAAI/bge-base-en-v1.5 (768-dim) with the BGE query
  instruction prefix, a large step up from MiniLM for retrieval quality
- **Source-filtered retrieval** — in code, retrieval accepts a source list,
  so you can answer from CSV files only, PDF files only, or one file
  (`retrieve(question, sources=["genai_concepts.csv"])`)
- **CSV-first retrieval** — CSVs are first-class documents: each row is
  indexed as its own chunk, so glossary-style CSVs answer term questions
  directly (try: "What is a token?")
- **Retrieval evaluation** — hit-rate@k on a labeled question set, runnable
  from the CLI (30 questions: GenAI concepts + Python basics)
- **Self-maintaining vector DB** — Qdrant in local file mode, no server
  needed. A fingerprint of the docs + model + chunk settings is saved, so
  startup re-ingests only when something changed; switching embedding
  models rebuilds the collection instead of crashing on dimension mismatch
- **Simple UI, settings in code** — the web UI is just questions and
  answers; model, top-k, temperature and the API key live in
  `src/config.py` and `.env`, and documents live in `data/sample_docs`

## Architecture

```
documents ──► ingest.py ──► Qdrant (local file mode)
                                  │
user question ──► rewrite follow-up ──► retrieve.py ──► top-k chunks
                                                              │
                              ┌── answer.py ◄── chat history ──┘
                              │         │
                              ▼         ▼
                        Groq LLM ──► streamed answer + [1],[2] citations
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Settings live in code: copy .env.example to .env and add your free
#    Groq key (console.groq.com). Model/top-k/temperature are in .env too.
cp .env.example .env

# 2) Run the main file. It loads .env, loads data/sample_docs into the
#    vector DB (first run downloads the ~440MB embedding model once),
#    then starts the chat UI at http://localhost:8501
python app.py

# UI only (docs must already be ingested):
streamlit run app.py
```

## Project layout

```
├── app.py           # MAIN FILE: startup loading (env + docs into vector DB)
│                    # + minimal chat UI. Run: python app.py
├── .streamlit/config.toml  # dark theme
├── src/
│   ├── config.py    # all settings, overridable via env vars / .env
│   ├── ingest.py    # load → chunk → embed → upsert to Qdrant
│   ├── retrieve.py  # embed question (+ BGE instruction) → top-k chunks,
│   │                # optional source filter
│   ├── answer.py    # follow-up rewriting + prompt building + Groq (streaming)
│   ├── chat_render.py  # chat-card HTML (avatars, dark cards, Markdown)
│   └── eval.py      # hit-rate@k on labeled questions
├── data/
│   ├── sample_docs/ # bundled data — ingest + answer over these:
│   │                # genai_concepts_guide.pdf (9-page guide) +
│   │                # genai_concepts.csv (34-term glossary) +
│   │                # python_basics_guide.pdf (14-page guide) +
│   │                # python_basics.csv (glossary, one chunk per row)
│   └── uploads/     # your own files (gitignored)
├── eval/questions.json  # 30 labeled eval questions (GenAI + Python basics)
└── requirements.txt
```

## Tweaking it

| Setting | Env var / UI | Default |
|---|---|---|
| Chunks per answer | `TOP_K` | 4 |
| Chunk size / overlap | `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 100 |
| Embedding model (or local path) | `EMBEDDING_MODEL` | BAAI/bge-base-en-v1.5 |
| Groq model | `GROQ_MODEL` | llama-3.3-70b-versatile |
| Temperature | `TEMPERATURE` / `.env` | 0.2 |

## Notes

- The model answers **only** from your documents — if the answer isn't in the
  context it says so instead of hallucinating.
- Document chunks are upserted by stable id, so re-ingesting never duplicates.
