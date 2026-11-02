"""Main entry point for RAG Chatbot.

Two ways to run it — both end up at the same chat UI:

    python app.py          # recommended: loads .env, loads docs into the
                           # vector DB if they are missing or changed,
                           # then starts the UI
    streamlit run app.py   # UI only (docs must already be ingested)

Startup loading works like the UPYOG voice bot: the environment file and
the documents in data/sample_docs are loaded when the app starts, so the
UI itself stays just questions and answers.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.answer import answer
from src.chat_render import CSS, latency_badge_html, message_html, source_card_html
from src.config import RAGConfig
from src.ingest import get_client, ingest

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "sample_docs"


# ---------------------------------------------------------------- startup
def manifest_path(cfg: RAGConfig) -> Path:
    """Fingerprint file next to the vector DB folder."""
    return Path(cfg.qdrant_path).with_suffix(".manifest.json")


def compute_fingerprint(cfg: RAGConfig, docs_dir: Path) -> dict:
    """Hash of everything that affects the index: model, chunking, files."""
    files = {}
    for p in sorted(Path(docs_dir).rglob("*.csv")):
        if p.is_file():
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "files": files,
    }


def ensure_index(cfg: RAGConfig, model=None, docs_dir: Path = DOCS_DIR):
    """Make sure the vector DB holds the current docs."""
    fingerprint = compute_fingerprint(cfg, docs_dir)
    mpath = manifest_path(cfg)
    client = get_client(cfg)
    try:
        up_to_date, points = False, 0
        if mpath.exists() and client.collection_exists(cfg.collection_name):
            try:
                saved = json.loads(mpath.read_text())
                points = client.get_collection(cfg.collection_name).points_count
                up_to_date = saved == fingerprint and points > 0
            except Exception:
                up_to_date = False
        if up_to_date:
            return "up-to-date", points
    finally:
        client.close()  # local Qdrant locks the folder per client instance

    if model is None:
        print(f"  loading embedding model '{cfg.embedding_model}' "
              "(first run downloads it once)...")
        model = SentenceTransformer(cfg.embedding_model)
    client = get_client(cfg)
    try:
        if client.collection_exists(cfg.collection_name):
            client.delete_collection(cfg.collection_name)
    finally:
        client.close()
    n = ingest(str(docs_dir), cfg, model=model)
    mpath.write_text(json.dumps(fingerprint, indent=2))
    return "ingested", n


def resolve_paths(cfg: RAGConfig) -> RAGConfig:
    """Keep the vector DB next to this file, whatever folder you launch from."""
    if not Path(cfg.qdrant_path).is_absolute():
        cfg.qdrant_path = str(ROOT / cfg.qdrant_path)
    return cfg


def main() -> None:
    """Plain-python path: load env + docs, then start the UI."""
    cfg = resolve_paths(RAGConfig())
    print("RAG Chatbot starting...")
    env_file = ROOT / ".env"
    if env_file.exists():
        print("  env file: loaded (.env)")
    else:
        print("  env file: missing — copy .env.example to .env")
    if not cfg.groq_api_key:
        print("  warning: GROQ_API_KEY is not set — answers need a free key")
        print("             from console.groq.com in .env")
    print(f"  embedding model: {cfg.embedding_model}")
    status, n = ensure_index(cfg)  # loads the model only if a rebuild is needed
    if status == "up-to-date":
        print(f"  vector DB up to date: {n} chunks (model not reloaded)")
    else:
        print(f"  vector DB loaded: {n} chunks from data/sample_docs")
    print("  UI: http://localhost:8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.fileWatcherType",
            "none",
        ]
    )


# ---------------------------------------------------------------- UI
@st.cache_resource
def get_model(model_name: str):
    return SentenceTransformer(model_name)


def index_is_empty(cfg: RAGConfig) -> bool:
    try:
        client = get_client(cfg)
        if not client.collection_exists(cfg.collection_name):
            return True
        return client.get_collection(cfg.collection_name).points_count == 0
    except Exception:
        return True


def render_sources_and_meta(sources: list[dict], latency: dict | None = None):
    """Render latency breakdown badge and source confidence cards."""
    if latency:
        st.markdown(latency_badge_html(latency), unsafe_allow_html=True)
    if sources:
        with st.expander(f"📚 Sources & Retrieval Confidence ({len(sources)})", expanded=False):
            for i, s in enumerate(sources):
                st.markdown(
                    source_card_html(i + 1, s["source"], s["score"], s.get("text", "")),
                    unsafe_allow_html=True,
                )


def run_ui() -> None:
    """The chat UI with dynamic CSV upload, starter chips, latency badges & confidence bars."""
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">RAG Chatbot</h1>', unsafe_allow_html=True)
    st.caption("Ask questions grounded strictly in your CSV documents with source citations.")

    cfg = resolve_paths(RAGConfig())
    if not cfg.groq_api_key:
        try:
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                cfg.groq_api_key = str(st.secrets["GROQ_API_KEY"])
        except Exception:
            pass

    # ------------------ Sidebar: Controls & Dynamic Upload ------------------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📤 Upload New CSV")
        st.caption("Upload your own `.csv` file to instantly index it into the vector DB:")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            save_path = DOCS_DIR / uploaded_file.name
            if not save_path.exists():
                with st.spinner(f"Ingesting & embedding {uploaded_file.name}..."):
                    save_path.write_bytes(uploaded_file.getbuffer())
                    model = get_model(cfg.embedding_model)
                    ensure_index(cfg, model=model)
                    st.success(f"Indexed **{uploaded_file.name}** successfully!")
                    st.rerun()
            else:
                st.info(f"📄 **{uploaded_file.name}** is already indexed.")

        st.markdown("---")
        st.markdown("### 📊 Active Knowledge Base")
        csv_files = sorted(Path(DOCS_DIR).glob("*.csv"))
        for f in csv_files:
            col_name, col_del = st.columns([0.8, 0.2])
            with col_name:
                st.markdown(f"• `{f.name}`")
            with col_del:
                if st.button("❌", key=f"del_{f.name}", help=f"Delete {f.name}"):
                    try:
                        f.unlink()
                        model = get_model(cfg.embedding_model)
                        ensure_index(cfg, model=model)
                        st.success(f"Removed {f.name}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing: {e}")

    # ------------------ Message History ------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        st.markdown(message_html(m["role"], m["content"]), unsafe_allow_html=True)
        if m["role"] == "assistant":
            render_sources_and_meta(m.get("sources", []), m.get("latency"))

    if index_is_empty(cfg):
        with st.spinner("Initializing knowledge base from CSV documents (one-time setup)..."):
            model = get_model(cfg.embedding_model)
            ensure_index(cfg, model=model)
            st.rerun()


    # ------------------ Starter Chips for Recruiters ------------------
    if len(st.session_state.messages) == 0:
        st.markdown("<p style='color: #94a3b8; font-size: 0.88rem; margin-bottom: 6px;'>💡 Try asking:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        starter_selected = None
        with col1:
            if st.button("🤖 What is RAG?", use_container_width=True):
                starter_selected = "What is RAG?"
            if st.button("⚖️ Difference between List and Tuple", use_container_width=True):
                starter_selected = "What is the difference between a List and a Tuple?"
        with col2:
            if st.button("🔍 Explain Vector Space & Embeddings", use_container_width=True):
                starter_selected = "Explain Vector Space and Embeddings"
            if st.button("⚡ What is Context Window?", use_container_width=True):
                starter_selected = "What is a Context Window?"
    else:
        starter_selected = None

    # ------------------ User Input Handling ------------------
    prompt = st.chat_input("Ask a question about the CSV documents...")
    final_prompt = starter_selected if starter_selected else prompt

    if final_prompt:
        if not cfg.groq_api_key:
            st.warning(
                "No Groq API key found. Copy `.env.example` to `.env`, "
                "put your free key from console.groq.com in it, and restart."
            )
            st.stop()

        st.markdown(message_html("user", final_prompt), unsafe_allow_html=True)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        try:
            with st.spinner("Searching vectors & generating answer..."):
                gen, chunks, sources, rewritten, latency = answer(
                    final_prompt, cfg, history=history, stream=True,
                    model=get_model(cfg.embedding_model),
                )
            if rewritten != final_prompt:
                st.caption(f"🔍 Interpreted as: *{rewritten}*")

            placeholder = st.empty()
            full = ""
            for token in gen:
                full += token
                placeholder.markdown(
                    message_html("assistant", full), unsafe_allow_html=True
                )

            # Combine source scores and text for rich confidence cards
            rich_sources = [
                {"source": s["source"], "score": s["score"], "text": c["text"]}
                for s, c in zip(sources, chunks)
            ]
            render_sources_and_meta(rich_sources, latency)

            st.session_state.messages.append({"role": "user", "content": final_prompt})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full,
                    "sources": rich_sources,
                    "latency": latency,
                }
            )
        except RuntimeError as e:
            st.error(str(e))
            st.info("Run `python app.py` first — it loads the docs into the vector DB.")


# ---------------------------------------------------------------- dispatch
if st.runtime.exists():
    run_ui()
elif __name__ == "__main__":
    main()

