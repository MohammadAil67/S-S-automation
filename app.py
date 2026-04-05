"""
app.py — Streamlit UI for the SOP → Airtop Agent Prompt RAG system.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SOP → Airtop Prompt Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — only things Streamlit can't do natively ──────────────────
st.markdown("""
<style>
.prompt-box {
    background: var(--secondary-background-color);
    border-left: 4px solid #4f6ef7;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    font-family: 'Courier New', monospace;
    font-size: 0.87rem;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-color);
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "rag": None,
        "generated_prompt": "",
        "retrieved_chunks": [],
        "ingested_files": [],
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def get_rag():
    return st.session_state.get("rag")

def set_rag(rag) -> None:
    st.session_state["rag"] = rag


# ── Lazy import ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_class():
    from rag_pipeline import SOPAgentRAG
    return SOPAgentRAG


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🤖 SOP → Airtop")
    st.caption("RAG-powered agent prompt generator")
    st.divider()

    # ── API key ──────────────────────────────────────────────────────────
    st.subheader("🔑 Google API Key")
    api_key_input = st.text_input(
        "Paste your key",
        type="password",
        placeholder="AIza...",
        help="Stored only in this browser session. Get a free key at aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.session_state["api_key_set"] = True
        st.success("Key saved for this session.", icon="✅")
    elif os.environ.get("GOOGLE_API_KEY"):
        st.session_state["api_key_set"] = True
        st.info("Key loaded from environment.", icon="ℹ️")
    else:
        st.caption("🔗 [Get a free key](https://aistudio.google.com/app/apikey)")

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    chunk_size = st.slider("Chunk size (chars)", 400, 2000, 1000, step=100,
                           help="Larger chunks = more context per retrieval hit.")
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, step=50,
                              help="Overlap keeps context across chunk boundaries.")
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5,
                      help="How many SOP chunks to feed into the LLM.")
    llm_model = st.selectbox(
        "LLM model",
        ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )

    st.divider()

    # ── Collection status ─────────────────────────────────────────────────
    st.subheader("📦 Collection")
    rag = get_rag()
    if rag:
        info = rag.collection_info()
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", info["documents"])
        col_b.metric("Files", len(st.session_state["ingested_files"]))
        if st.button("🗑️ Clear collection", use_container_width=True):
            rag.clear_database()
            set_rag(None)
            st.session_state.update({
                "ingested_files": [],
                "generated_prompt": "",
                "retrieved_chunks": [],
            })
            st.rerun()
    else:
        st.caption("No documents ingested yet.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════

tab_ingest, tab_generate, tab_about = st.tabs(
    ["📂 Ingest Documents", "✨ Generate Prompt", "ℹ️ About"]
)


# ─── TAB 1: INGEST ────────────────────────────────────────────────────────

with tab_ingest:
    st.header("Upload SOP Documents")
    st.write("Supported formats: **PDF, DOCX, TXT**")

    if not st.session_state["api_key_set"]:
        st.warning("Add your Google API key in the sidebar before ingesting.", icon="⚠️")

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not st.session_state["api_key_set"],
    )

    col_ingest, col_demo = st.columns([2, 1], gap="medium")

    with col_ingest:
        ingest_btn = st.button(
            "⚡ Ingest uploaded files",
            use_container_width=True,
            disabled=not uploaded_files or not st.session_state["api_key_set"],
            type="primary",
        )

    with col_demo:
        demo_btn = st.button(
            "🎲 Load demo SOP",
            use_container_width=True,
            disabled=not st.session_state["api_key_set"],
            help="Loads a built-in sample SOP so you can try without uploading anything.",
        )

    # ── Ingest uploaded files ────────────────────────────────────────────
    if ingest_btn and uploaded_files:
        SOPAgentRAG = load_rag_class()
        import rag_pipeline as _rp
        _rp.CHUNK_SIZE = chunk_size
        _rp.CHUNK_OVERLAP = chunk_overlap
        _rp.TOP_K_RESULTS = top_k

        with st.status("Ingesting documents…", expanded=True) as status:
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_paths = []
                for f in uploaded_files:
                    dest = Path(tmpdir) / f.name
                    dest.write_bytes(f.read())
                    saved_paths.append(str(dest))
                    st.write(f"• Saved **{f.name}**")

                rag = get_rag() or SOPAgentRAG()
                for p in saved_paths:
                    st.write(f"• Ingesting **{Path(p).name}**…")
                    rag.ingest(p)
                    if Path(p).name not in st.session_state["ingested_files"]:
                        st.session_state["ingested_files"].append(Path(p).name)

                set_rag(rag)
                status.update(label="Ingestion complete!", state="complete")

        st.success(
            f"✅ {len(uploaded_files)} file(s) ingested. "
            f"Collection now has **{rag.collection_info()['documents']}** chunks.",
        )

    # ── Demo SOP ─────────────────────────────────────────────────────────
    if demo_btn:
        from main import DEMO_SOP_TEXT
        SOPAgentRAG = load_rag_class()
        import rag_pipeline as _rp
        _rp.CHUNK_SIZE = chunk_size
        _rp.CHUNK_OVERLAP = chunk_overlap
        _rp.TOP_K_RESULTS = top_k

        with st.status("Loading demo SOP…", expanded=True) as status:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(DEMO_SOP_TEXT)
                tmp_path = tmp.name

            rag = get_rag() or SOPAgentRAG()
            rag.ingest(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if "demo_sop.txt (built-in)" not in st.session_state["ingested_files"]:
                st.session_state["ingested_files"].append("demo_sop.txt (built-in)")

            set_rag(rag)
            status.update(label="Demo SOP loaded!", state="complete")

        st.success(
            f"✅ Demo SOP ingested. "
            f"Collection now has **{rag.collection_info()['documents']}** chunks.",
        )

    # ── Ingested file list ────────────────────────────────────────────────
    if st.session_state["ingested_files"]:
        st.subheader("Ingested files")
        for name in st.session_state["ingested_files"]:
            st.markdown(f"- 📄 `{name}`")


# ─── TAB 2: GENERATE ──────────────────────────────────────────────────────

with tab_generate:
    st.header("Generate Airtop Agent Prompt")

    rag = get_rag()
    if not rag or rag.collection_info()["documents"] == 0:
        st.info(
            "Ingest at least one SOP document first — go to the **Ingest Documents** tab.",
            icon="👈",
        )
    else:
        st.write(
            "Describe what you want the Airtop agent to do. "
            "The RAG system will find the most relevant SOP sections and generate "
            "a precise, step-by-step agent prompt grounded in your documentation."
        )

        task = st.text_area(
            "Task description",
            placeholder=(
                "e.g. Log into the order management portal and export all "
                "pending orders from the last 30 days as a CSV file."
            ),
            height=120,
        )

        generate_btn = st.button(
            "🚀 Generate prompt",
            disabled=not task.strip(),
            type="primary",
        )

        if generate_btn and task.strip():
            import rag_pipeline as _rp
            _rp.TOP_K_RESULTS = top_k

            def patched_build(vs):
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough

                retriever = vs.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.2)
                prompt = ChatPromptTemplate.from_template(_rp.AIRTOP_PROMPT_TEMPLATE)

                def format_docs(docs):
                    parts = []
                    for i, doc in enumerate(docs, 1):
                        src = doc.metadata.get("source_file", "unknown")
                        parts.append(f"[Chunk {i} — source: {src}]\n{doc.page_content}")
                    return "\n\n".join(parts)

                chain = (
                    {"context": retriever | format_docs, "task": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                return chain, retriever

            with st.spinner("Retrieving SOP chunks and generating prompt…"):
                chain, retriever = patched_build(rag._vector_store)
                rag._chain = chain
                rag._retriever = retriever

                chunks = rag._retriever.invoke(task)
                st.session_state["retrieved_chunks"] = chunks

                result = rag._chain.invoke(task)
                st.session_state["generated_prompt"] = result

        # ── Output ───────────────────────────────────────────────────────
        if st.session_state["generated_prompt"]:
            st.subheader("Generated Airtop Agent Prompt")

            st.markdown(
                f'<div class="prompt-box">{st.session_state["generated_prompt"]}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            st.download_button(
                label="⬇️ Download as .txt",
                data=st.session_state["generated_prompt"],
                file_name="airtop_agent_prompt.txt",
                mime="text/plain",
            )

            # ── Retrieved chunks ──────────────────────────────────────────
            if st.session_state["retrieved_chunks"]:
                with st.expander(
                    f"🔍 View {len(st.session_state['retrieved_chunks'])} retrieved SOP chunks",
                    expanded=False,
                ):
                    for i, doc in enumerate(st.session_state["retrieved_chunks"], 1):
                        src = doc.metadata.get("source_file", "unknown")
                        page = doc.metadata.get("page", "—")
                        with st.container(border=True):
                            st.caption(f"Chunk {i} · {src} · page {page}")
                            st.text(doc.page_content)


# ─── TAB 3: ABOUT ─────────────────────────────────────────────────────────

with tab_about:
    st.header("About this tool")
    st.markdown("""
This app uses **Retrieval-Augmented Generation (RAG)** to turn your internal
Standard Operating Procedure documents into ready-to-use
[Airtop](https://airtop.ai) browser agent prompts.

### Pipeline

```
SOP files (PDF / DOCX / TXT)
        │
        ▼
  LangChain document loaders
        │
        ▼
  Recursive text splitter  (configurable chunk size + overlap)
        │
        ▼
  sentence-transformers all-MiniLM-L6-v2  →  ChromaDB (persisted to disk)
        │
        ▼
  Similarity search  (top-k chunks for the given task)
        │
        ▼
  Gemini 2.0 Flash  +  Airtop prompt template
        │
        ▼
  Structured, step-by-step Airtop agent prompt
```

### Key design principles

- **Grounded output** — the LLM is instructed only to use information found in
  your SOPs. If context is thin it says so explicitly.
- **Persistent index** — ChromaDB stores embeddings on disk in `./chroma_db/`
  so you only pay embedding costs once per document.
- **Configurable** — chunk size, overlap, k, and LLM model are all adjustable
  from the sidebar without touching any code.

### Tech stack

| Layer | Library |
|---|---|
| Document loading | `langchain-community` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API key) |
| Vector store | ChromaDB |
| LLM | Gemini 2.0 Flash / 1.5 Pro / 1.5 Flash |
| Orchestration | LangChain LCEL |
| UI | Streamlit |
""")