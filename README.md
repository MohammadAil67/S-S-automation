# SOP → Airtop Agent Prompt RAG

A Retrieval-Augmented Generation (RAG) pipeline that ingests Standard Operating
Procedure (SOP) documents and generates structured prompts for **Airtop browser
agents** — grounded entirely in your own documentation.

---

## How it works

```
SOP files (PDF / DOCX / TXT)
        │
        ▼
  Document Loader  (LangChain)
        │
        ▼
   Text Splitter   (recursive, 1000-char chunks)
        │
        ▼
   OpenAI Embeddings  (text-embedding-3-small)
        │
        ▼
   ChromaDB  ←── persisted on disk
        │
        ▼
   Similarity Search  (top-5 chunks for a given task)
        │
        ▼
   GPT-4o  +  Airtop prompt template
        │
        ▼
   Structured Airtop Agent Prompt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key
```

---

## Usage

### Run the built-in demo (no files needed)

```bash
python main.py demo
```

This writes a sample SOP to disk, ingests it, and generates an Airtop prompt
for an order-export task.

---

### Ingest your own SOP documents

```bash
# Ingest an entire folder (PDF, DOCX, TXT are all supported)
python main.py ingest ./sop_docs

# Or ingest a single file
python main.py ingest ./sops/login_procedure.pdf
```

You can call `ingest` multiple times — new documents are **added** to the
existing ChromaDB collection without re-processing old ones.

---

### Generate an Airtop agent prompt

```bash
python main.py generate "Log into the order portal and export all pending orders from the last 30 days"
```

Example output:

```
======================================================================
GENERATED AIRTOP AGENT PROMPT
======================================================================
Objective: Log into the Acme order management portal and export all
pending orders from the last 30 days as a CSV file to the shared drive.

Steps:
1. Ensure your VPN is active before navigating to any internal URL.
2. Navigate to https://orders.internal.acme.com and click "Sign In".
3. Enter your corporate email and AD password; complete MFA if prompted.
4. In the left sidebar, select Orders → Pending.
5. Set the date filter to "Last 30 days" using the calendar picker.
6. Click Export in the top-right corner and select CSV.
7. Save the file to //fileserver/ops/exports/.
8. Verify the row count in the downloaded file matches the UI count.

Rules & restrictions:
- Do not attempt to manually scrape data if the export fails; log a ticket instead.
- Do not store the exported file on a personal device.
- Delete any local copy within 24 hours of uploading to the shared drive.

Success condition: The CSV file is present at //fileserver/ops/exports/ and its
row count matches the "Pending" order count shown in the portal UI.
======================================================================
```

---

### Other commands

```bash
python main.py info     # Show how many chunks are stored
python main.py clear    # Wipe the ChromaDB collection
```

---

## Using as a Python library

```python
from rag_pipeline import SOPAgentRAG

rag = SOPAgentRAG()

# Ingest documents
rag.ingest("./sop_docs")

# Generate a prompt
prompt = rag.generate_prompt(
    "Navigate to the HR portal and submit a PTO request for next Monday."
)
print(prompt)

# Check collection size
print(rag.collection_info())

# Clear and start fresh
rag.clear_database()
```

---

## Project structure

```
rag-airtop/
├── rag_pipeline.py   # Core RAG logic (load, chunk, embed, retrieve, generate)
├── main.py           # CLI entry point
├── requirements.txt
├── .env.example      # Copy to .env and add your OpenAI key
└── chroma_db/        # Created automatically on first ingest
```

---

## Configuration

All tunable constants are at the top of `rag_pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | 5 | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

---

## Swapping the LLM

The LLM is instantiated in `build_rag_chain()` in `rag_pipeline.py`. To use a
different model, replace:

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
```

with any LangChain-compatible chat model, e.g. Anthropic Claude:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-4-5", temperature=0.2)
```
