# Investor Presentation RAG

A small, reliable RAG system that ingests an investor presentation PDF and
answers analyst-style questions with page-level citations.

Built as a 72-hour intern evaluation assignment.

---

## Table of Contents

- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Stack](#stack)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [Engineering Decisions](#engineering-decisions)
- [Known Limitations](#known-limitations)
- [Evaluation Report](#evaluation-report)

## Screenshots

### Upload and ingest flow

![Upload screen](docs/screenshots/Screenshot%202026-05-14%20at%208.28.49%E2%80%AFPM.png)
![Ingested confirmation](docs/screenshots/Screenshot%202026-05-14%20at%208.29.13%E2%80%AFPM.png)

### Q&A with citations

![Answer with citations](docs/screenshots/Screenshot%202026-05-14%20at%208.29.36%E2%80%AFPM.png)
![Citation excerpts](docs/screenshots/Screenshot%202026-05-14%20at%208.29.44%E2%80%AFPM.png)

## Architecture

```
PDF
 │
 ▼
pymupdf4llm ──► vision fallback ──► chunker ──► embedder ──► Qdrant
                (llama-4-scout)     KPI norm    MiniLM-L6   (in-memory)
                sparse pages only   400 words   384-dim

Query
 │
 ▼
embedder (MiniLM-L6) ──► Qdrant cosine search ──► LLM (llama-3.3-70b) ──► cited answer
```

---

## Stack

| Component | Choice |
|---|---|
| API | FastAPI |
| PDF parser | pymupdf4llm |
| Vision fallback | llama-4-scout-17b (Groq) — for image-heavy slides |
| Embeddings | all-MiniLM-L6-v2 (local, no API key) |
| Vector store | Qdrant in-memory (no Docker required) |
| LLM | llama-3.3-70b-versatile (Groq) |
| Frontend | Streamlit |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ts_assist_submission.git
cd ts_assist_submission
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=gsk_...your_key_here...
```

Get a free key at https://console.groq.com

All other values have sensible defaults and do not need to be changed.

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Start the UI (separate terminal)

```bash
streamlit run frontend/app_ui.py
```

Open http://localhost:8501 in your browser.

---

## Usage

### Ingest a PDF

In the Streamlit UI: upload your PDF in the sidebar and click **Ingest PDF**.

Or via cURL:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@presentation.pdf" \
  -F "company_name=Trent Limited"
```

To re-ingest the same PDF (clears old data first):

```bash
curl -X POST "http://localhost:8000/ingest?force=true" \
  -F "file=@presentation.pdf"
```

### Ask a question

In the Streamlit UI: type your question and click **Ask**.

Or via cURL:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was revenue in Q2 FY26?", "top_k": 5}'
```

Interactive API docs: http://localhost:8000/docs

---

## API Reference

### GET /health

```json
{
  "status": "ok",
  "qdrant": "connected",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "llama-3.3-70b-versatile",
  "version": "1.0.0"
}
```

### POST /ingest

| Field | Type | Description |
|---|---|---|
| `file` | PDF (form) | The investor presentation PDF |
| `company_name` | string (form, optional) | Tag stored in chunk metadata |
| `force` | bool (query, default `false`) | Drop collection before ingesting |

**Response:**
```json
{
  "status": "success",
  "pages_processed": 35,
  "chunks_created": 142,
  "vision_pages": 6,
  "message": "Ingested presentation.pdf: 35 pages → 142 chunks (6 pages used vision fallback)."
}
```

### POST /query

**Request:**
```json
{
  "question": "What was revenue in Q2 FY26?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Revenue from operations was ₹4,724 Cr in Q2 FY26 [Page 5].",
  "citations": [
    {
      "page": 5,
      "chunk_id": "3f1a2b4c...",
      "excerpt": "Revenue from operations: ₹4,724 Cr..."
    }
  ],
  "retrieval": { "top_k": 5, "chunks_consulted": 5 },
  "limitations": []
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key. |
| `QDRANT_PATH` | `:memory:` | Leave blank for in-memory. Set `./qdrant_data` to persist across restarts. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name. |
| `CHUNK_SIZE` | `400` | Approximate words per chunk. |
| `CHUNK_OVERLAP` | `80` | Word overlap between adjacent chunks. |
| `DEFAULT_TOP_K` | `5` | Default chunks to retrieve. |

> **Note on Qdrant persistence:** By default, the vector store is in-memory and
> resets when the server restarts. Set `QDRANT_PATH=./qdrant_data` in `.env` to
> persist the index to disk — you won't need to re-ingest after a restart.

---

## Running Tests

```bash
pytest -v
```

140 tests covering chunking, KPI normalization, section tagging, LLM helpers,
embedding interface, vision extractor, and all three API endpoints.

---

## Engineering Decisions

### PDF Parser — `pymupdf4llm`

Converts each PDF page to structured Markdown, preserving headings, bold
numbers, bullet lists, and tables. Bold KPI values (e.g. `**1101**`) stay
adjacent to their label text, so the KPI normaliser can pair them reliably.

pdfplumber (the previous parser) produced flat text runs like
`"1101 Store count 251 City Presence"` where numbers and labels were
interleaved, causing the LLM to misassign values.

**Where it fails:** Scanned PDFs with no embedded text layer. Complex
infographic charts with no text.

---

### Vision Fallback — `llama-4-scout-17b-16e-instruct` (Groq)

Investor presentations render key data as bar charts and KPI infographics
rather than selectable text. pymupdf4llm misses these.

A page triggers the vision fallback if either:
1. Fewer than 30 words remain after stripping placeholder lines.
2. A chart indicator keyword (`"Store Count"`, `"Revenue"`, `"Q2 FY22"`, etc.)
   is present but no 3-digit number exists.

Sparse pages are rendered to PNG at 150 DPI via pdf2image and sent to the
Groq vision API. The structured description is prepended to any existing text.

**Note:** pdf2image is optional. If not installed, the vision fallback is
silently skipped and only pymupdf4llm text is used.

---

### Chunking Strategy

- **KPI normalization:** `"**1101** Store count 251 City Presence"` →
  `"Store count: 1101.  City Presence: 251."` before embedding.
- **Margin/growth labelling:** `"EBITDA margin 28%"` → `"Margin: 28%"`;
  `"Revenue grew 12% YoY"` → `"YoY Growth: 12%"`.
- **KPI blocks** (contiguous lines each containing a number + label) are
  atomic — never split across chunk boundaries.
- Pages exceeding 400 words are split at sentence boundaries with 80-word overlap.

---

### Embedding Model — `all-MiniLM-L6-v2`

- 384-dimensional vectors, runs fully locally, no API key needed.
- Symmetric model — queries and documents encoded identically, no prefix needed.
- 256-token context window; mitigated by 400-word chunks split with overlap.

---

### Vector Store — Qdrant (in-memory)

`QdrantClient(path=":memory:")` runs entirely in-process — no Docker, no
server. Set `QDRANT_PATH=./qdrant_data` in `.env` to persist to disk.

---

### Prompting Strategy

`SYSTEM_PROMPT` in `app/llm.py` has six rules:

1. Cite every claim as `[Page X]`.
2. Never reassign numbers to a different entity.
3. Distinguish margin % from YoY growth %.
4. Report contradictions rather than silently choosing one value.
5. Add a LIMITATIONS section when evidence is weak.
6. Never fabricate numbers or facts.

Temperature 0.0. Evidence and question injected via `{context}` / `{question}`.

---

### Citation Strategy

1. LLM embeds `[Page X]` markers inline.
2. Regex extracts all mentioned page numbers.
3. Each page matched to the retrieved chunk payload.
4. `Citation` objects built with `page`, `chunk_id`, and 200-char `excerpt`.
5. Uncited retrieved pages appended so the caller sees all consulted evidence.

---

## Known Limitations

1. **Image-rendered content:** Bar charts need pdf2image installed. Complex
   charts where bars have no text labels may still be missed.
2. **Scanned PDFs:** No OCR in this version — pages with no selectable text
   will be empty.
3. **In-memory store:** Qdrant resets on server restart unless
   `QDRANT_PATH=./qdrant_data` is set.
4. **Single collection:** Re-ingesting a second PDF without `?force=true`
   mixes chunks from both documents.
5. **Embedding truncation:** 256-token window on MiniLM; chunks over ~200
   words are silently truncated at the embedding stage.

---

## Evaluation Report

### Test Document

Trent Limited — Investor Presentation Q2 FY26 (35 pages)

### Results

| # | Question | Type | Result |
|---|---|---|---|
| 1 | What was revenue in Q2 FY26? | Factual | PASS |
| 2 | How many total stores? | Factual | PASS |
| 3 | Westside vs Zudio store count? | Segment | PARTIAL |
| 4 | Management growth priorities? | Strategy | PASS |
| 5 | How has PAT margin changed? | Financial trend | PARTIAL |
| 6 | What headwinds in Q2 FY26? | Risk | PASS |
| 7 | Westside vs Zudio revenue split? | Evidence challenge | PASS (correctly refused) |
| 8 | What new brands were launched? | Multi-hop | PARTIAL |

**4 Pass, 3 Partial, 0 Fail.** Partial results are due to key data existing
only inside bar chart images. The system never hallucinated an incorrect number.

### Root Causes Fixed

1. Flat text extraction → switched to pymupdf4llm structured Markdown.
2. LLM misreading adjacent numbers → KPI normalization pass + stricter prompt.
3. Wrong pages for narrative questions → section_title tagging + score boost.
4. Margin vs growth % confusion → explicit labelling in chunker + prompt rule 3.
