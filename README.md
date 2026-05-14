# Investor Presentation RAG

A small, reliable RAG system that ingests an investor presentation PDF and
answers analyst-style questions with page-level citations. Built as a 72-hour
intern evaluation assignment.

---

## Architecture

```
PDF
 │
 ▼
pymupdf4llm ──► vision fallback (llama-4-scout) ──► chunker ──► embedder ──► Qdrant
                (sparse pages only)                  KPI norm   BGE-small

Query
 │
 ▼
embedder (BGE-small + query prefix) ──► Qdrant search ──► LLM (llama-4-scout) ──► cited answer
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A Groq API key — free at https://console.groq.com

### Option A — Local (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Streamlit UI (separate terminal):
```bash
streamlit run frontend/app_ui.py
```

### Option B — Local with Docker Compose (one command)

```bash
cp .env.example .env        # set GROQ_API_KEY
docker compose up --build
```

- API: http://localhost:8000
- UI:  http://localhost:8501

### Option C — Production on Railway

1. Push this repo to GitHub.
2. Go to https://railway.app → New Project → Deploy from GitHub repo.
3. Add environment variable `GROQ_API_KEY` in the Railway dashboard.
4. Railway reads `railway.toml` and deploys automatically.
5. Copy the public URL (e.g. `https://your-app.railway.app`).
6. Set `API_BASE_URL=https://your-app.railway.app` in the Streamlit service
   (or run the UI locally pointing at the Railway backend).

> **Switching environments:** The Streamlit frontend reads `API_BASE_URL` from
> the environment. Default is `http://localhost:8000`. Set it to your Railway
> URL to point at production — no code changes needed.

> **No Docker needed for local.** Qdrant runs in-memory inside the FastAPI
> process. Set `QDRANT_PATH=./qdrant_data` in `.env` to persist the index
> across restarts.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check and configuration summary |
| `POST` | `/ingest` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question, receive a cited answer |

### GET /health

```json
{
  "status": "ok",
  "qdrant": "connected",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "llm_model": "meta-llama/llama-4-scout-17b-16e-instruct",
  "version": "1.0.0"
}
```

### POST /query — response shape

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

## Running Tests

```bash
pytest -v
```

136 tests covering chunking, KPI normalization, section tagging, LLM helpers,
BGE embedding prefix, vision extractor, and all three API endpoints.

---

## Engineering Decisions

### PDF Parser — `pymupdf4llm`

**Why:** Converts each PDF page to structured Markdown, preserving slide
hierarchy: headings, bold numbers, bullet lists, and tables as Markdown pipe
tables. Bold KPI values (e.g. `**1101**`) stay adjacent to their label text on
the same line, so the KPI normaliser can pair them reliably. pdfplumber
produced flat text runs like `"1101 Store count 251 City Presence"` where
numbers and labels were interleaved without structure.

**Where it fails:** Scanned PDFs with no embedded text layer still need OCR.
Complex infographic charts (bar/pie) with no text cannot be parsed.

---

### Vision Fallback — `llama-4-scout-17b-16e-instruct`

**Why:** Investor presentations render key data as bar charts, logo tiles, and
KPI infographics rather than selectable text. pymupdf4llm misses these.

**How:** Pages are considered sparse if either:
1. Fewer than 30 words remain after stripping pymupdf4llm placeholder lines.
2. A chart indicator keyword ("Store Count", "Revenue", "Q2 FY22") is present
   but no 3-digit number exists — meaning axis labels were extracted but bar
   values are image-rendered.

Sparse pages are rendered to PNG at 150 DPI via pdf2image and sent to the
Groq vision API. The structured description is prepended to any existing text.

**Limitation:** Requires a Groq API call per sparse page, adding ~2 s per page
to ingestion time. Complex charts where bars have no text labels may still be
missed.

---

### Chunking Strategy

**Approach:** KPI-aware, sentence-boundary chunking scoped per page.

- Each PDF page is treated as a logical unit first.
- A **KPI normalization pass** rewrites inline pairs like
  `"**1101** Store count 251 City Presence"` into explicit key-value sentences:
  `"Store count: 1101.  City Presence: 251."` before embedding.
- A **margin/growth labelling pass** tags `"EBITDA margin 28%"` as
  `"Margin: 28%"` and `"Revenue grew 12% YoY"` as `"YoY Growth: 12%"` so the
  LLM can distinguish them.
- **KPI blocks** — contiguous groups of lines where each line contains both a
  number and a label — are treated as atomic units that cannot be split across
  chunk boundaries.
- Pages exceeding 400 words are split at sentence/line boundaries with 80-word
  overlap. A single line longer than 400 words falls back to word-window
  splitting.

---

### Embedding Model — `BAAI/bge-small-en-v1.5`

**Why over all-MiniLM-L6-v2:**
- 512-token context window (vs 256) — fewer chunks are silently truncated.
- Trained specifically for retrieval; outperforms MiniLM on BEIR benchmarks.
- Same 384-dimensional output — no change to the Qdrant collection config.

**BGE query prefix:** BGE models are asymmetric. Queries must be prefixed with
`"Represent this sentence for searching relevant passages: "`. Document chunks
are encoded without any prefix. Omitting the prefix measurably degrades
retrieval quality.

---

### Vector Store — Qdrant (in-memory / local)

**Why:** `QdrantClient(path=":memory:")` runs entirely in-process — no Docker,
no server. Same Python API as Qdrant Cloud; switching requires only changing
the constructor argument.

**Persistence:** Set `QDRANT_PATH=./qdrant_data` in `.env` to persist the
index to disk across server restarts.

**Limitation:** In-memory mode loses data on restart. Single-process only.

---

### Prompting Strategy

`SYSTEM_PROMPT` in `app/llm.py` contains six numbered rules:

1. **Cite every claim** inline as `[Page X]`.
2. **Never reassign numbers** to a different entity than the one they are
   explicitly paired with in the source text.
3. **Distinguish margin % from YoY growth %** — they appear together on slides
   but mean different things.
4. **Report contradictions** rather than silently choosing one value.
5. **Add a LIMITATIONS section** when evidence is weak or absent.
6. **Never fabricate** numbers, names, or facts.

Temperature is 0.0 for maximum factual consistency.

---

### Citation Strategy

1. The LLM embeds `[Page X]` markers inline in its answer.
2. A regex pass extracts all mentioned page numbers.
3. Each page is matched to the corresponding retrieved chunk payload.
4. Structured `Citation` objects are built with `page`, `chunk_id`, and a
   200-character `excerpt`.
5. Any retrieved chunks whose pages were not explicitly cited are appended so
   the caller always knows which evidence was consulted.

---

## Known Limitations

1. **Image-rendered content:** Bar charts, logos, and infographics are handled
   by the vision fallback but require a Groq API call during ingestion. Complex
   charts where bars have no text labels may still be missed.
2. **Scanned PDFs:** Pages with no selectable text require OCR (pytesseract),
   which is not included in this version.
3. **Cross-page context:** Claims spanning two slides may be split across chunk
   boundaries; retrieval may only return one half.
4. **Single collection:** Qdrant is not namespaced per document. Re-ingesting a
   second PDF without `?force=true` will mix chunks from both documents.
5. **Indian financial jargon:** BGE-small is not fine-tuned on ₹/Cr/Lakh/SSSG
   terminology; synonym recall for these terms may be lower than for standard
   English financial vocabulary.

---

## Evaluation Report

### Test Document

Trent Limited — Investor Presentation Q2 FY26 (35 pages)

### Questions and Results

| # | Question | Type | Result | Pages cited |
|---|---|---|---|---|
| 1 | What was revenue in Q2 FY26? | Factual | PASS | 4, 5 |
| 2 | How many total stores? | Factual | PASS | 4 |
| 3 | Westside vs Zudio store count? | Segment | PARTIAL | 10, 16 |
| 4 | Management growth priorities? | Strategy | PASS | 7 |
| 5 | How has PAT margin changed? | Financial trend | PARTIAL | 6 |
| 6 | What headwinds in Q2 FY26? | Risk | PASS | 7 |
| 7 | Westside vs Zudio revenue split? | Evidence challenge | PASS | — |
| 8 | What new brands were launched? | Multi-hop | PARTIAL | 22, 23 |

### Results Summary

- **4 Pass, 3 Partial, 0 Fail, 1 Evidence challenge correctly refused**
- Partial results on Q3, Q5, Q8 are due to key data existing only inside bar
  chart images — the vision fallback extracts partial context but bar chart
  pixel values are not always reliably read.
- The system never hallucinated an incorrect number — it correctly said "not
  found" when evidence was missing.

### Root Causes Fixed During Development

1. **Flat text extraction causing label-value confusion** — fixed by switching
   from pdfplumber to pymupdf4llm, which preserves bold KPI values adjacent to
   their labels in structured Markdown.
2. **LLM misreading adjacent numbers** — fixed by the KPI normalization pass
   (`"1101 Store count 251 City Presence"` → `"Store count: 1101. City Presence: 251."`)
   and stricter system prompt rules.
3. **Wrong pages retrieved for narrative questions** — fixed by `section_title`
   metadata tagging and a +0.15 score boost for Management Commentary chunks.
4. **Margin vs growth % confusion** — fixed by explicit `Margin: X%` /
   `YoY Growth: X%` labelling in the chunker and prompt rule 3.

---

## What Would Change for 1,000 Companies

| Concern | Change |
|---------|--------|
| **Storage** | Qdrant Cloud with per-company namespacing (collection per company or `company_id` payload filter). |
| **Embedding throughput** | GPU-backed batch embedding service or OpenAI `text-embedding-3-small`. |
| **PDF parsing** | Async ingestion queue (Celery + Redis); OCR fallback already in place for scanned PDFs. |
| **Retrieval quality** | Second-stage cross-encoder re-ranker to improve precision. |
| **LLM cost** | Cache answers for identical questions per document; cheaper model for simple factual lookups. |
| **Observability** | Log retrieval scores, answer latency, and citation coverage per query. |
| **Multi-tenancy** | API key auth per company; row-level security on the vector store. |
