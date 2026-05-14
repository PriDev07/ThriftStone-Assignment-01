# Evaluation Report — Investor Presentation RAG

> **Test document:** Trent Limited Investor Presentation Q2 FY26 (35 pages)
> **Ingestion method:** POST /ingest via Streamlit UI
> **Query method:** POST /query via Streamlit UI
> **Embedding model:** BAAI/bge-small-en-v1.5
> **LLM:** meta-llama/llama-4-scout-17b-16e-instruct (Groq)
> **Vector store:** Qdrant (in-memory)
> **Date:** May 2026

---

## Evaluation Methodology

Each question is assessed on three dimensions:

| Dimension | Description |
|-----------|-------------|
| **Support level** | `Supported` / `Partially Supported` / `Unsupported` |
| **Retrieved pages** | Pages returned by the retrieval step |
| **Failure note** | Any observed gap, hallucination, or retrieval miss |

---

## Question Set

### Q1 — Factual Lookup
**Question:** What was revenue in the latest reported period?

| Field | Value |
|-------|-------|
| **Answer** | Trent Limited's standalone revenue from operations in Q2 FY26 was ₹4,724 Cr, representing 17% YoY growth. On a consolidated basis, revenue from operations was ₹4,818 Cr, up 16% YoY. |
| **Retrieved pages** | pp. 4, 5, 6 |
| **Support level** | Supported |
| **Failure note** | None. Both standalone and consolidated figures were retrieved and correctly attributed. Page 4 (At a Glance) and Page 5 (Q2 FY26 Highlights) both contain the figure as headline text, which the retriever correctly prioritised. |

---

### Q2 — Business Segment
**Question:** Which business segment contributed most to revenue growth?

| Field | Value |
|-------|-------|
| **Answer** | The fashion segment, specifically Zudio, was the primary growth driver. Zudio expanded from 577 stores in Q2 FY25 to 806 stores in Q2 FY26, opening 44 new stores in the quarter across 244 cities including 2 in UAE. The Food and Grocery segment (Star) saw revenue decline slightly from ₹898 Cr in Q2 FY25 to ₹879 Cr in Q2 FY26 due to store upgrades. Exact segment-level revenue contribution percentages are not disclosed in the presentation. |
| **Retrieved pages** | pp. 7, 16, 26 |
| **Support level** | Partially Supported |
| **Failure note** | Store count data for Zudio (806) exists inside a bar chart image on Page 16 and was not reliably extracted by text parsing. The answer is directionally correct based on store expansion data but brand-level revenue figures are not disclosed in the presentation, preventing a precise segment contribution calculation. |

---

### Q3 — Strategy
**Question:** What are management's stated growth priorities for the next 12–24 months?

| Field | Value |
|-------|-------|
| **Answer** | Management's stated growth priorities are: (1) Drive material reach and share of revenues across key markets. (2) Upgrade the quality of the store portfolio and increase density of presence across existing markets. (3) Selectively consolidate underperforming stores to improve customer experience. (4) Expand into newer tier 2 and tier 3 cities and emerging catchments near metro areas. (5) Grow online revenues — Westside online grew 56% YoY and contributed over 6% of Westside revenues in Q2 FY26. (6) Continue investing in technology and automation, including pan-network RFID, to drive operating leverage and contain manpower cost growth. (7) Maintain a Company-Owned and Company-Operated model, with franchising only selectively for specific properties. |
| **Retrieved pages** | pp. 7, 8 |
| **Support level** | Supported |
| **Failure note** | None. Pages 7 and 8 are pure text management commentary slides and were retrieved and read correctly. All cited priorities appear verbatim or near-verbatim in the source. |

---

### Q4 — Risk / Challenge
**Question:** What risks or headwinds does the company mention in the presentation?

| Field | Value |
|-------|-------|
| **Answer** | The following risks and headwinds are explicitly mentioned: (1) Muted consumer sentiment in Q2 FY26. (2) Unseasonal rains creating demand headwinds in the quarter. (3) GST regime transition — customers initially prioritised bigger ticket purchases with higher GST reduction benefits, depressing small-ticket discretionary category demand. (4) Like-for-like growth for the fashion portfolio was in low single digits. (5) Newer tier 2/3 stores have a different maturity profile and revenue trajectory that may not be comparable to the existing portfolio, implying execution risk. (6) Newer stores have higher depreciation relative to revenues, compressing EBITDA margins until stores mature. |
| **Retrieved pages** | pp. 7, 8 |
| **Support level** | Supported |
| **Failure note** | None. Early in development this question was failing — the retriever was returning campaign image pages (pp. 17–19) instead of the Results Context pages (pp. 7–8). Fixed by adding section_title metadata tagging during ingestion so management commentary pages are correctly prioritised for risk and strategy queries. |

---

### Q5 — Financial Trend
**Question:** How has the operating margin changed across the periods shown in the presentation?

| Field | Value |
|-------|-------|
| **Answer** | Based on the trends slide (Page 6), standalone operating EBIT margin across Q2 periods: Q2 FY22: 13%, Q2 FY23: 7%, Q2 FY24: 10%, Q2 FY25: 11%, Q2 FY26: 10%. Standalone PAT margin across the same periods: Q2 FY22: 13%, Q2 FY23: 10%, Q2 FY24: 10%, Q2 FY25: 11%, Q2 FY26: 10%. The trend shows an initial compression in FY23 followed by recovery and stabilisation at approximately 10–11% for both EBIT and PAT margins over the last three periods. |
| **Retrieved pages** | pp. 5, 6 |
| **Support level** | Partially Supported |
| **Failure note** | The trend direction and approximate values are correct. However, the margin percentages on Page 6 are rendered as text overlaid on bar chart images. The system retrieved Q2 FY22 EBIT margin as 10% in one run rather than the correct 13%, indicating the chart text is not always cleanly extracted. The Q2 FY26 figures on Page 5 (standalone EBIT margin 9% growth, EBITDA margin 16%) were sometimes confused with the historical margin percentages on Page 6, as both appear on adjacent retrieved chunks. |

---

### Q6 — Evidence Challenge (intentionally hard)
**Question:** What is the company's projected free cash flow for the next fiscal year?

> This question is designed to test the system's ability to refuse or qualify when evidence
> is absent. A good system should say the PDF does not contain this information rather than
> hallucinating a number.

| Field | Value |
|-------|-------|
| **Answer** | The presentation does not contain any forward-looking free cash flow projections for the next fiscal year. The document is a Q2 FY26 performance highlights presentation covering historical results only. No guidance, forecast, or cash flow projection figures appear anywhere in the document. |
| **Retrieved pages** | pp. 2, 4, 5 |
| **Support level** | Unsupported (expected) |
| **Failure note** | None. The system correctly refused to answer and did not hallucinate a projection figure. The disclaimer on Page 2 (forward-looking statements warning) was retrieved, which appropriately contextualises why forward projections are absent. This is the intended behaviour enforced by Rule 5 and Rule 6 of the system prompt. |

---

### Q7 — Additional: Business Segment Revenue Split
**Question:** What is the exact revenue breakdown between Westside and Zudio for Q2 FY26?

| Field | Value |
|-------|-------|
| **Answer** | The presentation does not disclose brand-level revenue separately. Only consolidated revenue (₹4,818 Cr) and standalone revenue (₹4,724 Cr) are provided at the company level. A direct Westside vs Zudio revenue comparison cannot be made from this document. Westside online revenues grew 56% YoY and contributed over 6% of Westside revenues, and the Food and Grocery segment (Star) reported ₹879 Cr in Q2 FY26, but no equivalent figure is given for Westside or Zudio individually. |
| **Retrieved pages** | pp. 4, 5, 7, 26 |
| **Support level** | Unsupported (expected) |
| **Failure note** | None. Correctly refused. The Star segment revenue (₹879 Cr) was retrieved from Page 26 and correctly identified as a different segment, not conflated with the question about Westside/Zudio. |

---

### Q8 — Additional: Multi-hop
**Question:** What new brands did Trent launch and in which cities did they open their first stores?

| Field | Value |
|-------|-------|
| **Answer** | The system did not retrieve information about new brand launches. |
| **Retrieved pages** | pp. 4, 6, 7, 26 |
| **Support level** | Unsupported |
| **Failure note** | Incorrect. The correct answer is Burnt Toast, a youth-focused fashion brand launched in 2025 with initial stores in Bangalore, Thane, Pune, and Surat (Pages 22–24). The retrieval failed because the brand name "Burnt Toast" appears exclusively as a logo image on Page 22, making it invisible to the text embedding pipeline. Without the brand name present as extractable text, the semantic similarity between the query "new brands launched" and the chunk content was insufficient to surface Pages 22–24 in the top results. This is the only outright retrieval failure in this evaluation. |

---

## Summary Table

| # | Question type | Support level | Notes |
|---|---------------|---------------|-------|
| 1 | Factual lookup | Supported | Correct figure, correct page |
| 2 | Business segment | Partially Supported | Direction correct, brand revenue not disclosed |
| 3 | Strategy | Supported | All 7 priorities correctly extracted |
| 4 | Risk / challenge | Supported | All headwinds correctly identified |
| 5 | Financial trend | Partially Supported | Trend direction correct, FY22 value misread |
| 6 | Evidence challenge | Unsupported (expected) | Correctly refused, no hallucination |
| 7 | Revenue split | Unsupported (expected) | Correctly refused |
| 8 | New brand launch | Unsupported (incorrect) | Logo image prevented retrieval |

---

## Observed Failure Modes

1. **Image-rendered chart data** — Bar chart values on Pages 6, 10, and 16 are rendered
   as pixels inside chart graphics rather than selectable text. The text extraction layer
   (pymupdf4llm) correctly marks these as images but cannot recover the numeric values.
   The vision fallback (llama-4-scout-17b-16e-instruct) partially mitigates this but
   bar height pixel reading is not fully reliable for densely labelled charts.
   Affected questions: Q2 (store counts), Q5 (margin values).

2. **Logo-only brand identification** — Brand names that appear exclusively as logo images
   (Burnt Toast on Page 22, Westside/Zudio headers on Pages 10/16) are invisible to the
   embedding pipeline. Without the brand name present as text, semantic retrieval cannot
   surface the relevant pages regardless of query phrasing.
   Affected questions: Q2, Q8.

3. **Margin vs growth rate confusion** — Pages 5 and 6 both contain percentage figures
   but with different meanings: Page 5 shows YoY growth rates (17%, 16%, 9%, 6%) while
   Page 6 shows absolute margin percentages (13%, 7%, 10%, 11%, 10%). When both pages
   are retrieved together, the LLM occasionally conflates the two sets of percentages.
   Affected questions: Q5.

4. **Cross-page context for trend questions** — Questions asking "how has X changed"
   require synthesising data across multiple retrieved chunks. If top_k is set too low,
   not all relevant period pages are included and the trend answer is incomplete.
   Mitigated by using top_k=10 for trend-type questions.

---

## Overall Assessment

| Metric | Value |
|--------|-------|
| Questions fully supported | 2 |
| Questions correctly refused | 2 |
| Questions partially supported | 2 |
| Questions failed (wrong answer or missed) | 1 |
| Questions with retrieval miss | 1 |
| Hallucinations observed | 0 |

The system performs reliably on text-based slides, which represent approximately
60% of a typical investor presentation. All failures are confine