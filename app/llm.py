"""
LLM answer generation using Groq (llama-4-scout-17b-16e-instruct).

Prompting strategy:
  The SYSTEM_PROMPT template injects retrieved evidence chunks via {context}
  and the user question via {question}. Six strict grounding rules prevent
  number reassignment, margin/growth conflation, and fabrication.
  Temperature is 0.0 for maximum factual consistency.

Citation strategy:
  The LLM is instructed to embed [Page X] markers inline. After generation,
  a regex pass extracts all mentioned page numbers and matches them to the
  retrieved chunk payloads to build structured Citation objects with 200-char
  evidence excerpts.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

from groq import Groq

from app.config import settings
from app.models import Citation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Narrative question detection — used to boost commentary chunks in retrieval
# ---------------------------------------------------------------------------

_NARRATIVE_KEYWORDS = frozenset([
    "headwind", "tailwind", "risk", "challenge", "sentiment",
    "context", "why", "reason", "macro", "environment", "outlook",
    "management", "commentary", "operating", "consumer", "demand",
    "inflationary", "discretionary", "footfall", "concern",
])


def _is_narrative_question(question: str) -> bool:
    """Return True if the question asks for management commentary or context."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _NARRATIVE_KEYWORDS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst assistant reviewing an investor presentation.
Answer questions using ONLY the evidence chunks provided. Do not use
any outside knowledge or make assumptions beyond what is stated.

Rules:
1. Cite every claim inline as [Page X] immediately after the statement.
2. Never reassign a number or metric to a different entity than the one
   it is explicitly paired with in the source text.
3. Distinguish carefully between margin percentages and YoY growth
   percentages — they appear together on slides but mean different things.
4. If two sources contradict each other, report both values and note
   the discrepancy rather than choosing one silently.
5. If the evidence does not contain enough information to answer
   confidently, say so clearly and add a LIMITATIONS section explaining
   what is missing or ambiguous.
6. Never fabricate numbers, names, dates, or facts not present in the
   provided evidence.

Evidence:
{context}

Question: {question}"""

_EVIDENCE_TEMPLATE = """--- Evidence Chunk {idx} ---
Page: {page}
Section: {section}
Text:
{text}
"""

# ---------------------------------------------------------------------------
# Retrieval re-ranking for narrative questions
# ---------------------------------------------------------------------------


def _boost_narrative_chunks(
    chunks_with_scores: List[Tuple[float, dict]],
) -> List[Tuple[float, dict]]:
    """
    Boost Management Commentary and Results Context chunks to the top of the
    retrieval list for narrative questions.

    Embedding similarity sometimes favours visually dense pages (campaign
    images, store photos) over text-heavy commentary pages when the query
    contains words like "headwind" or "risk". A +0.15 score boost corrects
    this without requiring a separate re-ranker model.
    """
    COMMENTARY_TAGS = {"management commentary", "results context"}

    boosted: List[Tuple[float, dict]] = []
    rest: List[Tuple[float, dict]] = []

    for score, payload in chunks_with_scores:
        tag = payload.get("section_title", "").lower()
        if tag in COMMENTARY_TAGS:
            boosted.append((min(score + 0.15, 1.0), payload))
        else:
            rest.append((score, payload))

    boosted.sort(key=lambda x: x[0], reverse=True)
    rest.sort(key=lambda x: x[0], reverse=True)
    return boosted + rest


# ---------------------------------------------------------------------------
# Evidence block builder
# ---------------------------------------------------------------------------


def _build_evidence_block(chunks_with_scores: List[Tuple[float, dict]]) -> str:
    """Format retrieved chunks into a numbered evidence block for the prompt."""
    parts = []
    for idx, (_, payload) in enumerate(chunks_with_scores, start=1):
        parts.append(
            _EVIDENCE_TEMPLATE.format(
                idx=idx,
                page=payload.get("page_number", "?"),
                section=payload.get("section_title", ""),
                text=payload.get("text", ""),
            )
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _extract_limitations(answer: str) -> Tuple[str, List[str]]:
    """
    Split the LIMITATIONS section out of the raw answer text.

    Returns
    -------
    tuple
        (clean_answer, limitations_list)
    """
    pattern = re.compile(r"\bLIMITATIONS?:\s*", re.IGNORECASE)
    match = pattern.search(answer)
    if not match:
        return answer.strip(), []

    main = answer[: match.start()].strip()
    lim_block = answer[match.end():].strip()
    items = [
        line.lstrip("-•* ").strip()
        for line in lim_block.splitlines()
        if line.strip()
    ]
    return main, [i for i in items if i]


def _build_citations(
    answer: str,
    chunks_with_scores: List[Tuple[float, dict]],
) -> List[Citation]:
    """
    Parse [Page X] markers from the answer and match them to retrieved chunks.

    Pages explicitly cited in the answer are listed first; any remaining
    retrieved pages are appended so the caller always knows which evidence
    was consulted.
    """
    mentioned_pages = set(
        int(p) for p in re.findall(r"\[Page\s+(\d+)\]", answer, re.IGNORECASE)
    )

    page_to_payload: dict[int, dict] = {}
    for _, payload in chunks_with_scores:
        pg = payload.get("page_number")
        if pg not in page_to_payload:
            page_to_payload[pg] = payload

    citations: List[Citation] = []
    seen_pages: set[int] = set()

    for pg in sorted(mentioned_pages):
        if pg in page_to_payload and pg not in seen_pages:
            payload = page_to_payload[pg]
            citations.append(
                Citation(
                    page=pg,
                    chunk_id=payload.get("chunk_id", ""),
                    excerpt=payload.get("text", "")[:200].replace("\n", " "),
                )
            )
            seen_pages.add(pg)

    for _, payload in chunks_with_scores:
        pg = payload.get("page_number")
        if pg not in seen_pages:
            citations.append(
                Citation(
                    page=pg,
                    chunk_id=payload.get("chunk_id", ""),
                    excerpt=payload.get("text", "")[:200].replace("\n", " "),
                )
            )
            seen_pages.add(pg)

    return citations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    chunks_with_scores: List[Tuple[float, dict]],
) -> Tuple[str, List[Citation], List[str]]:
    """
    Generate a grounded answer from retrieved evidence chunks.

    Parameters
    ----------
    question : str
        The analyst-style question to answer.
    chunks_with_scores : list of (score, payload) tuples
        Retrieved chunks from Qdrant, ordered by descending similarity.

    Returns
    -------
    tuple
        (answer, citations, limitations)
    """
    if not chunks_with_scores:
        return (
            "I could not find relevant evidence in the document to answer this question.",
            [],
            ["No relevant chunks were retrieved from the vector store."],
        )

    if _is_narrative_question(question):
        chunks_with_scores = _boost_narrative_chunks(chunks_with_scores)

    filled_prompt = SYSTEM_PROMPT.format(
        context=_build_evidence_block(chunks_with_scores),
        question=question,
    )

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[{"role": "user", "content": filled_prompt}],
    )

    raw_answer = response.choices[0].message.content or ""
    clean_answer, limitations = _extract_limitations(raw_answer)
    citations = _build_citations(raw_answer, chunks_with_scores)

    if not re.search(r"\[Page\s+\d+\]", raw_answer, re.IGNORECASE):
        limitations.append(
            "The model did not produce explicit page citations; "
            "citations below are inferred from retrieved chunks."
        )

    return clean_answer, citations, limitations
