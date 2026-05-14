"""
Streamlit frontend for the Investor Presentation RAG system.

Run with:
    streamlit run frontend/app_ui.py

The FastAPI backend must be running on http://localhost:8000.
Start it with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Investor Presentation RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Investor Presentation RAG")
st.caption(
    "Ingest an investor PDF and ask analyst-style questions with page-level citations."
)

# ---------------------------------------------------------------------------
# Sidebar — Ingest
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📥 Ingest Document")

    uploaded_file = st.file_uploader("Upload investor presentation PDF", type=["pdf"])
    company_name = st.text_input(
        "Company name (optional)", placeholder="e.g. Trent Limited"
    )
    force = st.checkbox(
        "Force re-ingest (clears existing data)",
        value=False,
        help="Drops the Qdrant collection before ingesting. Use when re-uploading the same PDF.",
    )

    if st.button("Ingest PDF", type="primary", disabled=uploaded_file is None):
        with st.spinner("Parsing, chunking, and embedding…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    params={"force": str(force).lower()},
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={"company_name": company_name},
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['message']}")
                else:
                    st.error(
                        f"Ingest failed ({resp.status_code}): "
                        f"{resp.json().get('detail', resp.text)}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the API at localhost:8000. "
                    "Run: uvicorn app.main:app --reload --port 8000"
                )

    st.divider()

    st.subheader("🔌 System Status")
    if st.button("Check health"):
        try:
            h = requests.get(f"{API_BASE}/health", timeout=5).json()
            st.json(h)
        except Exception as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# Main — Query
# ---------------------------------------------------------------------------
st.header("💬 Ask a Question")

top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=15, value=5)
question = st.text_area(
    "Your question",
    placeholder="What was revenue in Q2 FY26?",
    height=80,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence and generating answer…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": question.strip(), "top_k": top_k},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()

                st.subheader("Answer")
                st.write(data["answer"])

                if data.get("limitations"):
                    with st.expander("⚠️ Limitations / Caveats"):
                        for lim in data["limitations"]:
                            st.warning(lim)

                st.subheader("📄 Citations")
                for cit in data["citations"]:
                    with st.expander(
                        f"Page {cit['page']}  —  chunk `{cit['chunk_id']}`"
                    ):
                        st.markdown(f"**Excerpt:**\n> {cit['excerpt']}")

                with st.expander("🔍 Retrieval info"):
                    st.json(data["retrieval"])

            else:
                st.error(
                    f"Query failed ({resp.status_code}): "
                    f"{resp.json().get('detail', resp.text)}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the API at localhost:8000. "
                "Run: uvicorn app.main:app --reload --port 8000"
            )
