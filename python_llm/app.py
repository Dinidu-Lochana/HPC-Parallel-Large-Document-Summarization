"""
Streamlit frontend for the HPC Parallel Document Summarizer.

Calls the FastAPI backend (backend/main.py) which in turn runs the
compiled MPI / OpenMP / Hybrid C binaries.

Start backend first:
    uvicorn backend.main:app --reload --port 8000

Then run this frontend:
    streamlit run python_llm/app.py
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HPC Document Summarizer",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ HPC Parallel Document Summarizer")
st.caption("MPI · OpenMP · Hybrid MPI+OpenMP  —  powered by Gemini 2.5 Flash")

# ── sidebar: backend status ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Backend Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success("FastAPI backend is running")

        methods = requests.get(f"{BACKEND_URL}/methods", timeout=3).json()
        st.subheader("Compiled Binaries")
        for name, ready in methods.items():
            icon = "✅" if ready else "❌"
            st.write(f"{icon} `{name}`")
        if not any(methods.values()):
            st.warning("No binaries compiled yet.\nCompile with:\n```\nmpicc -O2 -o bin/mpi_summarizer mpi/mpi_summarizer.c\ngcc -O2 -fopenmp -o bin/openmp_summarizer openmp/openmp_summarizer.c\nmpicc -O2 -fopenmp -o bin/hybrid_summarizer hybrid/hybrid_summarizer.c\n```")
    except Exception:
        st.error(f"Cannot reach backend at {BACKEND_URL}\nStart it with:\n```\nuvicorn backend.main:app --reload --port 8000\n```")

# ── main form ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document",
        type=["pdf", "txt"],
    )
    topic = st.text_input("Document topic", placeholder="e.g. Machine Learning")

    method = st.selectbox(
        "Parallel method",
        options=["mpi", "openmp", "hybrid"],
        format_func=lambda m: {
            "mpi":    "MPI  (distributed processes)",
            "openmp": "OpenMP  (shared-memory threads)",
            "hybrid": "Hybrid MPI + OpenMP",
        }[m],
    )

    # show process / thread sliders based on chosen method
    processes = 4
    threads   = 4

    if method in ("mpi", "hybrid"):
        processes = st.slider("MPI processes", min_value=2, max_value=8, value=4)
    if method in ("openmp", "hybrid"):
        threads = st.slider("OpenMP threads per process", min_value=1, max_value=8, value=4)

    if method == "hybrid":
        st.info(f"Total parallelism: **{processes} × {threads} = {processes * threads}** parallel units")

    run = st.button("▶  Run Summarizer", type="primary", use_container_width=True)

# ── run & display ─────────────────────────────────────────────────────────────
with col_right:
    st.subheader("Results")

    if run:
        if not uploaded_file:
            st.warning("Please upload a file first.")
        elif not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Running {method.upper()} summarizer…"):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/summarize",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        uploaded_file.type or "application/octet-stream")},
                        data={
                            "topic":     topic,
                            "method":    method,
                            "processes": processes,
                            "threads":   threads,
                        },
                        timeout=360,
                    )

                    if response.status_code != 200:
                        st.error(f"Backend error ({response.status_code}):\n{response.json().get('detail', response.text)}")
                    else:
                        data    = response.json()
                        metrics = data.get("metrics", {})
                        summary = data.get("summary", "")

                        # ── metrics ───────────────────────────────────────────
                        st.subheader("📊 Performance Metrics")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("⏱ Execution Time",
                                  f"{metrics.get('execution_time', 0):.2f} s")
                        m2.metric("🚀 Speedup",
                                  f"{metrics.get('speedup', 0):.2f}×")
                        m3.metric("⚙ Efficiency",
                                  f"{metrics.get('efficiency', 0):.1f}%")
                        m4.metric("🖥 CPU Utilization",
                                  f"{metrics.get('resource_utilization', 0):.1f}%")

                        st.divider()

                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.markdown(f"**Method:** `{metrics.get('method','').upper()}`")
                            st.markdown(f"**MPI Processes:** `{metrics.get('processes')}`")
                            st.markdown(f"**OMP Threads:** `{metrics.get('threads')}`")
                            st.markdown(f"**Chunks processed:** `{metrics.get('chunks')}`")
                        with detail_col2:
                            st.markdown(f"**Sequential estimate:** `{metrics.get('sequential_estimate', 0):.2f} s`")
                            st.markdown(f"**Scalability:** `{metrics.get('scalability', 0):.1f}%` of ideal")
                            st.markdown(f"**CPU cores available:** `{int(metrics.get('cpu_cores') or 0)}`")

                        # ── summary ───────────────────────────────────────────
                        st.divider()
                        st.subheader("📄 Final Summary")
                        st.write(summary if summary else "_No summary generated._")

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {BACKEND_URL}.\n"
                             "Make sure it is running:\n"
                             "```\nuvicorn backend.main:app --reload --port 8000\n```")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The document may be too large or the "
                             "Gemini API is slow. Try a smaller file.")
