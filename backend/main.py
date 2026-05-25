"""
FastAPI backend for HPC Parallel Document Summarizer.

Bridges the Streamlit frontend to the compiled C binaries
(MPI / OpenMP / Hybrid) and returns summaries + metrics as JSON.

Run from PROJECT ROOT:
    uvicorn backend.main:app --reload --port 8000
"""

import io
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="HPC Document Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# one summarisation at a time (C binaries write to fixed output file names)
_lock = threading.Lock()

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_text(data: bytes, filename: str) -> str:
    """Return plain text from a PDF or TXT upload."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return data.decode("utf-8", errors="replace")


def parse_metrics(stdout: str, method: str, processes: int, threads: int) -> dict:
    """Extract numeric metrics from C program stdout using regex."""
    def grab(pattern):
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else None

    chunks_m = re.search(r"Chunks Processed\s*:\s*(\d+)", stdout)

    return {
        "method":               method,
        "processes":            processes,
        "threads":              threads,
        "chunks":               int(chunks_m.group(1)) if chunks_m else None,
        "execution_time":       grab(r"Execution Time\s*:\s*([\d.]+)"),
        "sequential_estimate":  grab(r"Sequential Estimate\s*:\s*([\d.]+)"),
        "speedup":              grab(r"Speedup\s*:\s*([\d.]+)"),
        "efficiency":           grab(r"Efficiency\s*:\s*([\d.]+)"),
        "scalability":          grab(r"Scalability\s*:\s*([\d.]+)"),
        "cpu_cores":            grab(r"CPU Cores Available\s*:\s*(\d+)"),
        "resource_utilization": grab(r"Resource Utilization\s*:\s*([\d.]+)"),
    }


def build_command(method: str, doc_path: str, topic: str,
                  processes: int, threads: int) -> list:
    """Return the shell command list for the chosen method."""
    bin_dir = PROJECT_ROOT / "bin"

    if method == "mpi":
        binary = bin_dir / "mpi_summarizer"
        if not binary.exists():
            raise HTTPException(
                status_code=500,
                detail="bin/mpi_summarizer not found — compile with: "
                       "mpicc -O2 -o bin/mpi_summarizer mpi/mpi_summarizer.c"
            )
        return ["mpirun", "-np", str(processes), str(binary), doc_path, topic]

    elif method == "openmp":
        binary = bin_dir / "openmp_summarizer"
        if not binary.exists():
            raise HTTPException(
                status_code=500,
                detail="bin/openmp_summarizer not found — compile with: "
                       "gcc -O2 -fopenmp -o bin/openmp_summarizer openmp/openmp_summarizer.c"
            )
        return [str(binary), doc_path, topic, str(threads)]

    elif method == "hybrid":
        binary = bin_dir / "hybrid_summarizer"
        if not binary.exists():
            raise HTTPException(
                status_code=500,
                detail="bin/hybrid_summarizer not found — compile with: "
                       "mpicc -O2 -fopenmp -o bin/hybrid_summarizer hybrid/hybrid_summarizer.c"
            )
        return ["mpirun", "-np", str(processes), str(binary), doc_path, topic, str(threads)]

    elif method == "serial":
        binary = bin_dir / "serial_summarizer"
        if not binary.exists():
            raise HTTPException(
                status_code=500,
                detail="bin/serial_summarizer not found — compile with: "
                       "gcc -O2 -o bin/serial_summarizer serial/serial_summarizer.c"
            )
        return [str(binary), doc_path, topic]

    raise HTTPException(status_code=400, detail=f"Unknown method: {method}")


OUTPUT_FILES = {
    "serial": "serial_output.txt",
    "mpi":    "mpi_output.txt",
    "openmp": "omp_output.txt",
    "hybrid": "hybrid_output.txt",
}

# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/methods")
def list_methods():
    """Return available methods and whether their binaries are compiled."""
    bin_dir = PROJECT_ROOT / "bin"
    return {
        "serial": (bin_dir / "serial_summarizer").exists(),
        "mpi":    (bin_dir / "mpi_summarizer").exists(),
        "openmp": (bin_dir / "openmp_summarizer").exists(),
        "hybrid": (bin_dir / "hybrid_summarizer").exists(),
    }


@app.post("/summarize")
async def summarize(
    file:      UploadFile = File(...),
    topic:     str        = Form("General"),
    method:    str        = Form("mpi"),     # mpi | openmp | hybrid
    processes: int        = Form(4),
    threads:   int        = Form(4),
):
    """
    Upload a PDF or TXT document, choose a parallel method, get back
    the final summary and all performance metrics.
    """
    # ── 1. read + validate upload ────────────────────────────────────────────
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = extract_text(raw, file.filename)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    # ── 2. write text to a temp file in project root ─────────────────────────
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
        dir=str(PROJECT_ROOT), encoding="utf-8"
    )
    tmp.write(text)
    tmp.close()

    try:
        cmd = build_command(method, tmp.name, topic, processes, threads)

        # ── 3. run the C binary (serialised — one at a time) ─────────────────
        with _lock:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=300,        # 5-minute hard timeout
            )

            stdout   = result.stdout
            stderr   = result.stderr
            out_path = PROJECT_ROOT / OUTPUT_FILES[method]

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Binary exited with code {result.returncode}.\n"
                           f"stderr: {stderr[:500]}"
                )

            summary = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Summarization timed out (5 min).")
    finally:
        os.unlink(tmp.name)

    metrics = parse_metrics(stdout, method, processes, threads)

    return {
        "summary": summary,
        "metrics": metrics,
        "stdout":  stdout,        # raw C program output (useful for debugging)
    }
