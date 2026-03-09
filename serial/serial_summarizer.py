"""
serial_summarizer.py
Phase 3 - Serial (single-threaded) summarization using Gemini API.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict

from google import genai # type: ignore
from google.genai import types # type: ignore
from dotenv import load_dotenv # type: ignore

# Load .env from same directory or one level up (try both)
_env_same = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_env_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_same if os.path.exists(_env_same) else _env_parent)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Set it in your .env file:\n  GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-2.0-flash"

# ─── Prompt Templates ─────────────────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = """You are a precise document summarizer.
Summarize the following text chunk clearly and concisely.
Capture all key points, facts, and important details.
Write in plain English. Be thorough but not repetitive.

Text chunk:
{chunk_text}

Summary:"""

FINAL_COMBINE_PROMPT = """You are an expert document summarizer.
Below are summaries of individual sections of a larger document.
Combine them into one coherent, well-structured final summary.
Remove redundancy. Preserve all important information.
Use clear paragraphs. Do NOT use bullet points.

Section summaries:
{section_summaries}

Final combined summary:"""


# ─── Core Functions ────────────────────────────────────────────────────────────

def summarize_chunk(chunk: dict, retry_count: int = 5) -> str:
    """Call Gemini API to summarize a single chunk with detailed error logging."""
    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk["text"])

    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            text = response.text
            if not text or not text.strip():
                print(f"  [WARNING] Empty response for chunk {chunk['chunk_id']} on attempt {attempt+1}")
                time.sleep(5)
                continue
            return text.strip()

        except Exception as e:
            error_msg = str(e)
            print(f"  [ERROR] Attempt {attempt+1}/{retry_count} failed for chunk {chunk['chunk_id']}")
            print(f"  [ERROR] Reason: {error_msg}")

            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                wait = 30 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s before retry...")
            elif "500" in error_msg or "503" in error_msg:
                wait = 10 * (attempt + 1)
                print(f"  [SERVER ERROR] Waiting {wait}s before retry...")
            else:
                wait = 2 ** attempt
                print(f"  [BACKOFF] Waiting {wait}s before retry...")

            time.sleep(wait)

    error_text = f"[ERROR: Could not summarize chunk {chunk['chunk_id']}]"
    print(f"  [FAILED] All {retry_count} attempts exhausted for chunk {chunk['chunk_id']}")
    return error_text


def combine_summaries(chunk_summaries: List[str], retry_count: int = 3) -> str:
    """Combine all chunk summaries into one final summary."""
    numbered = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    prompt = FINAL_COMBINE_PROMPT.format(section_summaries=numbered)

    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Retry {attempt+1}/{retry_count}] Combine error: {e}. Waiting {wait}s...")
            time.sleep(wait)

    print("  [WARNING] Combine failed — concatenating chunk summaries as fallback.")
    return "\n\n".join(chunk_summaries)


def summarize_document(chunks_file: str, output_dir: str) -> dict:
    """Serially summarize all chunks from one document."""
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return {"status": "empty", "file": chunks_file}

    doc_name = chunks[0]["doc_name"]
    total_chunks = len(chunks)

    print(f"\n{'='*60}")
    print(f"Summarizing: {doc_name}  ({total_chunks} chunks)")
    print(f"{'='*60}")
    print(f"  [INFO] Using model: {MODEL_NAME}")
    print(f"  [INFO] API key loaded: {'Yes' if GEMINI_API_KEY else 'NO - THIS IS THE PROBLEM'}")
    print(f"  [INFO] API key prefix: {GEMINI_API_KEY[:8]}..." if GEMINI_API_KEY else "")

    chunk_summaries = []
    chunk_times = []
    start_total = time.time()

    # ── Step 1: Summarize each chunk serially ─────────────────────────────
    for i, chunk in enumerate(chunks):
        t0 = time.time()
        print(f"\n  Chunk {i+1}/{total_chunks} (est. {chunk['estimated_tokens']} tokens)...", flush=True)

        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)

        elapsed = time.time() - t0
        chunk_times.append(elapsed)
        print(f"  Chunk {i+1} done in ({elapsed:.2f}s)")

        # Rate limit: wait 5s between calls
        if i < total_chunks - 1:
            print(f"  [Rate limit] Waiting 5s before next chunk...", flush=True)
            time.sleep(5)

    # ── Step 2: Combine all chunk summaries ───────────────────────────────
    print(f"\n  Combining {total_chunks} summaries...", end=" ", flush=True)
    t0 = time.time()

    if total_chunks == 1:
        final_summary = chunk_summaries[0]
    else:
        final_summary = combine_summaries(chunk_summaries)

    combine_time = time.time() - t0
    print(f"done ({combine_time:.2f}s)")

    total_time = time.time() - start_total

    # ── Step 3: Save output ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    safe_name = doc_name.replace(" ", "_").replace(".", "_")
    out_path = os.path.join(output_dir, f"{safe_name}_summary.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"DOCUMENT: {doc_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(final_summary)
        f.write(f"\n\n{'='*60}\n")
        f.write(f"[Chunks: {total_chunks} | Total time: {total_time:.2f}s | Model: {MODEL_NAME}]\n")

    result = {
        "doc_name": doc_name,
        "status": "success",
        "total_chunks": total_chunks,
        "total_time_seconds": round(total_time, 2),
        "avg_chunk_time_seconds": round(sum(chunk_times) / len(chunk_times), 2),
        "combine_time_seconds": round(combine_time, 2),
        "summary_file": out_path,
        "final_summary": final_summary,
        "chunk_summaries": chunk_summaries,
    }

    print(f"\n[Done] {doc_name}: summarized in {total_time:.2f}s → {out_path}")
    return result


def run_serial(
    input_path: str,
    output_dir: str = "summaries/serial",
) -> List[dict]:
    """Run serial summarization on one chunks file or a folder of chunks files."""

    if os.path.isfile(input_path) and input_path.endswith(".json"):
        chunk_files = [input_path]
    elif os.path.isdir(input_path):
        chunk_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith("_chunks.json")
        ]
        chunk_files.sort()
    else:
        raise FileNotFoundError(f"Input not found or not a chunks JSON: {input_path}")

    if not chunk_files:
        print("[WARNING] No chunk files found.")
        return []

    print(f"\n[Serial] Processing {len(chunk_files)} document(s) serially...")

    pipeline_start = time.time()
    results = []

    for cf in chunk_files:
        result = summarize_document(cf, output_dir)
        results.append(result)
        time.sleep(2)

    pipeline_time = time.time() - pipeline_start

    metrics = {
        "mode": "serial",
        "documents": len(results),
        "total_pipeline_time_seconds": round(pipeline_time, 2),
        "results": results,
    }

    metrics_path = os.path.join(output_dir, "serial_metrics.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SERIAL SUMMARIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Documents  : {len(results)}")
    print(f"  Total time : {pipeline_time:.2f}s")
    print(f"  Metrics    : {metrics_path}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Serially summarize pre-chunked documents using Gemini"
    )
    parser.add_argument("input", help="Path to a *_chunks.json file OR a folder containing chunk files")
    parser.add_argument("--output-dir", default="summaries/serial", help="Where to save summaries")
    args = parser.parse_args()

    run_serial(args.input, args.output_dir)


if __name__ == "__main__":
    main()