"""
preprocess.py
Phase 2 - Main preprocessing pipeline.
Usage:
    python preprocess.py <file_or_folder> [--ocr] [--max-tokens 3000] [--output-dir chunks/]
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict

# Add parent directory to path so we can import from preprocessing/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_extractor import extract_text
from text_chunker import chunk_document, save_chunks


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def get_documents(input_path: str) -> List[str]:
    """Return list of document file paths from a file or folder."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        docs = []
        for fname in os.listdir(input_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                docs.append(os.path.join(input_path, fname))
        return sorted(docs)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def preprocess_document(
    file_path: str,
    output_dir: str,
    max_tokens: int = 3000,
    overlap_tokens: int = 100,
    force_ocr: bool = False,
) -> Dict:
    """
    Full preprocessing pipeline for a single document:
      1. Extract text
      2. Chunk text
      3. Save chunks to JSON

    Returns a summary dict with stats.
    """
    start = time.time()
    doc_name = os.path.basename(file_path)

    print(f"\n{'='*60}")
    print(f"Processing: {doc_name}")
    print(f"{'='*60}")

    # Step 1: Extract
    try:
        text = extract_text(file_path, force_ocr=force_ocr)
        if not text.strip():
            print(f"[WARNING] No text extracted from: {doc_name}")
            return {"file": doc_name, "status": "empty", "chunks": 0}
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {doc_name}: {e}")
        return {"file": doc_name, "status": "error", "error": str(e), "chunks": 0}

    # Step 2: Chunk
    chunks = chunk_document(file_path, text, max_tokens, overlap_tokens)

    # Step 3: Save
    chunks_file = save_chunks(chunks, output_dir, doc_name)

    elapsed = time.time() - start
    summary = {
        "file": doc_name,
        "status": "success",
        "total_chars": len(text),
        "estimated_tokens": len(text) // 4,
        "chunks": len(chunks),
        "chunks_file": chunks_file,
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"[Done] {doc_name}: {len(text):,} chars → {len(chunks)} chunks in {elapsed:.2f}s")
    return summary


def run_pipeline(
    input_path: str,
    output_dir: str = "chunks",
    max_tokens: int = 3000,
    overlap_tokens: int = 100,
    force_ocr: bool = False,
) -> List[Dict]:
    """Run the full preprocessing pipeline for all documents."""
    documents = get_documents(input_path)

    if not documents:
        print(f"[WARNING] No supported documents found in: {input_path}")
        return []

    print(f"\n[Pipeline] Found {len(documents)} document(s) to preprocess")
    print(f"[Pipeline] Output directory: {output_dir}")
    print(f"[Pipeline] Max tokens per chunk: {max_tokens}")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for doc_path in documents:
        result = preprocess_document(
            doc_path, output_dir, max_tokens, overlap_tokens, force_ocr
        )
        results.append(result)

    # Save pipeline summary
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print final report
    success = [r for r in results if r["status"] == "success"]
    total_chunks = sum(r["chunks"] for r in success)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Documents processed : {len(success)}/{len(results)}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Summary saved to    : {summary_path}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Preprocess documents into chunks for LLM summarization"
    )
    parser.add_argument("input", help="Path to a document file or folder of documents")
    parser.add_argument("--ocr", action="store_true", help="Force OCR for PDFs (for scanned docs)")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max tokens per chunk (default: 3000)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap tokens between chunks (default: 100)")
    parser.add_argument("--output-dir", default="chunks", help="Directory to save chunks (default: chunks/)")

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        force_ocr=args.ocr,
    )


if __name__ == "__main__":
    main()