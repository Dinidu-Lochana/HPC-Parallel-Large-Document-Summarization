"""
run_pipeline.py
End-to-end runner: Phase 2 (preprocessing) → Phase 3 (serial summarization)

Usage:
    python run_pipeline.py <document_or_folder> [options]

Examples:
    python run_pipeline.py docs/my_report.pdf
    python run_pipeline.py docs/ --ocr --max-tokens 2000
"""

import sys
import os
import argparse

# Add subfolders to path so modules can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "preprocessing"))
sys.path.insert(0, os.path.join(BASE_DIR, "serial"))

from preprocess import run_pipeline as run_preprocessing # type: ignore
from serial_summarizer import run_serial # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="HPC Parallel Summarization - End-to-end pipeline (Phase 2 + Phase 3)"
    )
    parser.add_argument("input", help="Document file or folder to summarize")
    parser.add_argument("--ocr", action="store_true", help="Force OCR for scanned PDFs")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap tokens between chunks")
    parser.add_argument("--chunks-dir", default="chunks", help="Where to store chunks")
    parser.add_argument("--output-dir", default="summaries/serial", help="Where to store summaries")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  HPC PARALLEL SUMMARIZATION PROJECT")
    print("  Phase 2: Preprocessing + Phase 3: Serial Summarization")
    print("="*60)

    # ── Phase 2: Preprocessing ─────────────────────────────────────────────
    print("\n>> PHASE 2: Document Preprocessing")
    preprocessing_results = run_preprocessing(
        input_path=args.input,
        output_dir=args.chunks_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        force_ocr=args.ocr,
    )

    if not preprocessing_results:
        print("[ERROR] No documents were preprocessed. Exiting.")
        sys.exit(1)

    # ── Phase 3: Serial Summarization ─────────────────────────────────────
    print("\n>> PHASE 3: Serial Summarization")
    serial_results = run_serial(
        input_path=args.chunks_dir,
        output_dir=args.output_dir,
    )

    print("\n>> PIPELINE FINISHED")
    print(f"   Summaries saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()