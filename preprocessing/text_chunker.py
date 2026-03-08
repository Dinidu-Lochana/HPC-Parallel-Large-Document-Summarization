"""
text_chunker.py
Phase 2 - Split extracted text into chunks that fit within the LLM context window.
"""

import os
import json
import re
from typing import List, Dict


def split_into_chunks(
    text: str,
    max_tokens: int = 3000,
    overlap_tokens: int = 100,
    chars_per_token: float = 4.0,
) -> List[str]:
    """
    Split text into overlapping chunks based on estimated token count.

    Args:
        text:             Full extracted text.
        max_tokens:       Max tokens per chunk (Gemini supports ~30k, keep 3000 for safety).
        overlap_tokens:   Overlap between chunks to preserve context.
        chars_per_token:  Rough estimate: 1 token ≈ 4 characters (English text).

    Returns:
        List of text chunks.
    """
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    # Prefer splitting at sentence or paragraph boundaries
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous chunk
            overlap_text = current_chunk[-overlap_chars:] if overlap_chars > 0 else ""
            current_chunk = overlap_text + (" " if overlap_text else "") + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(
    file_path: str,
    text: str,
    max_tokens: int = 3000,
    overlap_tokens: int = 100,
) -> List[Dict]:
    """
    Chunk a document's text and return structured chunk objects.

    Args:
        file_path:     Source file path (used for metadata).
        text:          Extracted text from the document.
        max_tokens:    Max tokens per chunk.
        overlap_tokens: Overlap between chunks.

    Returns:
        List of chunk dicts: { chunk_id, source_file, chunk_index, total_chunks, text }
    """
    raw_chunks = split_into_chunks(text, max_tokens, overlap_tokens)
    doc_name = os.path.basename(file_path)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{doc_name}_chunk_{i}",
            "source_file": file_path,
            "doc_name": doc_name,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "char_count": len(chunk_text),
            "estimated_tokens": len(chunk_text) // 4,
            "text": chunk_text,
        })

    print(f"[Chunker] '{doc_name}' → {len(chunks)} chunks "
          f"(max {max_tokens} tokens each, {overlap_tokens} overlap)")
    return chunks


def save_chunks(chunks: List[Dict], output_dir: str, doc_name: str) -> str:
    """Save chunks to a JSON file for later processing."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = doc_name.replace(" ", "_").replace(".", "_")
    output_path = os.path.join(output_dir, f"{safe_name}_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Chunks saved to: {output_path}")
    return output_path


def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunks from a saved JSON file."""
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    # Quick test with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines.
    Unlike the natural intelligence displayed by animals, AI involves machines mimicking cognitive functions.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning uses neural networks with many layers to analyze various factors of data.
    Natural language processing allows computers to understand human language.
    Computer vision enables machines to interpret and understand visual information.
    Robotics combines AI with mechanical engineering to create autonomous machines.
    """ * 50  # Repeat to simulate a larger document

    chunks = chunk_document("test_document.txt", sample_text, max_tokens=500)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"First chunk preview:\n{chunks[0]['text'][:200]}...")

    if "--save" in sys.argv:
        save_chunks(chunks, "chunks_output", "test_document.txt")