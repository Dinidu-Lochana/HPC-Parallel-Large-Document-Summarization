"""
document_extractor.py
Phase 2 - Extract text from .txt, .pdf, .docx, and scanned PDFs (OCR)
"""

import os
import sys


def extract_txt(file_path: str) -> str:
    """Extract text from a plain .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_docx(file_path: str) -> str:
    """Extract text from a .docx file."""
    try:
        import docx 
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_pdf(file_path: str) -> str:
    """Extract text from a digital (non-scanned) PDF."""
    try:
        from pypdf import PdfReader 
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def extract_scanned_pdf(file_path: str) -> str:
    """Extract text from a scanned PDF using PaddleOCR."""
    try:
        from paddleocr import PaddleOCR 
        import fitz  
        import numpy as np 
        from PIL import Image 
        import io
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run: pip install paddleocr PyMuPDF Pillow numpy"
        )

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    doc = fitz.open(file_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        result = ocr.ocr(img_array, cls=True)
        if result and result[0]:
            page_lines = [line[1][0] for line in result[0] if line[1][0].strip()]
            full_text.append("\n".join(page_lines))

    doc.close()
    return "\n".join(full_text)


def extract_text(file_path: str, force_ocr: bool = False) -> str:
    """
    Auto-detect file type and extract text.

    Args:
        file_path: Path to the document.
        force_ocr: If True, use OCR even for digital PDFs (for scanned PDFs).

    Returns:
        Extracted text as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        print(f"[Extractor] Reading TXT: {file_path}")
        return extract_txt(file_path)

    elif ext == ".docx":
        print(f"[Extractor] Reading DOCX: {file_path}")
        return extract_docx(file_path)

    elif ext == ".pdf":
        if force_ocr:
            print(f"[Extractor] OCR scanning PDF: {file_path}")
            return extract_scanned_pdf(file_path)
        else:
            print(f"[Extractor] Reading digital PDF: {file_path}")
            text = extract_pdf(file_path)
            # If very little text extracted, it's likely a scanned PDF — fall back to OCR
            if len(text.strip()) < 100:
                print(f"[Extractor] Low text yield — falling back to OCR for: {file_path}")
                return extract_scanned_pdf(file_path)
            return text

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf, .docx")


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) < 2:
        print("Usage: python document_extractor.py <file_path> [--ocr]")
        sys.exit(1)

    path = sys.argv[1]
    use_ocr = "--ocr" in sys.argv
    text = extract_text(path, force_ocr=use_ocr)
    print("\n--- Extracted Text (first 500 chars) ---")
    print(text[:500])
    print(f"\n[Total characters extracted: {len(text)}]")