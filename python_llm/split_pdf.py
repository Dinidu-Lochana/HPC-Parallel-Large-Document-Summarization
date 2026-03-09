# split_pdf.py
# Usage: python split_pdf.py <pdf_path> <chunk_size> <output_dir>
from pypdf import PdfReader
import sys, os

pdf_path   = sys.argv[1]
chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
output_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/chunks"

os.makedirs(output_dir, exist_ok=True)

reader = PdfReader(pdf_path)
text = "".join(page.extract_text() for page in reader.pages)

chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

for i, chunk in enumerate(chunks):
    path = os.path.join(output_dir, f"chunk_{i}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(chunk)

print(len(chunks))   # prints chunk count → read by C program