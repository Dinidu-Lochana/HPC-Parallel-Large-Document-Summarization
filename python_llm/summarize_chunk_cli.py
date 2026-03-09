# summarize_chunk_cli.py
import google.generativeai as genai
from pypdf import PdfReader
import os, sys, time, re, random
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def _call_with_retry(prompt, label=""):
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, stream=False)
            return response.text
        except ResourceExhausted as e:
            if attempt == max_retries - 1:
                raise
            match = re.search(r'retry in (\d+(\.\d+)?)s', str(e), re.IGNORECASE)
            delay = (float(match.group(1)) if match else 45) + random.uniform(2, 8)
            print(f"[{label}] Rate limit. Waiting {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)

if __name__ == "__main__":
    # Usage: python summarize_chunk_cli.py <chunk_file> <file_name> <topic>
    chunk_file = sys.argv[1]
    file_name  = sys.argv[2] if len(sys.argv) > 2 else "Document"
    topic      = sys.argv[3] if len(sys.argv) > 3 else ""

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunk = f.read()

    prompt = f"""You are an expert summarizer.
File Name: {file_name}
Topic: {topic}
Summarize the following text clearly and concisely:
{chunk}"""

    result = _call_with_retry(prompt, label=f"chunk:{chunk_file}")
    print(result)   # stdout → captured by C program