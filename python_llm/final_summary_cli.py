# final_summary_cli.py
# Usage: python final_summary_cli.py <summaries_dir> <file_name> <topic>
import google.generativeai as genai
import os, sys, re, time, random
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

summaries_dir = sys.argv[1]
file_name     = sys.argv[2] if len(sys.argv) > 2 else "Document"
topic         = sys.argv[3] if len(sys.argv) > 3 else ""

# Read all summary files in order
files = sorted(f for f in os.listdir(summaries_dir) if f.startswith("summary_"))
combined = " ".join(
    open(os.path.join(summaries_dir, f), encoding="utf-8").read() for f in files
)

prompt = f"""You are an expert summarizer.
File Name: {file_name}
Topic: {topic}
Combine the following chunk summaries into one clear final summary:
{combined}"""

def _call_with_retry(prompt):
    for attempt in range(6):
        try:
            return model.generate_content(prompt, stream=False).text
        except ResourceExhausted as e:
            match = re.search(r'retry in (\d+(\.\d+)?)s', str(e), re.IGNORECASE)
            delay = (float(match.group(1)) if match else 45) + random.uniform(2, 8)
            time.sleep(delay)

print(_call_with_retry(prompt))