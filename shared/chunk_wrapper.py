#!/usr/bin/env python3
"""
Shared chunk summarizer — called by all C programs (MPI / OpenMP / Hybrid).
Usage: python3 chunk_wrapper.py <input_file> <topic> <output_file>
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL   = "llama-3.3-70b-versatile"
client  = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def main():
    if len(sys.argv) != 4:
        print("Usage: chunk_wrapper.py <input_file> <topic> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_file, topic, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError as e:
        print(f"[chunk_wrapper] cannot read {input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not text:
        with open(output_file, "w") as f:
            f.write("[empty chunk]")
        return

    prompt = (
        f"Summarize the following text concisely. Topic context: '{topic}'.\n\n"
        f"{text}\n\n"
        f"Provide a clear, concise summary in 2-4 sentences."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"[Summarization error: {e}]"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
