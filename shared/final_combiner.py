#!/usr/bin/env python3
"""
Combines per-chunk summaries into one coherent final summary.
Usage: python3 final_combiner.py <summaries_file> <topic> <output_file>
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL  = "llama-3.3-70b-versatile"
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def main():
    if len(sys.argv) != 4:
        print("Usage: final_combiner.py <summaries_file> <topic> <output_file>", file=sys.stderr)
        sys.exit(1)

    summaries_file, topic, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        with open(summaries_file, "r", encoding="utf-8", errors="replace") as f:
            summaries = f.read()
    except OSError as e:
        print(f"[final_combiner] cannot read {summaries_file}: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = (
        f"The following are summaries of different sections of a document about '{topic}':\n\n"
        f"{summaries}\n\n"
        f"Create a single coherent final summary capturing all key points."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        final_summary = response.choices[0].message.content
    except Exception as e:
        final_summary = f"[Combination error: {e}]\n\n--- Chunk Summaries ---\n{summaries}"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(final_summary)


if __name__ == "__main__":
    main()
