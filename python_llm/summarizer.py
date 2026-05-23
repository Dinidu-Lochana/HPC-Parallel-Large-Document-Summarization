<<<<<<< HEAD
=======
from google import genai
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431
import os
import sys
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
<<<<<<< HEAD
from groq import Groq
=======
import time
from datetime import datetime
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

<<<<<<< HEAD
MODEL  = "llama-3.3-70b-versatile"
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
=======
# Load .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_chunk(chunk, file_name="", topic=""):
<<<<<<< HEAD
    prompt = (
        f"File: {file_name}\nTopic: {topic}\n\n"
        f"Summarize the following text clearly and concisely:\n\n{chunk}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content
=======
    prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Summarize the following text clearly and concisely:

    {chunk}
    """

    request_start = time.time()

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )

    first_token_time = None
    text = ""

    for part in response:
        if part.text:

            if first_token_time is None:
                first_token_time = time.time()
                exact_time = datetime.now().strftime("%I:%M:%S %p")
                print(f"[Chunk] Streaming started at {exact_time} ({first_token_time - request_start:.2f} seconds elapsed)")

            text += part.text

    finish_time = time.time()
    print(f"[Chunk] Streaming finished at {finish_time - request_start:.2f} seconds")
    print("========================================================================")
    return text


def summarize_document(file, topic="", file_name=None):

    start_total = time.time()

    text = read_pdf(file)
    chunks = split_text(text)

    file_name = file_name or getattr(file, "name", "Document")

    summaries = []

    for chunk in chunks:
        summary = summarize_chunk(chunk, file_name=file_name, topic=topic)
        summaries.append(summary)

    combined = " ".join(summaries)

    final_prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Combine the following summaries into a clear final summary:

    {combined}
    """

    request_start = time.time()
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=final_prompt
    )

    first_token_time = None

    for part in response:
        if part.text:
            if first_token_time is None:
                first_token_time = time.time()
                exact_time = datetime.now().strftime("%I:%M:%S %p")
                print(f"[Final] Streaming started at {exact_time} ({first_token_time - request_start:.2f} seconds elapsed)")

            yield part.text

    finish_time = time.time()
    print(f"[Final] Streaming finished at {finish_time - request_start:.2f} seconds")
    print(f"[Total] Full summarization finished in {finish_time - start_total:.2f} seconds")

def combine_summaries(summaries_text, topic=""):
    prompt = (
        f"Topic: {topic}\n\n"
        f"Combine the following summaries into a clear, coherent final summary:\n\n{summaries_text}"
    )
    try:
<<<<<<< HEAD
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content
=======
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431
    except Exception as e:
        return f"[Error during final summarization: {e}]"


def summarize_document(file, topic=""):
    text     = read_pdf(file)
    chunks   = split_text(text)
    file_name = getattr(file, "name", "Document")
    summaries = [summarize_chunk(c, file_name=file_name, topic=topic) for c in chunks]
    return combine_summaries("\n\n".join(summaries), topic=topic)


# ── CLI interface used by MPI C code ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "summarize_chunk" and len(sys.argv) == 5:
        _, _, input_file, topic, output_file = sys.argv
        text    = read_text_file(input_file)
        summary = summarize_chunk(text, topic=topic)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
<<<<<<< HEAD

    elif command == "combine_summaries" and len(sys.argv) == 5:
        _, _, input_file, topic, output_file = sys.argv
        text  = read_text_file(input_file)
        final = combine_summaries(text, topic=topic)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final)

    elif command == "extract_pdf" and len(sys.argv) == 4:
        _, _, pdf_file, output_file = sys.argv
        with open(pdf_file, "rb") as f:
=======
    
    elif command == "combine_summaries":
        if len(sys.argv) != 5:
            print("Usage: python summarizer.py combine_summaries <input_file> <topic> <output_file>")
            sys.exit(1)
        
        input_file = sys.argv[2]
        topic = sys.argv[3]
        output_file = sys.argv[4]
        
        # Read combined summaries
        summaries_text = read_text_file(input_file)
        
        # Combine
        final_summary = combine_summaries(summaries_text, topic)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_summary)
    
    elif command == "extract_pdf":
        if len(sys.argv) != 4:
            print("Usage: python summarizer.py extract_pdf <pdf_file> <output_txt_file>")
            sys.exit(1)
        
        pdf_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Extract text from PDF
        with open(pdf_file, 'rb') as f:
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431
            text = read_pdf(f)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

    else:
<<<<<<< HEAD
        print(f"Unknown command or wrong args: {sys.argv[1:]}")
        sys.exit(1)
=======
        print(f"Unknown command: {command}")
>>>>>>> d99e81fb0b5e45b62231d7aba116991354717431
