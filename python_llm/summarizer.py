import google.generativeai as genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import time

# Load .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


def summarize_chunk(chunk, file_name="", topic=""):
    prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Summarize the following text clearly and concisely:

    {chunk}
    """

    request_start = time.time()

    response = model.generate_content(prompt, stream=True)

    first_token_time = None
    text = ""

    for part in response:
        if part.text:

            if first_token_time is None:
                first_token_time = time.time()
                print(f"[Chunk] Streaming started at {first_token_time - request_start:.2f} seconds")

            text += part.text

    finish_time = time.time()
    print(f"[Chunk] Streaming finished at {finish_time - request_start:.2f} seconds")
    print("========================================================================")
    return text


def summarize_document(file, topic=""):

    start_total = time.time()

    text = read_pdf(file)
    chunks = split_text(text)

    file_name = getattr(file, "name", "Document")

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

    response = model.generate_content(final_prompt, stream=True)

    first_token_time = None

    for part in response:
        if part.text:

            if first_token_time is None:
                first_token_time = time.time()
                print(f"[Final] Streaming started at {first_token_time - request_start:.2f} seconds")

            yield part.text

    finish_time = time.time()

    print(f"[Final] Streaming finished at {finish_time - request_start:.2f} seconds")
    print(f"[Total] Full summarization finished in {finish_time - start_total:.2f} seconds")