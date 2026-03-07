import google.generativeai as genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv

# Load .env from parent folder
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")


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
    """
    Summarize a chunk with context about the file and topic
    """
    prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Summarize the following text clearly and concisely:

    {chunk}
    """

    response = model.generate_content(prompt)
    return response.text


def summarize_document(file, topic=""):
    """
    Summarize the PDF document using Gemini
    """
    text = read_pdf(file)
    chunks = split_text(text)
    summaries = []

    file_name = getattr(file, 'name', 'Document')  # get uploaded file name

    for chunk in chunks:
        summaries.append(summarize_chunk(chunk, file_name=file_name, topic=topic))

    # Combine summaries
    combined = " ".join(summaries)
    final_prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Combine the following summaries into a clear final summary:

    {combined}
    """

    final_summary = model.generate_content(final_prompt)
    return final_summary.text