import google.generativeai as genai
import os
import sys
from pypdf import PdfReader
from dotenv import load_dotenv



# Load .env from parent folder
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def read_text_file(file_path):
    """
    Read a plain text file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


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


def combine_summaries(summaries_text, topic=""):
    """
    Combine multiple summaries into a coherent final summary
    Used by MPI to merge chunk summaries
    """
    prompt = f"""
    You are an expert summarizer.
    
    Topic: {topic}
    
    Combine the following summaries into a clear, coherent final summary:
    
    {summaries_text}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error during final summarization: {str(e)}]"


if __name__ == "__main__":
    # CLI interface for MPI to call
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <command> [args...]")
        print("Commands:")
        print("  summarize_chunk <input_file> <topic> <output_file>")
        print("  combine_summaries <input_file> <topic> <output_file>")
        print("  extract_pdf <pdf_file> <output_txt_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "summarize_chunk":
        if len(sys.argv) != 5:
            print("Usage: python summarizer.py summarize_chunk <input_file> <topic> <output_file>")
            sys.exit(1)
        
        input_file = sys.argv[2]
        topic = sys.argv[3]
        output_file = sys.argv[4]
        
        # Read chunk text
        chunk_text = read_text_file(input_file)
        
        # Summarize
        summary = summarize_chunk(chunk_text, file_name="", topic=topic)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
    
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
        
        if not PDF_SUPPORT:
            print("Error: pypdf is not installed. Cannot extract PDF text.")
            sys.exit(1)
        
        pdf_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Extract text from PDF
        with open(pdf_file, 'rb') as f:
            text = read_pdf(f)
        
        # Write to text file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)