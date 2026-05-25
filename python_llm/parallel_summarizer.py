import os
import time
from datetime import datetime
import concurrent.futures
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# Load .env from parent folder
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def read_pdf(file_path):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_text_file(file_path):
    """
    Read a plain text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text(text, chunk_size=2000):
    """
    Split text into chunks of specified size.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def summarize_chunk_api(chunk_data):
    """
    Function to summarize a single chunk.
    Designed to be run in parallel.
    chunk_data is a tuple of (chunk_index, chunk_text, file_name, topic)
    """
    chunk_index, chunk_text, file_name, topic = chunk_data
    
    prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Summarize the following text clearly and concisely:

    {chunk_text}
    """

    print(f"[Thread] Starting API request for chunk {chunk_index}...")
    start_time = time.time()
    
    try:
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
                    print(f"[Thread] Chunk {chunk_index} streaming started at {exact_time} ({first_token_time - start_time:.2f} seconds elapsed).")
                text += part.text
                
        finish_time = time.time()
        print(f"[Thread] Chunk {chunk_index} finished in {finish_time - start_time:.2f} seconds.")
        return (chunk_index, text)
    except Exception as e:
        print(f"[Thread] Error in chunk {chunk_index}: {e}")
        return (chunk_index, f"[Error: {str(e)}]")

def parallel_summarize_document(file_path, topic="", max_workers=5):
    """
    Summarizes a document by processing chunks in parallel using ThreadPoolExecutor.
    """
    start_total = time.time()

    # Read the file
    file_name = os.path.basename(file_path)
    if file_path.lower().endswith('.pdf'):
        text = read_pdf(file_path)
    else:
        text = read_text_file(file_path)
        
    # Split text into chunks
    chunks = split_text(text)
    print(f"Document split into {len(chunks)} chunks.")

    # Prepare data for parallel execution
    chunk_data_list = [(i, chunk, file_name, topic) for i, chunk in enumerate(chunks)]
    
    summaries = []
    
    # Use ThreadPoolExecutor for parallel API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the summarize function to the chunk data
        # Using executor.map ensures results are in the same order as input
        results = executor.map(summarize_chunk_api, chunk_data_list)
        
        # Collect results
        for index, summary_text in results:
            summaries.append(summary_text)

    combined_summaries = " ".join(summaries)

    # Final summary to combine the chunk summaries
    final_prompt = f"""
    You are an expert summarizer.

    File Name: {file_name}
    Topic: {topic}

    Combine the following summaries into a clear final summary:

    {combined_summaries}
    """
    
    print("\nGenerating final combined summary...")
    final_request_start = time.time()
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=final_prompt
    )
    
    first_token_time = None
    final_text = ""
    for part in response:
        if part.text:
            if first_token_time is None:
                first_token_time = time.time()
                exact_time = datetime.now().strftime("%I:%M:%S %p")
                print(f"[Final] Streaming started at {exact_time} ({first_token_time - final_request_start:.2f} seconds elapsed).")
            final_text += part.text
    
    finish_total = time.time()
    print(f"\n[Total] Parallel summarization finished in {finish_total - start_total:.2f} seconds.")
    
    return final_text

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parallel_summarizer.py <file_path> [topic] [max_workers]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else ""
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    summary = parallel_summarize_document(file_path, topic, max_workers)
    
    print("\n=== FINAL SUMMARY ===")
    print(summary)
    print("=====================")
