import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

def summarize_chunk(chunk_file, topic, output_file):
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunk_text = f.read()
    
    prompt = f"""
    You are an expert summarizer.
    
    Topic: {topic}
    
    Summarize the following text clearly and concisely:
    
    {chunk_text}
    """
    
    try:
        response = model.generate_content(prompt)
        summary = response.text
    except Exception as e:
        summary = f"[Error during summarization: {str(e)}]"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mpi_python_wrapper.py <input_file> <topic> <output_file>")
        sys.exit(1)
    
    chunk_file = sys.argv[1]
    topic = sys.argv[2]
    output_file = sys.argv[3]
    
    summarize_chunk(chunk_file, topic, output_file)
