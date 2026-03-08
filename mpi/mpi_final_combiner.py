import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

def combine_summaries(combined_file, topic, output_file):
    with open(combined_file, 'r', encoding='utf-8') as f:
        combined_summaries = f.read()
    
    prompt = f"""
    You are an expert summarizer.
    
    Topic: {topic}
    
    Combine the following summaries into a clear, coherent final summary:
    
    {combined_summaries}
    """
    
    try:
        response = model.generate_content(prompt)
        final_summary = response.text
    except Exception as e:
        final_summary = f"[Error during final summarization: {str(e)}]"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_summary)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mpi_final_combiner.py <combined_file> <topic> <output_file>")
        sys.exit(1)
    
    combined_file = sys.argv[1]
    topic = sys.argv[2]
    output_file = sys.argv[3]
    
    combine_summaries(combined_file, topic, output_file)
