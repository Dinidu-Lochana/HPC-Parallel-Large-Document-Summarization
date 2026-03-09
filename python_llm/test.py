import google.generativeai as genai 
import os
from dotenv import load_dotenv 

# Load .env from parent folder
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# List available models
models = genai.list_models()
for m in models:
    print(m)