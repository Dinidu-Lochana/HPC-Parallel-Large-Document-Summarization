from google import genai
from dotenv import load_dotenv
import os

load_dotenv(".env")
load_dotenv(os.path.join("..", ".env"))

api_key = os.getenv("GEMINI_API_KEY")
print(f"API key prefix: {api_key[:8]}...")

client = genai.Client(api_key=api_key)

print("\nAvailable models that support generateContent:\n")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(f"  {m.name}")
