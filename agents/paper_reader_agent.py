import os
from dotenv import load_dotenv

# Load from local .env (for local development)
load_dotenv()

# Try getting API key from env
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("API key missing! Please set OPENROUTER_API_KEY.")

print("API Key Loaded Successfully.")
