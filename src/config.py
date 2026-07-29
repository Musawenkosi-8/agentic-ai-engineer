import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    """Configuration class to manage environment variables."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
