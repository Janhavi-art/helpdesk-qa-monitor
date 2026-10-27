"""Central paths and environment settings for Basic HelpDeskQA RAG."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "data" / "kb_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
THRESHOLD_FILE = BASE_DIR / "data" / "similarity_threshold.json"
COLLECTION_NAME = "helpdesk_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
TOP_K = int(os.getenv("TOP_K", "3"))
