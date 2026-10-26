"""Central configuration and project paths."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb_docs"
KB_DRAFTS_DIR = DATA_DIR / "kb_drafts"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

TICKETS_RAW_FILE = DATA_DIR / "tickets_raw.csv"
DEMO_TICKETS_FILE = DATA_DIR / "demo_tickets.csv"
TICKETS_CLEAN_FILE = PROCESSED_DIR / "tickets_clean.csv"
TOPIC_PROFILE_FILE = PROCESSED_DIR / "topic_profile.json"
INTERACTIONS_LOG = LOG_DIR / "interactions.jsonl"
API_USAGE_LOG = LOG_DIR / "api_usage.jsonl"
CACHE_FILE = CACHE_DIR / "response_cache.json"
LAST_API_CALL_FILE = LOG_DIR / "last_api_call.txt"
EVAL_SET_FILE = DATA_DIR / "eval_set.json"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EXPERIMENT_LABEL = os.getenv("EXPERIMENT_LABEL", "optimized")
EVAL_RESULTS_FILE = PROCESSED_DIR / f"evaluation_results_{EXPERIMENT_LABEL}.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "helpdesk_kb"

MAX_DAILY_API_CALLS = int(os.getenv("MAX_DAILY_API_CALLS", "25"))
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "30000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))
MAX_KB_DRAFT_TOKENS = int(os.getenv("MAX_KB_DRAFT_TOKENS", "450"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3600"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.30"))
ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() in {"1", "true", "yes", "on"}
API_MIN_INTERVAL_SECONDS = float(os.getenv("API_MIN_INTERVAL_SECONDS", "12"))

for folder in (DATA_DIR, KB_DIR, KB_DRAFTS_DIR, PROCESSED_DIR, CACHE_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)
