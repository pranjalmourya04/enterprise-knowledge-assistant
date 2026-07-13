"""
Central configuration for the Knowledge Assistant backend.
Keeping this in one place means we're not hardcoding paths/constants
across services as the project grows week over week.
"""
from pathlib import Path

# --- Paths ---
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking (Week 1) ---
CHUNK_SIZE_TOKENS = 600      # target chunk size, roughly 500-800 as per roadmap
CHUNK_OVERLAP_TOKENS = 100   # overlap between consecutive chunks

# --- Embedding model (Week 1) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- ChromaDB (Week 1) ---
CHROMA_COLLECTION_NAME = "knowledge_assistant_docs"

# --- Upload constraints ---
ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = 20
