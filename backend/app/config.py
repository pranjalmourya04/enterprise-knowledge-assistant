from dotenv import load_dotenv
import os

load_dotenv()
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_TOKENS = 600      # target chunk size, roughly 500-800 as per roadmap
CHUNK_OVERLAP_TOKENS = 100   # overlap between consecutive chunks

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_COLLECTION_NAME = "knowledge_assistant_docs"

CHROMA_SIMILARITY_SPACE = "cosine"

DEFAULT_TOP_K = 5

CANDIDATE_POOL_SIZE = 20  # hybrid search returns this many candidates for re-ranking

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_TOP_K = 5  # after re-ranking, this many chunks go to the LLM

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = 20

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    print("[config] WARNING: GROQ_API_KEY not set. LLM generation will fail "
          "until you create a .env file with your Groq API key.")

SENSITIVITY_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

ROLE_CLEARANCE = {
    "intern": 0,        # public only
    "employee": 1,      # public + internal
    "manager": 2,        # + confidential
    "hr": 2,              # + confidential
    "admin": 3,           # everything, including restricted
}

DEFAULT_SENSITIVITY = "internal"
DEFAULT_DEPARTMENT = "general"