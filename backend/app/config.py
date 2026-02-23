# ──────────────────────────────────────────────
# AegisNode Backend Configuration
# ──────────────────────────────────────────────
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Database ────────────────────────────────
DATABASE_URL = f"sqlite:///{DATA_DIR / 'aegisnode.db'}"

# ─── Embedding Model ────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── Chunking ───────────────────────────────
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between chunks

# ─── FAISS ───────────────────────────────────
FAISS_INDEX_PATH = str(VECTORSTORE_DIR / "faiss_index")

# ─── LLM (Ollama) ───────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ─── RAG Retrieval ───────────────────────────
TOP_K_RESULTS = 5         # number of chunks to retrieve

# ─── Auth / JWT ──────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "aegisnode-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ─── Allowed file types ─────────────────────
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 50
