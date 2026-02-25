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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")          # default / RAG model
OLLAMA_RAG_MODEL = os.getenv("OLLAMA_RAG_MODEL", OLLAMA_MODEL)    # RAG Q&A
OLLAMA_MEETING_MODEL = os.getenv("OLLAMA_MEETING_MODEL", "mistral:latest")  # meeting analysis

# ─── Meeting LLM Options ────────────────────
MEETING_LLM_NUM_PREDICT = 512      # cap output tokens for speed
MEETING_LLM_TEMPERATURE = 0.1      # deterministic
MEETING_LLM_TOP_K = 10             # restrict sampling pool
MEETING_TRANSCRIPT_MAX_CHARS = 6000   # single-pass limit before chunking

# ─── Whisper (Speech-to-Text) ────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")   # tiny|base|small|medium|large-v3
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")         # skip auto-detection
WHISPER_BEAM_SIZE = 1               # greedy decoding — 3x faster
WHISPER_VAD_FILTER = True           # skip silent parts
WHISPER_VAD_MIN_SILENCE_MS = 500    # skip pauses > 0.5s
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

# ─── RAG Retrieval ───────────────────────────
TOP_K_RESULTS = 5         # number of chunks to retrieve

# ─── Auth / JWT ──────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "aegisnode-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ─── Allowed file types ─────────────────────
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_AUDIO_EXTENSIONS_SET = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
MAX_FILE_SIZE_MB = 50
