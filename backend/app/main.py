# ──────────────────────────────────────────────
# AegisNode — FastAPI Main Application
# ──────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import init_db, get_db, SessionLocal
from app.db.repositories import get_user_by_username, create_user
from app.api.routes import auth, documents, query

app = FastAPI(
    title="AegisNode",
    description="Offline Organizational AI Appliance — RAG Backend",
    version="0.1.0",
)

# ─── CORS (allow frontend to connect) ───────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routes ────────────────────────
app.include_router(auth.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


# ─── Startup Event ──────────────────────────
@app.on_event("startup")
def on_startup():
    """Initialize database and seed default admin user."""
    init_db()

    # Create default admin if not exists
    db = SessionLocal()
    try:
        admin = get_user_by_username(db, "admin")
        if not admin:
            create_user(db, username="admin", password="admin123", role="admin")
            print("[AegisNode] Default admin created — username: admin, password: admin123")
        else:
            print("[AegisNode] Admin user already exists.")
    finally:
        db.close()


# ─── Root ────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "AegisNode",
        "tagline": "Secure. Local. Intelligent.",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
