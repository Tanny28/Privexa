# ──────────────────────────────────────────────
# Document Model (SQLAlchemy)
# ──────────────────────────────────────────────
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql import func
from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_by = Column(String, nullable=False)  # user ID
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
