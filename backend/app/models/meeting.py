# ──────────────────────────────────────────────
# Meeting Model (SQLAlchemy)
# ──────────────────────────────────────────────
from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func
from app.db.database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    transcript_filename = Column(String, nullable=False)
    transcript_path = Column(String, nullable=False)
    report_path = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    analysis_json = Column(Text, nullable=True)  # JSON string of full analysis
    estimated_duration = Column(String, nullable=True)
    uploaded_by = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
