# ──────────────────────────────────────────────
# Database Repository Layer
# ──────────────────────────────────────────────
import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.document import Document
from app.models.meeting import Meeting
from app.core.security import hash_password


# ─── User Operations ────────────────────────

def create_user(db: Session, username: str, password: str, role: str = "user") -> User:
    """Create a new user."""
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        hashed_password=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Find a user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Find a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session) -> List[User]:
    """List all users."""
    return db.query(User).all()


# ─── Document Operations ────────────────────

def create_document_record(
    db: Session,
    document_id: str,
    filename: str,
    file_path: str,
    page_count: int,
    chunk_count: int,
    uploaded_by: str,
) -> Document:
    """Save a document metadata record."""
    doc = Document(
        id=document_id,
        filename=filename,
        file_path=file_path,
        page_count=page_count,
        chunk_count=chunk_count,
        uploaded_by=uploaded_by,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document_by_id(db: Session, document_id: str) -> Optional[Document]:
    """Find a document by ID."""
    return db.query(Document).filter(Document.id == document_id).first()


def list_documents(db: Session) -> List[Document]:
    """List all documents."""
    return db.query(Document).all()


def delete_document_record(db: Session, document_id: str) -> bool:
    """Delete a document record."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False


# ─── Meeting Operations ─────────────────────

def create_meeting_record(
    db: Session,
    meeting_id: str,
    title: str,
    transcript_filename: str,
    transcript_path: str,
    report_path: str,
    summary: str,
    analysis_json: str,
    estimated_duration: str,
    uploaded_by: str,
) -> Meeting:
    """Save a meeting analysis record."""
    meeting = Meeting(
        id=meeting_id,
        title=title,
        transcript_filename=transcript_filename,
        transcript_path=transcript_path,
        report_path=report_path,
        summary=summary,
        analysis_json=analysis_json,
        estimated_duration=estimated_duration,
        uploaded_by=uploaded_by,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def get_meeting_by_id(db: Session, meeting_id: str) -> Optional[Meeting]:
    """Find a meeting by ID."""
    return db.query(Meeting).filter(Meeting.id == meeting_id).first()


def list_meetings(db: Session) -> List[Meeting]:
    """List all meetings."""
    return db.query(Meeting).all()


def delete_meeting_record(db: Session, meeting_id: str) -> bool:
    """Delete a meeting record."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if meeting:
        db.delete(meeting)
        db.commit()
        return True
    return False
