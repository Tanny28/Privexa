# ──────────────────────────────────────────────
# Database Repository Layer
# ──────────────────────────────────────────────
import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.document import Document
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
