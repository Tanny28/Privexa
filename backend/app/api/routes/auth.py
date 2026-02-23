# ──────────────────────────────────────────────
# Auth Routes — Login, Register, User Info
# ──────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.repositories import get_user_by_username, create_user, list_users
from app.core.security import verify_password, create_access_token
from app.api.dependencies import get_current_user, require_permission
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ─── Schemas ─────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool


# ─── Endpoints ───────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT token."""
    user = get_user_by_username(db, req.username)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    token = create_access_token(data={"sub": user.id, "role": user.role})
    return TokenResponse(access_token=token)


@router.post("/register", response_model=UserResponse)
def register(
    req: RegisterRequest,
    current_user: User = Depends(require_permission("manage_users")),
    db: Session = Depends(get_db),
):
    """Register a new user (Admin only)."""
    existing = get_user_by_username(db, req.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists.",
        )

    user = create_user(db, req.username, req.password, req.role)
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        is_active=current_user.is_active,
    )


@router.get("/users", response_model=list[UserResponse])
def get_users(
    current_user: User = Depends(require_permission("manage_users")),
    db: Session = Depends(get_db),
):
    """List all users (Admin only)."""
    users = list_users(db)
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
        )
        for u in users
    ]
