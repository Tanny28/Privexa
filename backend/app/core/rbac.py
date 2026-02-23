# ──────────────────────────────────────────────
# RBAC — Role-Based Access Control
# ──────────────────────────────────────────────
from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"


# Permissions mapping
PERMISSIONS = {
    Role.ADMIN: {
        "upload_documents",
        "delete_documents",
        "list_documents",
        "query_documents",
        "manage_users",
    },
    Role.USER: {
        "list_documents",
        "query_documents",
    },
}


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    try:
        r = Role(role)
        return permission in PERMISSIONS.get(r, set())
    except ValueError:
        return False
