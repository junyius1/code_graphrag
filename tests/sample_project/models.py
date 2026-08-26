"""Domain models for the sample service."""

from __future__ import annotations


class BaseUser:
    """Base class for user records."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    def to_dict(self) -> dict:
        return {"user_id": self.user_id}


class User(BaseUser):
    """A registered user of the service."""

    def __init__(self, user_id: str, email: str) -> None:
        super().__init__(user_id)
        self.email = email
        self.preferences = {}

    def set_preference(self, key: str, value: object) -> None:
        """Store a single user preference."""
        if not key:
            raise ValueError("preference key must be non-empty")
        self.preferences[key] = value

    def get_preference(self, key: str) -> object:
        """Read a user preference, returning None when absent."""
        if key in self.preferences:
            return self.preferences[key]
        return None


class AdminUser(User):
    """A user with administrative privileges."""

    def __init__(self, user_id: str, email: str) -> None:
        super().__init__(user_id, email)
        self.role = "admin"

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["role"] = self.role
        return data
