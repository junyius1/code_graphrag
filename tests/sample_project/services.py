"""Business logic services for the sample service."""

from __future__ import annotations

import json

from models import AdminUser, User

MAX_USERS = 10_000
STORAGE_PATH = "data/users.json"


class UserService:
    """Persists and queries user records."""

    def __init__(self, store_path: str = STORAGE_PATH) -> None:
        self.store_path = store_path
        self._users: dict[str, User] = {}

    def register(self, user: User) -> User:
        """Register a new user, rejecting duplicates."""
        if user.user_id in self._users:
            raise ValueError(f"user {user.user_id} already registered")
        if len(self._users) >= MAX_USERS:
            raise RuntimeError("user store is full")
        self._users[user.user_id] = user
        return user

    def get(self, user_id: str) -> User | None:
        """Return the user with the given id, or None."""
        return self._users.get(user_id)

    def count(self) -> int:
        """Return the number of registered users."""
        return len(self._users)

    def dump(self) -> str:
        """Serialize all users to a JSON string."""
        return json.dumps({uid: u.to_dict() for uid, u in self._users.items()})


class NotificationService:
    """Sends notifications to users."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_email(self, user: User, message: str) -> None:
        """Send an email notification to a user."""
        if not user.email:
            raise ValueError("user has no email address")
        self.sent.append(f"{user.email}: {message}")

    def send_admin_alert(self, user: AdminUser, message: str) -> None:
        """Send an alert to an admin user."""
        self.send_email(user, f"[ADMIN] {message}")
