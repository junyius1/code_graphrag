"""HTTP application wiring for the sample service."""

from __future__ import annotations

from models import User
from services import NotificationService, UserService

DEFAULT_PORT = 8000


class App:
    """A minimal request router over the user services."""

    def __init__(self) -> None:
        self.users = UserService()
        self.notifications = NotificationService()

    def handle(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Route an incoming request to the matching handler."""
        if method == "POST" and path == "/users":
            return self.handle_register(payload or {})
        if method == "GET" and path.startswith("/users/"):
            return self.handle_lookup(path.split("/")[-1])
        if method == "POST" and path.startswith("/users/") and path.endswith("/notify"):
            return self.handle_notify(path.split("/")[2], payload or {})
        return {"error": "not found"}

    def handle_register(self, payload: dict) -> dict:
        """Create a user from a request payload."""
        user = self.users.register(User(user_id=payload["id"], email=payload.get("email", "")))
        return {"status": "created", "user": user.to_dict()}

    def handle_lookup(self, user_id: str) -> dict:
        """Fetch one user by id."""
        user = self.users.get(user_id)
        if user is None:
            return {"error": "not found"}
        return {"user": user.to_dict()}

    def handle_notify(self, user_id: str, payload: dict) -> dict:
        """Notify one user by id."""
        user = self.users.get(user_id)
        if user is None:
            return {"error": "not found"}
        self.notifications.send_email(user, payload.get("message", ""))
        return {"status": "notified"}


app = App()
