"""Command line interface for the sample service."""

from __future__ import annotations

import json
import sys

from models import User
from services import NotificationService, UserService


def main() -> int:
    """Entry point: parse argv and dispatch to a command."""
    if len(sys.argv) < 2:
        print("usage: cli.py <register|lookup|notify> ...")
        return 2
    command = sys.argv[1]
    service = UserService()
    if command == "register":
        user = service.register(User(user_id=sys.argv[2], email=sys.argv[3]))
        print(json.dumps(user.to_dict()))
        return 0
    if command == "lookup":
        user = service.get(sys.argv[2])
        print(json.dumps(user.to_dict() if user else None))
        return 0
    if command == "notify":
        notifier = NotificationService()
        user = service.get(sys.argv[2])
        if user is None:
            print("unknown user")
            return 1
        notifier.send_email(user, sys.argv[3])
        print("notified")
        return 0
    print(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
