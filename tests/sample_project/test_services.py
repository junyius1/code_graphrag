"""Tests for the sample services."""

from __future__ import annotations

import pytest
from models import User
from services import NotificationService, UserService


def test_register_and_get() -> None:
    service = UserService()
    user = service.register(User(user_id="u1", email="a@b.c"))
    assert service.get("u1") is user
    assert service.count() == 1


def test_register_duplicate_raises() -> None:
    service = UserService()
    service.register(User(user_id="u1", email="a@b.c"))
    with pytest.raises(ValueError):
        service.register(User(user_id="u1", email="x@y.z"))


def test_notification_requires_email() -> None:
    notifier = NotificationService()
    with pytest.raises(ValueError):
        notifier.send_email(User(user_id="u1", email=""), "hi")
