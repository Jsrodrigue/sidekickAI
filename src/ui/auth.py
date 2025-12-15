from __future__ import annotations

from src.services.auth_service import AuthService

auth_service = AuthService()


def login_user(username: str, password: str):
    """
    UI wrapper: returns (success: bool, message: str)
    """
    return auth_service.login(username, password)


def register_user(username: str, password: str):
    """
    UI wrapper: returns (success: bool, message: str)
    """
    return auth_service.register(username, password)
