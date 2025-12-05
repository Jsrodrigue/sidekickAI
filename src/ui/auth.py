from src.services.auth_service import AuthService

auth_service = AuthService()


def login_user(username: str, password: str):
    """
    Wrapper para la UI: devuelve (success: bool, message: str)
    """
    return auth_service.login(username, password)


def register_user(username: str, password: str):
    """
    Wrapper para la UI: devuelve (success: bool, message: str)
    """
    return auth_service.register(username, password)
