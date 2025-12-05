import hashlib

from src.db.user_repository import UserRepository


class AuthService:
    def __init__(self):
        self.repo = UserRepository()

    def hash_password(self, password: str) -> str:
        """
        Hash sencillo con SHA256 (para demo / entorno simple).
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str):
        if not username or not password:
            return False, "Username and password required"

        existing = self.repo.get_user(username)
        if existing:
            return False, "Username already exists"

        hashed = self.hash_password(password)
        created = self.repo.create_user(username, hashed)

        if not created:
            return False, "Could not create user"

        return True, "User registered successfully"

    def login(self, username: str, password: str):
        if not username or not password:
            return False, "Username and password required"

        user = self.repo.get_user(username)
        if not user:
            return False, "User not found"

        hashed = self.hash_password(password)
        if hashed != user["password"]:
            return False, "Wrong password"

        return True, "Logged in"
