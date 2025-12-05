from src.db.session_repository import SessionRepository
from src.core.state import SidekickState


class SessionService:
    def __init__(self, session_repo: SessionRepository):
        self.session_repo = session_repo
        # cache en memoria: clave = (username, folder)
        self.sessions = {}

    def _key(self, username: str, folder: str) -> tuple:
        return (username, folder or "")

    def load(self, username: str, folder: str) -> SidekickState:
        """
        Load session state for a specific (username, folder),
        using in-memory cache + DB fallback.
        """
        key = self._key(username, folder)

        if key in self.sessions:
            return self.sessions[key]

        state = self.session_repo.load(username, folder)
        self.sessions[key] = state
        return state

    def save(self, username: str, folder: str, state: SidekickState):
        """
        Save session state for a specific (username, folder),
        updating cache and DB.
        """
        key = self._key(username, folder)
        self.sessions[key] = state
        self.session_repo.save(username, folder, state)
