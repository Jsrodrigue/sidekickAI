import asyncio
from typing import Optional

from src.core.sidekick import init_sidekick
from src.db.session_repository import SessionRepository
from src.services.folder_service import FolderService
from src.services.session_service import SessionService
from src.services.sidekick_service import SidekickService
from src.ui.ui_controller import UIController

_controller: Optional[UIController] = None
_lock = asyncio.Lock()


async def get_controller() -> UIController:
    """
    Lazily build and return a singleton UIController instance (thread-safe).
    """
    global _controller

    if _controller is not None:
        return _controller

    async with _lock:
        if _controller is None:
            sidekick = await init_sidekick()

            session_repo = SessionRepository()
            session_service = SessionService(session_repo)
            folder_service = FolderService(sidekick)
            sidekick_service = SidekickService(sidekick)

            _controller = UIController(
                session_service=session_service,
                folder_service=folder_service,
                sidekick_service=sidekick_service,
            )

    return _controller
