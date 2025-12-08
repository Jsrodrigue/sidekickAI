"""
Gradio wrappers and helpers for Sidekick AI.

Responsibilities:
- Initialize Sidekick and services exactly once (lazy + thread-safe)
- Expose async wrapper functions ready to be used as Gradio callbacks
"""

import asyncio
from typing import Optional

import gradio as gr

from src.core.sidekick import init_sidekick, SEARCH_K
from src.ui.ui_handlers import UIHandlers
from src.services.session_service import SessionService
from src.services.folder_service import FolderService
from src.services.sidekick_service import SidekickService
from src.db.session_repository import SessionRepository

# -------------------- Globals --------------------

sidekick: Optional[object] = None
handlers: Optional[UIHandlers] = None
_init_lock = asyncio.Lock()

# Max retrieval k (should match retriever search_kwargs["k"])
MAX_RETRIEVAL_K = SEARCH_K


# -------------------- Initialization --------------------

async def ensure_initialized() -> UIHandlers:
    """
    Ensure that Sidekick and UIHandlers are initialized exactly once.

    This function is safe to call from multiple concurrent coroutines.
    """
    global sidekick, handlers

    # Fast path: already initialized
    if sidekick is not None and handlers is not None:
        return handlers

    # Slow path: initialize with a lock
    async with _init_lock:
        if sidekick is None:
            sidekick = await init_sidekick()

        if handlers is None:
            # Create services
            session_repo = SessionRepository()
            session_service = SessionService(session_repo)
            folder_service = FolderService(sidekick)
            sidekick_service = SidekickService(sidekick)

            # Wire them into a single UIHandlers instance
            handlers = UIHandlers(
                session_service=session_service,
                folder_service=folder_service,
                sidekick_service=sidekick_service,
            )

    return handlers


# -------------------- Helpers --------------------

def format_active_folder_label(folder: Optional[str]) -> str:
    """Helper to format the 'active folder' label."""
    if folder:
        return f"**Active folder for chat:** `{folder}`"
    return "**Active folder for chat:** _none selected_"


# -------------------- Wrappers (for Gradio) --------------------

async def wrapped_get_folders(username: Optional[str]):
    """Return the list of folders for the given user or [] if not logged in."""
    if not username:
        return []
    h = await ensure_initialized()
    return h.get_folders(username)


async def wrapped_add_folder(username: Optional[str], folder: str):
    """
    Add and index a folder for the given user.

    Returns:
        - Status message (str)
        - gr.update for the folder dropdown (choices + selected value)
    """
    if not username:
        return "Not logged in", gr.update()
    h = await ensure_initialized()
    msg, folders = await h.add_folder(username, folder)
    # Return status label + updated dropdown choices
    return msg, gr.update(choices=folders, value=folder)


async def wrapped_remove_folder(username: Optional[str], folder: Optional[str]):
    """
    Remove a folder from the user's indexed folders.

    Returns:
        - Status message (str)
        - gr.update for the folder dropdown
    """
    if not username or not folder:
        return "Not logged in", gr.update(choices=[], value=None)
    h = await ensure_initialized()
    msg, folders = await h.remove_folder(username, folder)
    return msg, gr.update(
        choices=folders,
        value=folders[0] if folders else None,
    )


async def wrapped_index_folder(
    username: Optional[str],
    folder: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Index a folder with custom chunk_size and chunk_overlap.

    Returns a simple status string.
    """
    if not username or not folder:
        return "Not logged in"
    h = await ensure_initialized()
    return await h.index_folder(username, folder, chunk_size, chunk_overlap)


async def wrapped_reindex_folder(
    username: Optional[str],
    folder: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Force reindex a folder with custom chunk_size and chunk_overlap.

    Returns a simple status string.
    """
    if not username or not folder:
        return "Not logged in"
    h = await ensure_initialized()
    return await h.reindex_folder(username, folder, chunk_size, chunk_overlap)


async def wrapped_chat(
    username: Optional[str],
    folder: Optional[str],
    message: str,
    top_k: int,
):
    """
    Chat wrapper: returns messages in Gradio type="messages" format
    plus an empty string to clear the textbox.

    top_k controls how many documents RAG will retrieve.
    """
    if not username or not folder:
        # Warning message if someone somehow sends without being logged in
        return (
            [
                {
                    "role": "assistant",
                    "content": "Please log in and select a folder before chatting.",
                }
            ],
            "",
        )
    h = await ensure_initialized()
    # UIHandlers.chat is expected to accept (username, folder, message, top_k)
    history, _ = await h.chat(username, folder, message, top_k)
    # 1st -> Chatbot (list of {role, content}), 2nd -> textbox (cleared)
    return history, ""


async def wrapped_load_chat(username: Optional[str], folder: Optional[str]):
    """
    Load existing chat history for (username, folder).
    """
    if not username or not folder:
        return []
    h = await ensure_initialized()
    history = await h.load_chat(username, folder)
    return history


async def wrapped_clear_chat(username: Optional[str], folder: Optional[str]):
    """
    Clear the server-side session messages for this (username, folder)
    and return an empty list for the Chatbot.
    """
    if not username or not folder:
        return []
    h = await ensure_initialized()
    messages = h.clear_chat(username, folder)
    # Chatbot(type="messages") expects a list of messages (dicts), so [] is valid
    return messages
