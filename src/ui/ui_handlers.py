from __future__ import annotations

import re
from langchain_core.messages import AIMessage, HumanMessage

# Special key for the user's GLOBAL state (e.g. list of indexed folders)
_GLOBAL_FOLDER_KEY = "__global__"
# Special key for chats when NO folder is selected
_NO_FOLDER_CHAT_KEY = "__no_folder__"

# Reminder injected into the prompt when Python tool is enabled (NOT meant to be visible)
IMPORTANT_PYTHON_PRINT_REMINDER = (
    "IMPORTANT REMINDER: If you use the Python tool to compute or inspect anything, "
    "ALWAYS use print() statements to show the final values you want the user to see. "
    "Do not rely on implicit output.\n"
)

# -------------------- LaTeX cleaner --------------------

_LATEX_BRACKET_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)   # \[ ... \]
_LATEX_PAREN_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)    # \( ... \)
_LATEX_SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def clean_latex_to_double_dollars(text: str) -> str:
    """
    Normalize LaTeX delimiters to $$...$$

    - Converts \[...\] to $$...$$
    - Converts \(...\) to $$...$$
    - Converts $...$ to $$...$$ only when it looks like math (contains a backslash
      or common math symbols), to avoid breaking currency like "$10".
    """
    if not text:
        return text

    text = _LATEX_BRACKET_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    text = _LATEX_PAREN_INLINE.sub(lambda m: f"$${m.group(1).strip()}$$", text)

    def _maybe_upgrade_single_dollar(m: re.Match) -> str:
        inner = (m.group(1) or "").strip()
        looks_like_math = (
            "\\" in inner
            or any(sym in inner for sym in ["=", "+", "-", "*", "/", "^", "_", "{", "}", "\\frac", "\\sum", "\\int"])
        )
        if not looks_like_math:
            return m.group(0)  # likely currency or plain text
        return f"$${inner}$$"

    text = _LATEX_SINGLE_DOLLAR.sub(_maybe_upgrade_single_dollar, text)
    return text


def _hide_injected_user_reminder(messages, injected: str, original: str) -> None:
    """
    If we injected a hidden reminder into the user prompt, undo that in the
    stored message history so it never appears in chat/history.

    We try to find the latest user message matching the injected content and replace it.
    """
    if not messages:
        return

    # Walk backwards to find the most recent user message
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        # LangChain HumanMessage
        if isinstance(msg, HumanMessage) and msg.content == injected:
            messages[i].content = original
            return

        # Dict message format
        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == injected:
            msg["content"] = original
            return


class UIHandlers:
    def __init__(self, session_service, folder_service, sidekick_service):
        self.session_service = session_service
        self.folder_service = folder_service
        self.sidekick_service = sidekick_service

    # -------------------- Load / Save GLOBAL Session --------------------

    def load_session(self, username: str):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            return True, "Session loaded", state.messages
        except Exception as e:
            return False, f"Error: {e}", []

    def save_session(self, username: str, state):
        try:
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, state)
            return True, "Session saved"
        except Exception as e:
            return False, f"Error: {e}"

    # -------------------- Chat (per user & folder OR no-folder) --------------------

    async def chat(
        self,
        username: str,
        folder: str | None,
        prompt: str,
        top_k: int,
        enabled_tools: list[str] | None = None,
    ):
        try:
            if not username:
                raise ValueError("No username provided for chat.")

            enabled_tools = enabled_tools or []

            folder_key = folder if folder else _NO_FOLDER_CHAT_KEY
            state = self.session_service.load(username, folder_key)

            # Inject reminder ONLY for the model call, but we will remove it from history afterwards
            original_prompt = prompt
            injected_prompt = prompt
            injected = False

            if "python" in enabled_tools:
                injected_prompt = IMPORTANT_PYTHON_PRINT_REMINDER + prompt
                injected = True

            new_state = await self.sidekick_service.send_message(
                prompt=injected_prompt,
                state=state,
                username=username,
                folder=folder if folder else None,
                top_k=top_k,
                enabled_tools=enabled_tools,
            )

            # Remove the injected reminder from stored history so it never appears in the UI
            if injected:
                _hide_injected_user_reminder(new_state.messages, injected_prompt, original_prompt)

            # Save updated state AFTER cleaning it
            self.session_service.save(username, folder_key, new_state)

            # Convert to Gradio format + LaTeX cleanup for assistant messages
            gr_messages = []
            for msg in new_state.messages:
                if isinstance(msg, HumanMessage):
                    gr_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    gr_messages.append(
                        {"role": "assistant", "content": clean_latex_to_double_dollars(msg.content)}
                    )
                elif isinstance(msg, dict) and "role" in msg:
                    content = msg.get("content", "")
                    if msg["role"] == "assistant":
                        content = clean_latex_to_double_dollars(content)
                    gr_messages.append({"role": msg["role"], "content": content})

            return gr_messages, ""

        except Exception as e:
            return [{"role": "assistant", "content": f"Error: {e}"}], ""

    async def load_chat(self, username: str, folder: str | None):
        if not username:
            return []

        try:
            folder_key = folder if folder else _NO_FOLDER_CHAT_KEY
            state = self.session_service.load(username, folder_key)
        except Exception:
            return []

        gr_messages = []
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                gr_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gr_messages.append(
                    {"role": "assistant", "content": clean_latex_to_double_dollars(msg.content)}
                )
            elif isinstance(msg, dict) and "role" in msg:
                content = msg.get("content", "")
                if msg["role"] == "assistant":
                    content = clean_latex_to_double_dollars(content)
                gr_messages.append({"role": msg["role"], "content": content})

        return gr_messages

    def clear_chat(self, username: str, folder: str | None):
        try:
            if not username:
                return []

            folder_key = folder if folder else _NO_FOLDER_CHAT_KEY
            state = self.session_service.load(username, folder_key)
            state.messages = []
            self.session_service.save(username, folder_key, state)
            return []
        except Exception:
            return []

    # -------------------- Folder Management (GLOBAL per user) --------------------

    def get_folders(self, username: str):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            return getattr(state, "indexed_directories", [])
        except Exception:
            return []

    async def add_folder(self, username: str, folder: str):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)

            if not hasattr(state, "indexed_directories"):
                state.indexed_directories = []

            if folder not in state.indexed_directories:
                state.indexed_directories.append(folder)
                new_state = await self.folder_service.ensure_indexed(folder, state)
                self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder added", new_state.indexed_directories

            return "Folder already exists", state.indexed_directories
        except Exception as e:
            return f"Error: {e}", []

    async def remove_folder(self, username: str, folder: str):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)

            if hasattr(state, "indexed_directories") and folder in state.indexed_directories:
                state.indexed_directories.remove(folder)
                new_state = await self.folder_service.clear_folder(folder, state)
                self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder removed", new_state.indexed_directories

            return "Folder not found", getattr(state, "indexed_directories", [])
        except Exception as e:
            return f"Error: {e}", []

    async def index_folder(self, username: str, folder: str, chunk_size: int, chunk_overlap: int):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.ensure_indexed(
                folder=folder,
                state=state,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder indexed"
        except Exception as e:
            return f"Error: {e}"

    async def reindex_folder(self, username: str, folder: str, chunk_size: int, chunk_overlap: int):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.reindex_folder(
                folder=folder,
                state=state,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder reindexed"
        except Exception as e:
            return f"Error: {e}"

    async def clear_folder(self, username: str, folder: str):
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.clear_folder(folder, state)
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder cleared"
        except Exception as e:
            return f"Error: {e}"
