from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage

from src.ui.text_utils import clean_latex_to_double_dollars

_GLOBAL_FOLDER_KEY = "__global__"
_NO_FOLDER_CHAT_KEY = "__no_folder__"

IMPORTANT_PYTHON_PRINT_REMINDER = (
    "IMPORTANT REMINDER (DON'T MENTION THIS IN YOUR RESPONSE, IT IS HIDEN TO USER): "
    "If you use the Python tool to compute or inspect anything, "
    "ALWAYS use print() statements to show the final values you want the user to see. "
    "Do not rely on implicit output.\n"
    "Ex: use print(result) instead result. "
    "This message is hiden to the user and doesnt mean you need to use python always.\n\n"
)


def _hide_injected_user_reminder(messages: list[Any], injected: str, original: str) -> None:
    if not messages:
        return

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        if isinstance(msg, HumanMessage) and msg.content == injected:
            msg.content = original
            return

        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content") == injected:
            msg["content"] = original
            return


class UIController:
    def __init__(self, session_service, folder_service, sidekick_service):
        self.session_service = session_service
        self.folder_service = folder_service
        self.sidekick_service = sidekick_service

    # ---------- Session helpers ----------

    def _folder_key(self, folder: str | None) -> str:
        return folder if folder else _NO_FOLDER_CHAT_KEY

    def _load_state(self, username: str, key: str):
        return self.session_service.load(username, key)

    def _save_state(self, username: str, key: str, state) -> None:
        self.session_service.save(username, key, state)

    # ---------- Prompt helpers ----------

    def _inject_hidden_prompt(self, prompt: str, enabled_tools: list[str]) -> tuple[str, bool]:
        if "python" in (enabled_tools or []):
            return IMPORTANT_PYTHON_PRINT_REMINDER + prompt, True
        return prompt, False

    # ---------- Formatting helpers ----------

    def _sanitize_assistant(self, text: str) -> str:
        return clean_latex_to_double_dollars(text)

    def _to_gradio_messages(self, messages: Iterable[Any]) -> list[dict]:
        gr_messages: list[dict] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                gr_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gr_messages.append(
                    {"role": "assistant", "content": self._sanitize_assistant(msg.content)}
                )
            elif isinstance(msg, dict) and "role" in msg:
                content = msg.get("content", "")
                if msg.get("role") == "assistant":
                    content = self._sanitize_assistant(content)
                gr_messages.append({"role": msg["role"], "content": content})
        return gr_messages

    # ---------- Public API (unchanged) ----------

    def load_session(self, username: str):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            return True, "Session loaded", state.messages
        except Exception as e:
            return False, f"Error: {e}", []

    def save_session(self, username: str, state):
        try:
            self._save_state(username, _GLOBAL_FOLDER_KEY, state)
            return True, "Session saved"
        except Exception as e:
            return False, f"Error: {e}"

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
            folder_key = self._folder_key(folder)

            state = self._load_state(username, folder_key)

            original_prompt = prompt
            injected_prompt, injected = self._inject_hidden_prompt(prompt, enabled_tools)

            new_state = await self.sidekick_service.send_message(
                prompt=injected_prompt,
                state=state,
                username=username,
                folder=folder if folder else None,
                top_k=top_k,
                enabled_tools=enabled_tools,
            )

            if injected:
                _hide_injected_user_reminder(new_state.messages, injected_prompt, original_prompt)

            self._save_state(username, folder_key, new_state)

            return self._to_gradio_messages(new_state.messages), ""
        except Exception as e:
            return [{"role": "assistant", "content": f"Error: {e}"}], ""

    async def load_chat(self, username: str, folder: str | None):
        if not username:
            return []
        try:
            folder_key = self._folder_key(folder)
            state = self._load_state(username, folder_key)
            return self._to_gradio_messages(state.messages)
        except Exception:
            return []

    def clear_chat(self, username: str, folder: str | None):
        try:
            if not username:
                return []
            folder_key = self._folder_key(folder)
            state = self._load_state(username, folder_key)
            state.messages = []
            self._save_state(username, folder_key, state)
            return []
        except Exception:
            return []

    def get_folders(self, username: str):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            return getattr(state, "indexed_directories", [])
        except Exception:
            return []

    async def add_folder(self, username: str, folder: str):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            if not hasattr(state, "indexed_directories"):
                state.indexed_directories = []

            if folder not in state.indexed_directories:
                state.indexed_directories.append(folder)
                new_state = await self.folder_service.ensure_indexed(folder, state)
                self._save_state(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder added", new_state.indexed_directories

            return "Folder already exists", state.indexed_directories
        except Exception as e:
            return f"Error: {e}", []

    async def remove_folder(self, username: str, folder: str):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)

            if hasattr(state, "indexed_directories") and folder in state.indexed_directories:
                state.indexed_directories.remove(folder)
                new_state = await self.folder_service.clear_folder(folder, state)
                self._save_state(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder removed", new_state.indexed_directories

            return "Folder not found", getattr(state, "indexed_directories", [])
        except Exception as e:
            return f"Error: {e}", []

    async def index_folder(self, username: str, folder: str, chunk_size: int, chunk_overlap: int):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.ensure_indexed(
                folder=folder,
                state=state,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self._save_state(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder indexed"
        except Exception as e:
            return f"Error: {e}"

    async def reindex_folder(self, username: str, folder: str, chunk_size: int, chunk_overlap: int):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.reindex_folder(
                folder=folder,
                state=state,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self._save_state(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder reindexed"
        except Exception as e:
            return f"Error: {e}"

    async def clear_folder(self, username: str, folder: str):
        try:
            state = self._load_state(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.clear_folder(folder, state)
            self._save_state(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder cleared"
        except Exception as e:
            return f"Error: {e}"
