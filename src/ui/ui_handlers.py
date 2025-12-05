"""
Gradio UI event handlers.
Connects the frontend with the backend services.
"""
from langchain_core.messages import AIMessage, HumanMessage

# Special key for the user's GLOBAL state (e.g. list of indexed folders)
_GLOBAL_FOLDER_KEY = "__global__"


class UIHandlers:
    def __init__(self, session_service, folder_service, sidekick_service):
        self.session_service = session_service
        self.folder_service = folder_service
        self.sidekick_service = sidekick_service

    # -------------------- Load / Save GLOBAL Session --------------------

    def load_session(self, username: str):
        """
        Load the GLOBAL session for a user (not tied to a specific folder).
        Useful if you want to keep other user-level data.
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            return True, "Session loaded", state.messages
        except Exception as e:
            return False, f"Error: {e}", []

    def save_session(self, username: str, state):
        """
        Save the GLOBAL session for a user.
        """
        try:
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, state)
            return True, "Session saved"
        except Exception as e:
            return False, f"Error: {e}"

    # -------------------- Chat (per user & folder) --------------------

    async def chat(self, username: str, folder: str, prompt: str):
        """
        Chat for a specific (username, folder) pair.
        Keeps separate, persistent histories per folder.
        """
        try:
            if not folder:
                raise ValueError("No folder selected for chat.")

            # 1) Load chat state for THIS (username, folder)
            state = self.session_service.load(username, folder)

            # 2) Send message to Sidekick (thread_id uses username + folder)
            new_state = await self.sidekick_service.send_message(
                prompt, state, username, folder
            )

            # 3) Save updated state for THIS (username, folder)
            self.session_service.save(username, folder, new_state)

            # 4) Convert to Gradio `type="messages"` format
            gr_messages = []
            for msg in new_state.messages:
                if isinstance(msg, HumanMessage):
                    gr_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    gr_messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, dict) and "role" in msg:
                    gr_messages.append(
                        {"role": msg["role"], "content": msg.get("content", "")}
                    )

            return gr_messages, ""

        except Exception as e:
            return [{"role": "assistant", "content": f"Error: {e}"}], ""

    async def load_chat(self, username: str, folder: str):
        """
        Load the existing chat history for (username, folder).
        Useful when switching folders or reopening the app.
        """
        if not folder:
            return []

        try:
            state = self.session_service.load(username, folder)
        except Exception:
            return []

        gr_messages = []
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                gr_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gr_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict) and "role" in msg:
                gr_messages.append(
                    {"role": msg["role"], "content": msg.get("content", "")}
                )

        return gr_messages

    def clear_chat(self, username: str, folder: str):
        """
        Clear ONLY the chat for (username, folder).
        Does not affect chats for other folders.
        """
        try:
            if not folder:
                return []

            state = self.session_service.load(username, folder)
            state.messages = []
            self.session_service.save(username, folder, state)
            return []
        except Exception:
            return []

    # -------------------- Folder Management (GLOBAL per user) --------------------

    def get_folders(self, username: str):
        """
        Return all indexed folders for the given user.
        Stored in the user's GLOBAL session (username, __global__).
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            return getattr(state, "indexed_directories", [])
        except Exception:
            return []

    async def add_folder(self, username: str, folder: str):
        """
        Add a folder to the user's index list (GLOBAL)
        and ensure it is indexed via FolderService/Sidekick.
        """
        try:
            # GLOBAL state for that user
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)

            if not hasattr(state, "indexed_directories"):
                state.indexed_directories = []

            if folder not in state.indexed_directories:
                state.indexed_directories.append(folder)
                # Ensure the folder is indexed
                new_state = await self.folder_service.ensure_indexed(folder, state)
                # Save GLOBAL state
                self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder added", new_state.indexed_directories

            return "Folder already exists", state.indexed_directories
        except Exception as e:
            return f"Error: {e}", []

    async def remove_folder(self, username: str, folder: str):
        """
        Remove a folder from the user's index list (GLOBAL),
        and optionally clear that folder's index in the vectorstore.
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)

            if hasattr(state, "indexed_directories") and folder in state.indexed_directories:
                state.indexed_directories.remove(folder)
                # Optionally clear the vector index for that folder
                new_state = await self.folder_service.clear_folder(folder, state)
                self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
                return "Folder removed", new_state.indexed_directories

            return "Folder not found"
        except Exception as e:
            return f"Error: {e}"

    async def index_folder(self, username: str, folder: str):
        """
        Index the contents of a specific folder.
        This is a GLOBAL per-user operation, not tied to chat history.
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.ensure_indexed(folder, state)
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder indexed"
        except Exception as e:
            return f"Error: {e}"

    async def reindex_folder(self, username: str, folder: str):
        """
        Force reindexing of the selected folder.
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.reindex_folder(folder, state)
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder reindexed"
        except Exception as e:
            return f"Error: {e}"

    async def clear_folder(self, username: str, folder: str):
        """
        Clear the vector index for the selected folder,
        and update the user's GLOBAL state.
        """
        try:
            state = self.session_service.load(username, _GLOBAL_FOLDER_KEY)
            new_state = await self.folder_service.clear_folder(folder, state)
            self.session_service.save(username, _GLOBAL_FOLDER_KEY, new_state)
            return "Folder cleared"
        except Exception as e:
            return f"Error: {e}"
