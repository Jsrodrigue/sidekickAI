"""
Manejadores de eventos de la interfaz Gradio.
Conecta la UI con los servicios del backend.
"""
import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from src.repositories.folder_repository import FolderRepository
from src.repositories.session_repository import SessionRepository
from src.core.state import SidekickState


class UIHandlers:
    """Manejadores de eventos para la interfaz Gradio."""

    def __init__(self, sidekick):
        self.sidekick = sidekick
        self.sessions = {}  # Cache en memoria de sesiones activas

    # -------------------- Gestión de carpetas --------------------

    def add_folder(self, username: str, folder_path: str):
        """Añade una carpeta a la base de datos."""
        success, message, folders = FolderRepository.add(username, folder_path)
        return message, folders

    async def remove_folder(self, username: str, folder_path: str):
        """Elimina una carpeta de la DB y limpia su vectorstore."""
        success, message, folders = FolderRepository.remove(username, folder_path)
        
        # Limpiar vectorstore si existe
        try:
            await asyncio.to_thread(self.sidekick.remove_directory, folder_path)
        except Exception as e:
            print(f"⚠️ Error cleaning vectorstore: {e}")
        
        return message, folders

    def get_folders(self, username: str):
        """Obtiene todas las carpetas de un usuario."""
        return FolderRepository.get_all(username)

    # -------------------- Indexación --------------------

    async def index_folder(self, username: str, folder_path: str, force_reindex: bool = False):
        """Indexa una carpeta de forma asíncrona."""
        if not folder_path:
            return "❌ Select a folder first"

        # Verificar si ya está indexada
        if (self.sidekick.retrieval_service.has_retriever(folder_path) 
            and not force_reindex):
            return f"✅ Folder already indexed: {folder_path}"

        # Indexar en thread separado
        try:
            result = await asyncio.to_thread(
                self.sidekick.index_directory,
                folder_path,
                force_reindex
            )
        except Exception as e:
            return f"❌ Indexing failed: {e}"

        # Actualizar sesión
        if username in self.sessions:
            state = self.sessions[username]
            state.current_directory = folder_path
            if folder_path not in state.indexed_directories:
                state.indexed_directories.append(folder_path)
            SessionRepository.save(username, state)

        return result

    async def reindex_folder(self, username: str, folder_path: str):
        """Fuerza reindexación de una carpeta."""
        return await self.index_folder(username, folder_path, force_reindex=True)

    # -------------------- Chat --------------------

    async def chat(self, username: str, folder_path: str, message: str):
        """Procesa un mensaje del usuario."""
        if not username:
            return [], "Please log in first"

        message = (message or "").strip()
        if not message:
            return [], ""

        # Obtener o cargar sesión
        if username not in self.sessions:
            self.sessions[username] = SessionRepository.load(username)
        
        state = self.sessions[username]

        # Cambiar carpeta activa si es necesario
        if folder_path and folder_path != state.current_directory:
            state.current_directory = folder_path
            
            # Indexar si no está indexada
            if not self.sidekick.retrieval_service.has_retriever(folder_path):
                await asyncio.to_thread(self.sidekick.index_directory, folder_path)
            
            if folder_path not in state.indexed_directories:
                state.indexed_directories.append(folder_path)

        # Añadir mensaje del usuario
        state.messages.append(HumanMessage(content=message))

        # Establecer criterio de éxito si no existe
        if not state.success_criteria:
            state.success_criteria = "Answer the user's question completely and accurately."

        # Invocar grafo LangGraph
        try:
            result_state = await self.sidekick.graph.ainvoke(
                state,
                {"configurable": {"thread_id": state.thread_id}}
            )

            # Actualizar estado
            if isinstance(result_state, dict):
                state.messages = result_state.get("messages", state.messages)
                state.evaluation_history = result_state.get(
                    "evaluation_history",
                    state.evaluation_history
                )
                state.criteria_met = result_state.get("criteria_met", state.criteria_met)
                state.needs_user_input = result_state.get(
                    "needs_user_input",
                    state.needs_user_input
                )
            else:
                state = result_state

            # Guardar sesión
            self.sessions[username] = state
            SessionRepository.save(username, state)

            # Construir historial para Gradio
            chat_history = self._build_chat_history(state)
            return chat_history, ""

        except Exception as e:
            # Manejar error
            state.messages.append(AIMessage(content=f"Error processing request: {e}"))
            self.sessions[username] = state
            SessionRepository.save(username, state)
            
            chat_history = self._build_chat_history(state)
            return chat_history, ""

    def clear_chat(self, username: str):
        """Limpia el historial de chat."""
        if username in self.sessions:
            self.sessions[username].messages = []
            SessionRepository.save(username, self.sessions[username])
            return []
        return []

    # -------------------- Utilidades --------------------

    def _build_chat_history(self, state: SidekickState) -> list:
        """Construye historial de chat para Gradio."""
        chat_history = []
        for m in state.messages:
            if isinstance(m, HumanMessage):
                chat_history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                # Filtrar mensajes de evaluación
                if not getattr(m, "content", "").startswith("💭"):
                    chat_history.append({"role": "assistant", "content": m.content})
        return chat_history