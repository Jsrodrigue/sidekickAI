"""
Servicio de recuperación (RAG).
Maneja el registro de retrievers y consultas de búsqueda.
"""
from typing import Dict, Any, Optional


class RetrievalService:
    """Servicio para gestionar retrievers y ejecutar búsquedas RAG."""

    def __init__(self):
        # Mapeo de folder_path -> retriever
        self.retriever_registry: Dict[str, Any] = {}
        # Mapeo de folder_path -> persist_dir
        self.vectorstore_paths: Dict[str, str] = {}

    def register_retriever(self, folder_path: str, retriever: Any, persist_dir: str):
        """Registra un retriever para una carpeta."""
        self.retriever_registry[folder_path] = retriever
        self.vectorstore_paths[folder_path] = persist_dir

    def unregister_retriever(self, folder_path: str) -> Optional[str]:
        """Elimina un retriever del registro y devuelve su persist_dir."""
        self.retriever_registry.pop(folder_path, None)
        return self.vectorstore_paths.pop(folder_path, None)

    def has_retriever(self, folder_path: str) -> bool:
        """Verifica si existe un retriever para la carpeta."""
        return folder_path in self.retriever_registry

    def get_retriever(self, folder_path: Optional[str] = None) -> Optional[Any]:
        """Obtiene un retriever. Si no se especifica carpeta, devuelve el primero."""
        if folder_path:
            return self.retriever_registry.get(folder_path)
        
        # Devolver el primer retriever disponible
        if self.retriever_registry:
            return next(iter(self.retriever_registry.values()))
        return None

    def search(self, query: str, k: int = 5, folder_path: Optional[str] = None) -> str:
        """Ejecuta una búsqueda RAG."""
        if not self.retriever_registry:
            return "❌ No indexed folder. Index a folder first."

        retriever = self.get_retriever(folder_path)
        if not retriever:
            return f"❌ No retriever found for folder: {folder_path}"

        try:
            docs = retriever.invoke(query)[:k]
            if not docs:
                return f"❌ No relevant documents for: '{query}'"

            results = []
            for i, doc in enumerate(docs, 1):
                fname = doc.metadata.get("file_name", "unknown")
                content = getattr(doc, "page_content", "")[:500]
                results.append(f"📄 Doc {i} ({fname}):\n{content}\n")
            
            return "\n---\n".join(results)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return f"❌ Search error: {e}"

    def clear(self):
        """Limpia todos los retrievers."""
        self.retriever_registry.clear()
        self.vectorstore_paths.clear()

    def get_indexed_folders(self) -> list:
        """Devuelve lista de carpetas indexadas."""
        return list(self.retriever_registry.keys())