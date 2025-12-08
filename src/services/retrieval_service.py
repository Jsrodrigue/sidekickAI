"""
Retrieval service (RAG).
Handles registration of retrievers and searches.
"""
from typing import Dict, Any, Optional


class RetrievalService:
    """Service to handle retrievers and perform RAG searches."""

    def __init__(self):
        # Map folder_path -> retriever
        self.retriever_registry: Dict[str, Any] = {}
        # Map folder_path -> persist_dir
        self.vectorstore_paths: Dict[str, str] = {}
        # Currently active folder (set from Sidekick.run based on UI selection)
        self.current_folder: Optional[str] = None
        # Default number of retrive documents
        self.default_k: int = 5


    def register_retriever(self, folder_path: str, retriever: Any, persist_dir: str):
        """Register a retriever for a folder."""
        self.retriever_registry[folder_path] = retriever
        self.vectorstore_paths[folder_path] = persist_dir

    def unregister_retriever(self, folder_path: str) -> Optional[str]:
        """Delete a retriever from the registry and return its persist_dir."""
        self.retriever_registry.pop(folder_path, None)
        return self.vectorstore_paths.pop(folder_path, None)

    def has_retriever(self, folder_path: str) -> bool:
        """Check if there is a retriever for a folder."""
        return folder_path in self.retriever_registry

    def set_current_folder(self, folder_path: Optional[str]):
        """Set the currently active folder for retrieval."""
        self.current_folder = folder_path

    def get_retriever(self, folder_path: Optional[str] = None) -> Optional[Any]:
        """
        Get a retriever.

        Priority:
        1) Explicit folder_path argument
        2) self.current_folder (set from UI)
        3) First registered retriever
        """
        # 1) explicit argument overrides everything
        if folder_path:
            return self.retriever_registry.get(folder_path)

        # 2) use active folder if set
        if self.current_folder and self.current_folder in self.retriever_registry:
            return self.retriever_registry[self.current_folder]

        # 3) fallback: first registered retriever
        if self.retriever_registry:
            return next(iter(self.retriever_registry.values()))

        return None

    def search(self, query: str, k: int = 5, folder_path: Optional[str] = None) -> str:
        """Execute a RAG search."""
        if not self.retriever_registry:
            return "❌ No indexed folder. Index a folder first."

        retriever = self.get_retriever(folder_path)
        if not retriever:
            return f"❌ No retriever found for folder: {folder_path or self.current_folder}"

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
            print(f"[ERROR] Search error: {e}")
            return f"❌ Search error: {e}"

    def clear(self):
        """Clean all retrievers."""
        self.retriever_registry.clear()
        self.vectorstore_paths.clear()
        self.current_folder = None

    def get_indexed_folders(self) -> list:
        """Returns list of indexed folders."""
        return list(self.retriever_registry.keys())
