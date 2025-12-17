"""
Retrieval service (RAG).
Handles registration of retrievers and searches.
"""

from typing import Dict, Any, Optional


class RetrievalService:
    """Service to handle retrievers and perform RAG searches."""

    def __init__(self):
        # Map index_key -> retriever
        self.retriever_registry: Dict[str, Any] = {}
        # Map index_key -> persist_dir
        self.vectorstore_paths: Dict[str, str] = {}
        # Map index_key -> vectorstore (Chroma) to release handles on Windows
        self.vectorstores: Dict[str, Any] = {}

        self.current_folder: Optional[str] = None
        self.default_k: int = 5

    def register_retriever(
        self,
        folder_path: str,
        retriever: Any,
        persist_dir: str,
        vectorstore: Any = None,
    ):
        """Register a retriever for a folder (and optionally its vectorstore)."""
        self.retriever_registry[folder_path] = retriever
        self.vectorstore_paths[folder_path] = persist_dir
        if vectorstore is not None:
            self.vectorstores[folder_path] = vectorstore

    def pop_vectorstore(self, folder_path: str) -> Any:
        """Pop and return the vectorstore for this key (if any)."""
        return self.vectorstores.pop(folder_path, None)

    def unregister_retriever(self, folder_path: str) -> Optional[str]:
        """
        Delete a retriever from the registry and return its persist_dir.
        NOTE: does NOT return vectorstore; use pop_vectorstore() for that.
        """
        self.retriever_registry.pop(folder_path, None)
        return self.vectorstore_paths.pop(folder_path, None)

    def has_retriever(self, folder_path: str) -> bool:
        return folder_path in self.retriever_registry

    def set_current_folder(self, folder_path: Optional[str]):
        self.current_folder = folder_path

    def get_retriever(self, folder_path: Optional[str] = None) -> Optional[Any]:
        if folder_path:
            return self.retriever_registry.get(folder_path)

        if self.current_folder and self.current_folder in self.retriever_registry:
            return self.retriever_registry[self.current_folder]

        if self.retriever_registry:
            return next(iter(self.retriever_registry.values()))

        return None

    def search(self, query: str, k: int = 5, folder_path: Optional[str] = None) -> str:
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
        self.retriever_registry.clear()
        self.vectorstore_paths.clear()
        self.vectorstores.clear()
        self.current_folder = None

    def get_indexed_folders(self) -> list:
        return list(self.retriever_registry.keys())
