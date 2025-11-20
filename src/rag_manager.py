# src/rag_manager.py

import os
import glob
import uuid
from datetime import datetime

from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, TextLoader, PythonLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
K = 8
EXCLUDED_DIRS = {".venv", "venv", "__pycache__"}


class RAGManager:
    """Handles document loading, indexing, vectorstores, retrievers and search."""

    def __init__(self, embedding):
        self.embedding = embedding
        self.retriever_registry = {}

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    def _is_excluded(self, path: str) -> bool:
        parts = path.replace("\\", "/").split("/")
        return any(ex in parts for ex in EXCLUDED_DIRS)

    def _load_documents(self, directory: str):
        docs = []

        # txt, md, py files
        loaders = [
            DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader),
            DirectoryLoader(directory, glob="**/*.py", loader_cls=PythonLoader),
        ]
        for loader in loaders:
            try:
                for doc in loader.load():
                    src = doc.metadata.get("source", "")
                    if not self._is_excluded(src):
                        docs.append(doc)
            except Exception as e:
                print(f"⚠️ Error loading {loader.__class__.__name__}: {e}")

        # PDF files
        pdf_files = glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True)
        for path in pdf_files:
            if self._is_excluded(path):
                continue
            try:
                docs.extend(PyPDFLoader(path).load())
            except Exception as e:
                print(f"⚠️ Error loading PDF {path}: {e}")

        return docs

    # ---------------------------------------------------
    # Indexing
    # ---------------------------------------------------
    def index_directory(self, directory: str, k: int = K) -> str:
        """Indexes a directory and registers its retriever."""
        if not os.path.exists(directory):
            return f"❌ Invalid directory: {directory}"

        docs = self._load_documents(directory)
        if not docs:
            return "❌ No readable documents found"

        # Metadata
        for doc in docs:
            src = doc.metadata.get("source", "")
            doc.metadata["file_name"] = os.path.basename(src)
            doc.metadata["file_path"] = src
            doc.metadata["indexed_at"] = datetime.now().isoformat()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        # Vectorstore
        persist_dir = f"vector_db/{os.path.basename(directory)}_{uuid.uuid4().hex[:8]}"
        os.makedirs(persist_dir, exist_ok=True)

        vectorstore = Chroma.from_documents(
            chunks,
            self.embedding,
            persist_directory=persist_dir
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": K})
        self.retriever_registry[directory] = retriever

        return f"✅ Indexed {len(chunks)} chunks from {len(docs)} documents in '{directory}'"

    # ---------------------------------------------------
    # Search
    # ---------------------------------------------------
    def search(self, query: str, k: int = 5) -> str:
        """Executes RAG search in the currently active directory."""
        if not self.retriever_registry:
            return "❌ No directory indexed. Use index_directory() first."

        active_dir = list(self.retriever_registry.keys())[0]
        retriever = self.retriever_registry[active_dir]

        try:
            docs = retriever.invoke(query)[:k]
        except Exception as e:
            return f"❌ Retrieval error: {e}"

        if not docs:
            return f"❌ No relevant documents found for '{query}'"

        formatted = []
        for i, doc in enumerate(docs, 1):
            fname = doc.metadata.get("file_name", "unknown")
            content = doc.page_content[:450]
            formatted.append(
                f"📄 **Result {i}** ({fname}):\n{content}\n"
            )

        return "\n---\n".join(formatted)
