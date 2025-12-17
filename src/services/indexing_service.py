"""
Index service.
Handles loading, processing and creation of vectorstore.
"""

import glob
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.path_utils import is_excluded_path, normalize_path


class IndexingService:
    """Service to index folders/files and create/load persistent vectorstores."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".pdf"}

    def __init__(self, embeddings: OpenAIEmbeddings, vectorstore_root: str):
        self.embeddings = embeddings
        self.vectorstore_root = vectorstore_root
        os.makedirs(self.vectorstore_root, exist_ok=True)

    # -------------------- Document loading --------------------

    def load_documents(
        self,
        path: str,
        excluded_dirs: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List:
        """
        Load documents from:
        - a directory (scan by extension)
        - OR a single file path (load only that file)

        Backward compatible: calling with a directory behaves like before.
        """
        excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
        path = normalize_path(path)

        if os.path.isdir(path):
            return self._load_documents_from_directory(
                directory=path, excluded_dirs=excluded_dirs, recursive=recursive
            )

        if os.path.isfile(path):
            return self._load_documents_from_file(path, excluded_dirs)

        print(f"[WARNING] Path not found or invalid: {path}")
        return []

    def load_documents_from_paths(
        self,
        paths: List[str],
        excluded_dirs: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List:
        """
        Load documents from a list of paths that can include BOTH directories and files.
        - Directories are scanned
        - Files are loaded directly
        """
        excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
        docs: List = []

        for raw in paths:
            if not raw:
                continue
            p = normalize_path(raw)

            if os.path.isdir(p):
                docs.extend(
                    self._load_documents_from_directory(
                        directory=p, excluded_dirs=excluded_dirs, recursive=recursive
                    )
                )
            elif os.path.isfile(p):
                docs.extend(self._load_documents_from_file(p, excluded_dirs))
            else:
                print(f"[WARNING] Path not found or invalid: {p}")

        return docs

    def _load_documents_from_directory(
        self, directory: str, excluded_dirs: List[str], recursive: bool = True
    ) -> List:
        """Load all supported documents from a directory (optionally recursive)."""
        docs: List = []

        # Preserve your original behavior (recursive scan)
        pattern = "**" if recursive else "*"

        # Load Markdown files
        docs.extend(self._load_markdown_files(directory, excluded_dirs, pattern))
        # Load text files
        docs.extend(self._load_text_files(directory, excluded_dirs, pattern))
        # Load Python files
        docs.extend(self._load_python_files(directory, excluded_dirs, pattern))
        # Load PDFs
        docs.extend(self._load_pdf_files(directory, excluded_dirs, pattern))

        return docs

    def _load_documents_from_file(self, file_path: str, excluded_dirs: List[str]) -> List:
        """Load a single supported file by extension."""
        docs: List = []
        file_path = normalize_path(file_path)

        if is_excluded_path(file_path, excluded_dirs):
            return docs

        if not os.path.isfile(file_path):
            print(f"[WARNING] File not found: {file_path}")
            return docs

        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"[INFO] Unsupported extension (skipped): {file_path}")
            return docs

        try:
            if ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
                docs.extend(loader.load())

            elif ext == ".txt":
                # keep your original encoding fallback behavior
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
                except UnicodeDecodeError:
                    loader = TextLoader(file_path, encoding="latin-1")
                    docs.extend(loader.load())

            elif ext == ".py":
                loader = PythonLoader(file_path)
                docs.extend(loader.load())

            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] Failed to load file {file_path}: {e}")

        return docs

    def _load_markdown_files(self, directory: str, excluded_dirs: List[str], pattern: str) -> List:
        """Load markdown files."""
        docs = []
        try:
            md_paths = glob.glob(os.path.join(directory, pattern, "*.md"), recursive=True)
            for md_path in md_paths:
                if is_excluded_path(md_path, excluded_dirs):
                    continue
                md_path = normalize_path(md_path)
                if not os.path.isfile(md_path):
                    print(f"[WARNING] MD file not found: {md_path}")
                    continue
                try:
                    loader = UnstructuredMarkdownLoader(md_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[ERROR] Error loading MD {md_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning MD files: {e}")
        print(f"[INFO] Loaded {len(docs)} Markdown files")
        return docs

    def _load_text_files(self, directory: str, excluded_dirs: List[str], pattern: str) -> List:
        """Load text files."""
        docs = []
        try:
            txt_paths = glob.glob(os.path.join(directory, pattern, "*.txt"), recursive=True)
            for p in txt_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                p = normalize_path(p)
                try:
                    loader = TextLoader(p, encoding="utf-8")
                    docs.extend(loader.load())
                except UnicodeDecodeError:
                    try:
                        loader = TextLoader(p, encoding="latin-1")
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"[ERROR] Failed to load text {p}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning txt files: {e}")

        print(f"[INFO] Loaded {len(docs)} text files")
        return docs

    def _load_python_files(self, directory: str, excluded_dirs: List[str], pattern: str) -> List:
        """Load python files."""
        docs = []
        try:
            py_paths = glob.glob(os.path.join(directory, pattern, "*.py"), recursive=True)
            for p in py_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                p = normalize_path(p)
                try:
                    loader = PythonLoader(p)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[ERROR] Failed to load python file {p}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning py files: {e}")

        print(f"[INFO] Loaded {len(docs)} Python files")
        return docs

    def _load_pdf_files(self, directory: str, excluded_dirs: List[str], pattern: str) -> List:
        """Load PDF files."""
        docs = []
        try:
            pdf_paths = glob.glob(os.path.join(directory, pattern, "*.pdf"), recursive=True)
            for p in pdf_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                p = normalize_path(p)
                try:
                    loader = PyPDFLoader(p)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[ERROR] Error loading PDF {p}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning PDFs: {e}")

        print(f"[INFO] Loaded {len(docs)} PDF files")
        return docs

    # -------------------- Preprocessing --------------------

    def normalize_document_metadata(self, docs: List) -> List:
        """Normalize and validate files metadata."""
        valid_docs = []
        for doc in docs:
            src = normalize_path(doc.metadata.get("source", ""))
            if not src or not os.path.exists(src):
                print(f"[WARNING] Document has invalid source: {src}")
                doc.metadata["file_name"] = doc.metadata.get("file_name") or "unknown"
                doc.metadata["file_path"] = src
            else:
                doc.metadata["file_name"] = os.path.basename(src)
                doc.metadata["file_path"] = src
            doc.metadata["indexed_at"] = datetime.now().isoformat()
            valid_docs.append(doc)
        return valid_docs

    def chunk_documents(self, docs: List, chunk_size: int = 600, chunk_overlap: int = 20) -> List:
        """Divide documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    # -------------------- Vectorstore creation & loading --------------------

    def create_vectorstore(
        self, chunks: List, directory_name: str
    ) -> Tuple[Optional[Chroma], Optional[str]]:
        """
        Create a persistent Chroma vectorstore.
        directory_name is used only to create a readable persist folder name.
        """
        persist_dir = os.path.join(
            self.vectorstore_root,
            f"{os.path.basename(directory_name)}_{uuid.uuid4().hex[:8]}",
        )

        try:
            os.makedirs(persist_dir, exist_ok=True)
            vectorstore = Chroma.from_documents(
                chunks, self.embeddings, persist_directory=persist_dir
            )
            return vectorstore, persist_dir
        except Exception as e:
            print(f"[ERROR] Failed to create vectorstore: {e}")
            # Cleanup in case of error
            try:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
            except Exception:
                pass
            return None, None

    def load_vectorstore(self, persist_dir: str) -> Optional[Chroma]:
        """
        Load an existing Chroma vectorstore from disk.

        Used at startup to restore previously indexed folders/files
        without re-indexing documents.
        """
        try:
            if not os.path.exists(persist_dir):
                print(f"[WARN] Persist directory does not exist: {persist_dir}")
                return None

            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            return vectorstore
        except Exception as e:
            print(f"[ERROR] Failed to load vectorstore from {persist_dir}: {e}")
            return None

    def remove_vectorstore(self, persist_dir: str) -> bool:
        """Delete a vector store from disk."""
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                return True
        except Exception as e:
            print(f"[ERROR] Failed to remove vectorstore {persist_dir}: {e}")
        return False
