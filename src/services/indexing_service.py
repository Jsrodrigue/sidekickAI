"""
Servicio de indexación de documentos.
Maneja la carga, procesamiento y creación de vectorstores.
"""
import os
import glob
import uuid
import shutil
from typing import List, Optional, Tuple
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.path_utils import normalize_path, is_excluded_path


class IndexingService:
    """Servicio para indexar carpetas y crear vectorstores."""

    def __init__(self, embeddings: OpenAIEmbeddings, vectorstore_root: str):
        self.embeddings = embeddings
        self.vectorstore_root = vectorstore_root
        os.makedirs(self.vectorstore_root, exist_ok=True)

    def load_documents(self, directory: str, excluded_dirs: Optional[List[str]] = None) -> List:
        """Carga todos los documentos de un directorio."""
        excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
        docs = []

        # Cargar archivos Markdown
        docs.extend(self._load_markdown_files(directory, excluded_dirs))
        
        # Cargar archivos de texto
        docs.extend(self._load_text_files(directory, excluded_dirs))
        
        # Cargar archivos Python
        docs.extend(self._load_python_files(directory, excluded_dirs))
        
        # Cargar PDFs
        docs.extend(self._load_pdf_files(directory, excluded_dirs))

        return docs

    def _load_markdown_files(self, directory: str, excluded_dirs: List[str]) -> List:
        """Carga archivos Markdown."""
        docs = []
        try:
            md_paths = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
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
                    print(f"📘 MD loaded: {md_path}")
                except Exception as e:
                    print(f"[ERROR] Error loading MD {md_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning MD files: {e}")
        return docs

    def _load_text_files(self, directory: str, excluded_dirs: List[str]) -> List:
        """Carga archivos de texto."""
        docs = []
        try:
            txt_paths = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)
            for p in txt_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                try:
                    loader = TextLoader(p, encoding="utf-8")
                    docs.extend(loader.load())
                except UnicodeDecodeError:
                    try:
                        loader = TextLoader(p, encoding="latin-1")
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"⚠️ Failed to load text {p}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning txt files: {e}")
        return docs

    def _load_python_files(self, directory: str, excluded_dirs: List[str]) -> List:
        """Carga archivos Python."""
        docs = []
        try:
            py_paths = glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)
            for p in py_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                try:
                    loader = PythonLoader(p)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Failed to load python file {p}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning py files: {e}")
        return docs

    def _load_pdf_files(self, directory: str, excluded_dirs: List[str]) -> List:
        """Carga archivos PDF."""
        docs = []
        try:
            pdf_paths = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
            for p in pdf_paths:
                if is_excluded_path(p, excluded_dirs):
                    continue
                try:
                    loader = PyPDFLoader(p)
                    docs.extend(loader.load())
                    print(f"✅ PDF loaded: {p}")
                except Exception as e:
                    print(f"⚠️ Error loading PDF {p}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning PDFs: {e}")
        return docs

    def normalize_document_metadata(self, docs: List) -> List:
        """Normaliza y valida metadatos de documentos."""
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
        """Divide documentos en chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    def create_vectorstore(self, chunks: List, directory_name: str) -> Tuple[Optional[Chroma], Optional[str]]:
        """Crea un vectorstore persistente."""
        persist_dir = os.path.join(
            self.vectorstore_root,
            f"{os.path.basename(directory_name)}_{uuid.uuid4().hex[:8]}"
        )
        
        try:
            os.makedirs(persist_dir, exist_ok=True)
            vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=persist_dir
            )
            return vectorstore, persist_dir
        except Exception as e:
            print(f"[ERROR] Failed to create vectorstore: {e}")
            # Cleanup en caso de error
            try:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
            except Exception:
                pass
            return None, None

    def remove_vectorstore(self, persist_dir: str) -> bool:
        """Elimina un vectorstore del disco."""
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                return True
        except Exception as e:
            print(f"⚠️ Failed to remove vectorstore {persist_dir}: {e}")
        return False