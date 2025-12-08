"""
Main Sidekick class.
Coordinates indexing, retrieval services, and the LangGraph pipeline.
"""

import json
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from src.core.graph import GraphBuilder
from src.core.state import SidekickState
from src.models.output_models import EvaluatorOutput
from src.services.indexing_service import IndexingService
from src.services.retrieval_service import RetrievalService
from src.tools import build_all_tools
from src.utils.path_utils import (
    get_absolute_path,
    normalize_path,
    validate_directory,
)

load_dotenv(override=True)

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "vector_db")
INDEX_MANIFEST = os.path.join(VECTORSTORE_ROOT, "index_manifest.json")
SEARCH_K=15

class Sidekick:
    """
    Main coordinator for the Sidekick system.

    Responsibilities:
    - Coordinate indexing and retrieval services
    - Manage the LangGraph execution graph
    - Provide a high-level interface for system operations
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sidekick_id = str(uuid.uuid4())

        # LLMs
        self.worker_llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0,
        )
        self.evaluator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0,
        )

        # Services
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.indexing_service = IndexingService(embeddings, VECTORSTORE_ROOT)
        self.retrieval_service = RetrievalService()

        # Memory and graph
        self.memory = MemorySaver()
        self.graph = None
        self.tools = []

        # Restore previously indexed folders (Chroma) from disk
        self._bootstrap_retrievers_from_manifest()

    # ---------- Index manifest helpers ----------

    def _load_index_manifest(self) -> dict:
        """Load the mapping folder_key -> persist_dir from disk."""
        if not os.path.exists(VECTORSTORE_ROOT):
            os.makedirs(VECTORSTORE_ROOT, exist_ok=True)

        if not os.path.exists(INDEX_MANIFEST):
            return {}

        try:
            with open(INDEX_MANIFEST, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
        except Exception as e:
            print(f"[WARN] Could not load index manifest: {e}")
            return {}

    def _save_index_manifest(self, manifest: dict) -> None:
        """Persist the mapping folder_key -> persist_dir to disk."""
        os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
        try:
            with open(INDEX_MANIFEST, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save index manifest: {e}")

    def _bootstrap_retrievers_from_manifest(self):
        """
        On startup, load existing vectorstores from disk and register retrievers,
        so we don't need to reindex every time.
        """
        manifest = self._load_index_manifest()
        if not manifest:
            print("[INFO] No existing indexes found in manifest.")
            return

        print(f"[INFO] Bootstrapping {len(manifest)} indexes from manifest...")

        for folder_key, persist_dir in manifest.items():
            try:
                vectorstore = self.indexing_service.load_vectorstore(persist_dir)
                if not vectorstore:
                    print(f"[WARN] Could not load vectorstore at {persist_dir}")
                    continue

                retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
                self.retrieval_service.register_retriever(
                    folder_key, retriever, persist_dir
                )
                print(f"[OK] Restored retriever for {folder_key} from {persist_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to restore index for {folder_key}: {e}")

    # -------------------- Setup (LangGraph + Tools) --------------------

    async def setup(self):
        """Initializes tools and builds the LangGraph pipeline."""

        # 1) Build all tools (RAG + files + web + etc.)
        self.tools = build_all_tools(self.retrieval_service)

        # 2) Bind tools and structured outputs
        self.worker_llm_with_tools = self.worker_llm.bind_tools(self.tools)
        self.evaluator_llm_with_output = self.evaluator_llm.with_structured_output(
            EvaluatorOutput
        )

        # 3) Build LangGraph
        graph_builder = GraphBuilder(
            worker_llm=self.worker_llm_with_tools,
            evaluator_llm=self.evaluator_llm_with_output,
            tools=self.tools,
            memory=self.memory,
        )
        self.graph = await graph_builder.build()

    # -------------------- Indexing Operations --------------------

    def index_directory(
        self,
        directory: str,
        force_reindex: bool = False,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
    ) -> str:
        """
        Synchronously indexes a directory.
        Should be executed inside asyncio.to_thread to avoid blocking.

        chunk_size and chunk_overlap control how documents are split
        during indexing.
        """
        directory = normalize_path(directory)

        if not validate_directory(directory):
            return f"[ERROR] Invalid directory: {directory}"

        folder_key = get_absolute_path(directory)

        # If already indexed and no force reindexing → skip
        if self.retrieval_service.has_retriever(folder_key) and not force_reindex:
            return f"[INFO] Already indexed: {directory}"

        # If force reindexing → clear old data
        if force_reindex:
            persist_dir = self.retrieval_service.unregister_retriever(folder_key)
            if persist_dir:
                self.indexing_service.remove_vectorstore(persist_dir)

        # Load documents
        docs = self.indexing_service.load_documents(directory)
        if not docs:
            return "❌ No readable documents found"

        # Normalize metadata
        valid_docs = self.indexing_service.normalize_document_metadata(docs)

        # Chunk into embeddings
        try:
            chunks = self.indexing_service.chunk_documents(
                valid_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            return f"[ERROR] Error during chunk splitting: {e}"

        # Create vectorstore
        vectorstore, persist_dir = self.indexing_service.create_vectorstore(
            chunks, directory
        )

        if not vectorstore or not persist_dir:
            return "[ERROR] Failed to create vectorstore"

        # Register retriever and update manifest
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
            self.retrieval_service.register_retriever(
                folder_key, retriever, persist_dir
            )

            # Update manifest so index persists across restarts
            manifest = self._load_index_manifest()
            manifest[folder_key] = persist_dir
            self._save_index_manifest(manifest)

        except Exception as e:
            return f"❌ Failed to register retriever: {e}"

        return f"✅ Indexed {len(chunks)} chunks from {len(valid_docs)} documents"

    def remove_directory(self, directory: str) -> str:
        """Removes an indexed directory and deletes its vectorstore."""
        directory = normalize_path(directory)
        folder_key = get_absolute_path(directory)

        persist_dir = self.retrieval_service.unregister_retriever(folder_key)
        if persist_dir:
            self.indexing_service.remove_vectorstore(persist_dir)

        # Remove from manifest
        manifest = self._load_index_manifest()
        if folder_key in manifest:
            manifest.pop(folder_key, None)
            self._save_index_manifest(manifest)

        return f"🗑️ Removed folder index: {directory}"

    # -------------------- Execution --------------------

    async def run(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        folder: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Runs a user query through the LangGraph pipeline (with debug).

        top_k (if provided) sets the default number of documents to retrieve
        for RAG via retrieval_service.default_k.
        """

        if not self.graph:
            await self.setup()

        # -------------------------
        # DEBUG: FOLDER SELECTION
        # -------------------------
        print("\n========== SIDEKICK RUN DEBUG ==========")
        print(f"User input: {user_input!r}")
        print(f"Folder passed in: {folder!r}")

        # Set active folder (RAG)
        if folder:
            folder_key = get_absolute_path(folder)
            self.retrieval_service.set_current_folder(folder_key)
            print(f">>> Active RAG folder set to: {folder_key}")
        else:
            self.retrieval_service.set_current_folder(None)
            print(">>> Active RAG folder set to: NONE")

        # Apply retrieval top_k if provided
        if top_k is not None and top_k > 0:
            print(f">>> Setting retrieval default_k to: {top_k}")
            self.retrieval_service.default_k = top_k

        # DEBUG: what retriever is being used?
        active_retriever = self.retrieval_service.get_retriever(
            self.retrieval_service.current_folder
        )
        print(f">>> Retriever selected: {active_retriever}")
        print("=========================================\n")

        # Thread separation for memory
        thread_id = thread_id or self.sidekick_id
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = SidekickState(
            messages=[HumanMessage(content=user_input)],
            success_criteria="Answer fully",
        )

        result = await self.graph.ainvoke(initial_state, config)

        # -------------------------
        # DEBUG: LANGGRAPH OUTPUT
        # -------------------------
        print("\n======= LANGGRAPH OUTPUT MESSAGES =======")
        for i, msg in enumerate(result["messages"]):
            print(f"[{i}] {type(msg).__name__}: {getattr(msg, 'content', None)!r}")
        print("=========================================\n")

        # Return last assistant answer
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("💭"):
                return msg.content

        return "No response generated"

    def cleanup(self):
        """Cleans up resources (vectorstores and in-memory retrievers)."""
        for persist_dir in self.retrieval_service.vectorstore_paths.values():
            self.indexing_service.remove_vectorstore(persist_dir)

        self.retrieval_service.clear()
        print("🧹 Resources cleaned")


async def init_sidekick() -> Sidekick:
    """Creates and initializes a fully configured Sidekick instance."""
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick
