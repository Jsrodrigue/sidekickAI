"""
Main Sidekick class.
Coordinates indexing, retrieval services, and the LangGraph pipeline.
"""

import gc
import json
import os
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from src.core.graph import GraphBuilder
from src.core.state import SidekickState
from src.services.indexing_service import IndexingService
from src.services.retrieval_service import RetrievalService
from src.tools import build_all_tools
from src.utils.path_utils import get_absolute_path, normalize_path, validate_directory
from src.utils.fs_utils import delete_dir_verified

load_dotenv(override=True)

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "vector_db")
INDEX_MANIFEST = os.path.join(VECTORSTORE_ROOT, "index_manifest.json")
SEARCH_K = 15


def _is_file(path: str) -> bool:
    return os.path.isfile(path)


def _is_dir(path: str) -> bool:
    return os.path.isdir(path)


def _make_index_key(path: str) -> str:
    """
    Stable key:
    - directories: absolute path
    - files: FILE::<absolute_path>
    """
    abs_path = get_absolute_path(path)
    if os.path.isfile(abs_path):
        return f"FILE::{abs_path}"
    return abs_path


class Sidekick:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sidekick_id = str(uuid.uuid4())

        self.worker_llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0,
        )

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.indexing_service = IndexingService(embeddings, VECTORSTORE_ROOT)
        self.retrieval_service = RetrievalService()

        self.memory = MemorySaver()
        self.graph = None

        self.all_tools = []
        self.tools = []

        self._bootstrap_retrievers_from_manifest()

    # ---------- Manifest ----------

    def _load_index_manifest(self) -> dict:
        if not os.path.exists(VECTORSTORE_ROOT):
            os.makedirs(VECTORSTORE_ROOT, exist_ok=True)

        if not os.path.exists(INDEX_MANIFEST):
            return {}

        try:
            with open(INDEX_MANIFEST, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[WARN] Could not load index manifest: {e}")
            return {}

    def _save_index_manifest(self, manifest: dict) -> None:
        os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
        try:
            with open(INDEX_MANIFEST, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save index manifest: {e}")

    def _bootstrap_retrievers_from_manifest(self):
        manifest = self._load_index_manifest()
        if not manifest:
            print("[INFO] No existing indexes found in manifest.")
            return

        print(f"[INFO] Bootstrapping {len(manifest)} indexes from manifest...")

        for index_key, persist_dir in manifest.items():
            try:
                vectorstore = self.indexing_service.load_vectorstore(persist_dir)
                if not vectorstore:
                    print(f"[WARN] Could not load vectorstore at {persist_dir}")
                    continue

                retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
                self.retrieval_service.register_retriever(
                    index_key,
                    retriever,
                    persist_dir,
                    vectorstore=vectorstore,
                )
                print(f"[OK] Restored retriever for {index_key} from {persist_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to restore index for {index_key}: {e}")

    # ---------- Chroma handle release (critical on Windows) ----------

    def _close_vectorstore_best_effort(self, vectorstore) -> None:
        """
        Best-effort close to release file handles on Windows.

        Tries a few common internal structures used by Chroma/chromadb.
        Safe: wrapped in try/except so it won't break other flows.
        """
        if vectorstore is None:
            return

        # Try to stop underlying chromadb system if present
        try:
            client = getattr(vectorstore, "_client", None) or getattr(vectorstore, "client", None)
            system = getattr(client, "_system", None)
            if system is not None and hasattr(system, "stop"):
                system.stop()
        except Exception as e:
            print("[DEBUG] close_vectorstore: client/system stop failed:", e)

        # Drop references (most important part)
        try:
            del vectorstore
        except Exception:
            pass

        gc.collect()
        time.sleep(0.35)

    # -------------------- Setup --------------------

    async def _build_graph_with_tools(self, tools):
        worker_llm_with_tools = self.worker_llm.bind_tools(tools)
        graph_builder = GraphBuilder(
            worker_llm=worker_llm_with_tools,
            tools=tools,
            memory=self.memory,
        )
        self.graph = await graph_builder.build()
        self.tools = tools

    async def setup(self):
        self.all_tools = build_all_tools(self.retrieval_service)
        await self._build_graph_with_tools(self.all_tools)

    # -------------------- Indexing --------------------

    def index_path(
        self,
        path: str,
        force_reindex: bool = False,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
        recursive: bool = True,
    ) -> str:
        path = normalize_path(path)

        if _is_dir(path):
            if not validate_directory(path):
                return f"[ERROR] Invalid directory: {path}"
        elif _is_file(path):
            pass
        else:
            return f"[ERROR] Invalid path (not found): {path}"

        index_key = _make_index_key(path)

        if self.retrieval_service.has_retriever(index_key) and not force_reindex:
            return f"[INFO] Already indexed: {path}"

        # Force reindex: unregister + close + delete old
        if force_reindex:
            old_vs = self.retrieval_service.pop_vectorstore(index_key)
            old_persist = self.retrieval_service.unregister_retriever(index_key)
            self._close_vectorstore_best_effort(old_vs)
            if old_persist:
                self.indexing_service.remove_vectorstore(old_persist)

        docs = self.indexing_service.load_documents(path, recursive=recursive)
        if not docs:
            return "❌ No readable documents found"

        valid_docs = self.indexing_service.normalize_document_metadata(docs)

        try:
            chunks = self.indexing_service.chunk_documents(
                valid_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            return f"[ERROR] Error during chunk splitting: {e}"

        vectorstore, persist_dir = self.indexing_service.create_vectorstore(chunks, directory_name=path)
        if not vectorstore or not persist_dir:
            return "[ERROR] Failed to create vectorstore"

        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
            self.retrieval_service.register_retriever(
                index_key,
                retriever,
                persist_dir,
                vectorstore=vectorstore,
            )

            manifest = self._load_index_manifest()
            manifest[index_key] = persist_dir
            self._save_index_manifest(manifest)

        except Exception as e:
            return f"❌ Failed to register retriever: {e}"

        return f"✅ Indexed {len(chunks)} chunks from {len(valid_docs)} documents"

    def remove_path(self, path: str) -> str:

        path = normalize_path(path)

        if not (_is_dir(path) or _is_file(path)):
            return f"[ERROR] Invalid path (not found): {path}"

        index_key = _make_index_key(path)

        # 1) Pop vectorstore object AND unregister retriever (drop retriever refs first)
        vs = self.retrieval_service.pop_vectorstore(index_key)
        persist_dir = self.retrieval_service.unregister_retriever(index_key)

    

        # 2) Explicitly close/release vectorstore (Windows lock fix)
        self._close_vectorstore_best_effort(vs)

        deleted_ok = False
        delete_err = None

        # 3) Delete folder from disk
        if persist_dir:
            deleted_ok, delete_err = delete_dir_verified(
                persist_dir,
                retries=5,
                sleep_s=0.35,
                debug=False,
            )

        manifest = self._load_index_manifest()

        if persist_dir:
            persist_abs = os.path.abspath(persist_dir)
            persist_gone = not os.path.exists(persist_abs)

            if deleted_ok and persist_gone:
                if index_key in manifest:
                    manifest.pop(index_key, None)
                    self._save_index_manifest(manifest)
                return f"🗑️ Removed index: {path}"

            
            return (
                f"⚠️ Unregistered index, but could not delete vectorstore folder on disk.\n"
                f"Persist dir: {persist_dir}\n"
                f"Likely Windows file lock or antivirus scanning.\n"
                f"Error: {delete_err}"
            )

        # If persist_dir is None (stale registry), remove manifest entry if present
        if index_key in manifest:
            manifest.pop(index_key, None)
            self._save_index_manifest(manifest)

        return f"🗑️ Removed index (no persist dir found): {path}"

    # -------------------- Backward compatible wrappers --------------------

    def index_directory(
        self,
        directory: str,
        force_reindex: bool = False,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
    ) -> str:
        return self.index_path(
            path=directory,
            force_reindex=force_reindex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=True,
        )

    def remove_directory(self, directory: str) -> str:
        return self.remove_path(directory)

    # -------------------- Execution --------------------

    async def run(
        self,
        user_input: str,
        folder: Optional[str] = None,
        top_k: Optional[int] = None,
        enabled_tools: Optional[list[str]] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> str:
        if not self.graph or not getattr(self, "all_tools", None):
            await self.setup()

        if enabled_tools is None:
            tools_to_use = self.all_tools
        else:
            enabled_groups = set(enabled_tools)
            tools_to_use = []
            for t in self.all_tools:
                tags = getattr(t, "tags", []) or []
                name = getattr(t, "name", None)
                if enabled_groups.intersection(tags) or name in enabled_groups:
                    tools_to_use.append(t)

        if set(id(t) for t in tools_to_use) != set(id(t) for t in self.tools):
            print(">>> Rebuilding graph for tools:", [getattr(t, "name", None) for t in tools_to_use])
            await self._build_graph_with_tools(tools_to_use)

        print(f"User input: {user_input!r}")
        print(f"Folder passed in: {folder!r}")
        print(f"Enabled tool groups: {enabled_tools}")
        print("Tools in graph:", [getattr(t, "name", None) for t in self.tools])

        rag_enabled = True if enabled_tools is None else ("rag" in enabled_tools)

        if folder and rag_enabled:
            folder_norm = normalize_path(folder)
            index_key = _make_index_key(folder_norm)
            self.retrieval_service.set_current_folder(index_key)
            print(f">>> Active RAG index set to: {index_key}")
        else:
            self.retrieval_service.set_current_folder(None)
            print(">>> Active RAG folder set to: NONE (RAG disabled or no folder)")

        if top_k is not None and top_k > 0:
            print(f">>> Setting retrieval default_k to: {top_k}")
            self.retrieval_service.default_k = top_k

        active_retriever = self.retrieval_service.get_retriever(self.retrieval_service.current_folder)
        print(f">>> Retriever selected: {active_retriever}")
        print("=========================================\n")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id, "enabled_tools": enabled_tools}}

        if history is not None and len(history) > 0:
            messages = history
        else:
            messages = [HumanMessage(content=user_input)]

        initial_state = SidekickState(messages=messages, success_criteria="Answer fully")
        result = await self.graph.ainvoke(initial_state, config)

        print("\n======= LANGGRAPH OUTPUT MESSAGES =======")
        for i, msg in enumerate(result["messages"]):
            print(f"[{i}] {type(msg).__name__}: {getattr(msg, 'content', None)!r}")
        print("=========================================\n")

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("💭"):
                return msg.content

        return "No response generated"

    def cleanup(self):
        # Best-effort release of all vectorstores first
        for k in list(getattr(self.retrieval_service, "vectorstores", {}).keys()):
            vs = self.retrieval_service.pop_vectorstore(k)
            self._close_vectorstore_best_effort(vs)

        for persist_dir in self.retrieval_service.vectorstore_paths.values():
            self.indexing_service.remove_vectorstore(persist_dir)

        self.retrieval_service.clear()
        print("🧹 Resources cleaned")


async def init_sidekick() -> Sidekick:
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick
