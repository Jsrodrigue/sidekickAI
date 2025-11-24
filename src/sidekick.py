import os
import shutil
import glob
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.output_models import EvaluatorOutput
from src.state import SidekickState

load_dotenv(override=True)

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "vector_db")
os.makedirs(VECTORSTORE_ROOT, exist_ok=True)


class Sidekick:
    """Robust Sidekick implementation with safe loaders, persistent vectorstores,
    exclusion rules, and threaded indexing to avoid blocking the event loop.

    Key features:
    - Uses UnstructuredMarkdownLoader for .md files (handles frontmatter/encoding)
    - Normalizes and validates file paths returned by loaders
    - Excludes virtualenvs and __pycache__ directories
    - Persists Chroma vectorstores per-folder and keeps mapping in memory
    - Removes vectorstore directory on folder removal or forced reindex
    - Provides synchronous methods intended to be called via asyncio.to_thread
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sidekick_id = str(uuid.uuid4())

        # LLMs
        self.worker_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        self.evaluator_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)

        # Embeddings and storage
        self.emb = OpenAIEmbeddings(openai_api_key=api_key)
        # registry maps normalized folder_path -> retriever
        self.retriever_registry: Dict[str, Any] = {}
        # vectorstore_paths maps folder_path -> persist_dir on disk
        self.vectorstore_paths: Dict[str, str] = {}

        # Memory and graph
        self.memory = MemorySaver()
        self.graph = None
        self.tools = []

    # ------------------------
    # Setup
    # ------------------------
    async def setup(self):
        """Create tools and graph. Safe to call from an asyncio context."""

        @tool
        def search_documents(query: str, k: int = 5) -> str:
            """
            Search relevant information in the documents of the active directory.

            This tool executes a Retrieval-Augmented Generation (RAG) query using the
            current directory configured in the Sidekick state. It retrieves the top-k
            most relevant chunks and returns a text response summarizing the findings.

            Parameters
            ----------
            query : str
                Natural-language question or search instruction used to query the
                document index.
            k : int, optional
                Number of most relevant retrieved chunks. Defaults to 5.

            Returns
            -------
            str
                A plain-text summary generated from the retrieved documents.
            """
            return self._rag_query(query, k)

        self.tools = [search_documents]
        self.worker_llm_with_tools = self.worker_llm.bind_tools(self.tools)
        self.evaluator_llm_with_output = self.evaluator_llm.with_structured_output(EvaluatorOutput)

        await self.build_graph()

    # ------------------------
    # Utilities
    # ------------------------
    @staticmethod
    def _normalize_path(path: str) -> str:
        try:
            return os.path.normpath(path) if path else ""
        except Exception:
            return path

    @staticmethod
    def _is_excluded_path(path: str, excluded_dirs: Optional[List[str]] = None) -> bool:
        if not path:
            return True
        excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
        parts = Sidekick._normalize_path(path).replace("\\", "/").split("/")
        return any(ex in parts for ex in excluded_dirs)

    # ------------------------
    #         Indexing
    # ------------------------
    def load_and_register_directory(self, directory: str, force_reindex: bool = False) -> str:
        """Index a directory. This is synchronous and potentially IO-heavy —
        call it inside asyncio.to_thread to avoid blocking the event loop.

        Returns a human-readable status string.
        """
        directory = self._normalize_path(directory)

        if not directory or not os.path.exists(directory):
            return f"[ERROR] Invalid directory: {directory}"

        # Canonical key for registry
        folder_key = os.path.abspath(directory)

        # If already indexed and not forcing reindex, reuse
        if folder_key in self.retriever_registry and not force_reindex:
            return f"[INFO] Already indexed: {directory}"

        # If reindexing, remove previous vectorstore and registry entry
        if force_reindex and folder_key in self.vectorstore_paths:
            prev = self.vectorstore_paths.pop(folder_key, None)
            if prev and os.path.exists(prev):
                try:
                    shutil.rmtree(prev)
                except Exception as e:
                    print(f"[WARNING] Failed to delete old vectorstore {prev}: {e}")
            self.retriever_registry.pop(folder_key, None)

        excluded_dirs = [".venv", "venv", "__pycache__"]

        docs: List[Any] = []

        # --- Manual MD loading ---
        try:
            md_paths = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
            for md_path in md_paths:
                if self._is_excluded_path(md_path, excluded_dirs):
                    continue
                md_path = self._normalize_path(md_path)
                if not os.path.isfile(md_path):
                    print(f"[WARNING] MD file not found/skipping: {md_path}")
                    continue
                try:
                    loader = UnstructuredMarkdownLoader(md_path)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"📘 MD loaded: {md_path}")
                except Exception as e:
                    print(f"[ERROR] Error loading MD {md_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Error scanning MD files: {e}")

        # --- Other text-like files (.txt, .py) via DirectoryLoader but filtered ---
        try:
            # Text files
            txt_paths = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)
            for p in txt_paths:
                if self._is_excluded_path(p, excluded_dirs):
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

            # Python files
            py_paths = glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)
            for p in py_paths:
                if self._is_excluded_path(p, excluded_dirs):
                    continue
                try:
                    loader = PythonLoader(p)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Failed to load python file {p}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning txt/py files: {e}")

        # --- PDFs ---
        try:
            pdf_paths = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
            for p in pdf_paths:
                if self._is_excluded_path(p, excluded_dirs):
                    continue
                try:
                    loader = PyPDFLoader(p)
                    docs.extend(loader.load())
                    print(f"✅ PDF loaded: {p}")
                except Exception as e:
                    print(f"⚠️ Error loading PDF {p}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning PDFs: {e}")

        if not docs:
            return "❌ No readable documents found"

        # Add normalized metadata and validate sources
        valid_docs = []
        for doc in docs:
            src = self._normalize_path(doc.metadata.get("source", ""))
            if not src or not os.path.exists(src):
                # In some loaders the source may be empty — still keep doc but mark
                print(f"[WARNING] Document has invalid source metadata, marking source as unknown: {src}")
                doc.metadata["file_name"] = doc.metadata.get("file_name") or "unknown"
                doc.metadata["file_path"] = src
            else:
                doc.metadata["file_name"] = os.path.basename(src)
                doc.metadata["file_path"] = src
            doc.metadata["indexed_at"] = datetime.now().isoformat()
            valid_docs.append(doc)

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)
        try:
            chunks = splitter.split_documents(valid_docs)
        except Exception as e:
            print(f"[ERROR] Error splitting documents: {e}")
            return "[ERROR] Error during splitting"

        # Create persistent vectorstore
        persist_dir = os.path.join(VECTORSTORE_ROOT, f"{os.path.basename(directory)}_{uuid.uuid4().hex[:8]}")
        try:
            os.makedirs(persist_dir, exist_ok=True)
            vectorstore = Chroma.from_documents(chunks, self.emb, persist_directory=persist_dir)
        except Exception as e:
            print(f"[ERROR] Failed to create vectorstore: {e}")
            # Attempt cleanup
            try:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
            except Exception:
                pass
            return "[ERROR] Failed to create vectorstore"

        # Register retriever
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            self.retriever_registry[folder_key] = retriever
            self.vectorstore_paths[folder_key] = persist_dir
        except Exception as e:
            print(f"⚠️ Failed to register retriever: {e}")
            return "❌ Failed to register retriever"

        return f"✅ Indexed {len(chunks)} chunks from {len(valid_docs)} documents in '{directory}'"

    def remove_directory(self, directory: str) -> str:
        """Remove a previously indexed folder and delete its vectorstore on disk."""
        directory = self._normalize_path(directory)
        folder_key = os.path.abspath(directory)
        if folder_key in self.retriever_registry:
            self.retriever_registry.pop(folder_key, None)
        if folder_key in self.vectorstore_paths:
            persist_dir = self.vectorstore_paths.pop(folder_key)
            try:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
            except Exception as e:
                print(f"⚠️ Failed to remove vectorstore {persist_dir}: {e}")
        return f"🗑️ Removed folder index: {directory}"

    # ------------------------
    # RAG query (sync)
    # ------------------------
    def _rag_query(self, query: str, k: int = 5) -> str:
        if not self.retriever_registry:
            return "❌ No indexed folder. Use load_and_register_directory() first."

        # select the most recent vectorstore (or first)
        active_dir = next(iter(self.retriever_registry.keys()))
        retriever = self.retriever_registry[active_dir]

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
            print(f"⚠️ _rag_query error: {e}")
            return f"❌ Search error: {e}"

    # ------------------------
    # Graph nodes
    # ------------------------
    def worker(self, state: SidekickState) -> dict:
        system_msg = SystemMessage(
            content=(
                f"You are a helpful assistant with tool access.\n\n"
                f"**Success criteria:** {state.success_criteria or 'Provide a clear, correct answer.'}\n\n"
                f"**Available tools:**\n- search_documents: search indexed documents\n\n"
                f"**Current time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                "Use tools when you need external information. Be concise."
            )
        )
        messages = [system_msg] + state.messages
        response = self.worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(self, state: SidekickState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    def evaluator(self, state: SidekickState) -> dict:
        conversation = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {getattr(m, 'content', '')}" for m in state.messages[-6:]]
        )
        last_response = state.messages[-1].content if state.messages else ""
        eval_prompt = (
            f"Evaluate the conversation:\n\n{conversation}\n\nLast response:\n{last_response}\n\n"
            f"Success criteria:\n{state.success_criteria or 'Provide clear and correct answer.'}\n\n"
            "Answer whether the response meets the criteria, whether more user info is required,"
            " and provide brief feedback."
        )

        eval_result = self.evaluator_llm_with_output.invoke(
            [SystemMessage(content="You are an objective AI evaluator."), HumanMessage(content=eval_prompt)]
        )

        eval_dict = eval_result.model_dump()
        eval_dict["timestamp"] = datetime.now().isoformat()

        return {
            "evaluation_history": [eval_dict],
            "criteria_met": eval_result.success_criteria_met,
            "needs_user_input": eval_result.user_input_needed,
            "messages": [AIMessage(content=f"💭 Evaluation: {eval_result.feedback}")],
        }

    def should_end(self, state: SidekickState) -> str:
        if state.criteria_met or state.needs_user_input:
            return "end"
        return "worker"

    async def build_graph(self):
        graph_builder = StateGraph(SidekickState)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)
        graph_builder.add_edge(START, "worker")
        graph_builder.add_conditional_edges("worker", self.should_continue, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.should_end, {"worker": "worker", "end": END})
        self.graph = graph_builder.compile(checkpointer=self.memory)
        return self.graph

    # ------------------------
    # Execution helpers
    # ------------------------
    async def run(self, user_input: str, thread_id: Optional[str] = None) -> str:
        if not self.graph:
            await self.setup()
        thread_id = thread_id or self.sidekick_id
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = SidekickState(messages=[HumanMessage(content=user_input)], success_criteria="Answer fully")
        result = await self.graph.ainvoke(initial_state, config)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("💭"):
                return msg.content
        return "No response generated"

    def cleanup(self):
        # Safely remove all vectorstores on cleanup
        for path in list(self.vectorstore_paths.values()):
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"⚠️ cleanup remove failed {path}: {e}")
        self.retriever_registry.clear()
        self.vectorstore_paths.clear()
        print("🧹 Resources cleaned")


async def init_sidekick() -> Sidekick:
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick
