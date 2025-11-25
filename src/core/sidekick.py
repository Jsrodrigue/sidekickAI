"""
Refactored main Sidekick class.
Coordinates indexing, retrieval services, and the LangGraph pipeline.
"""
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from src.services.indexing_service import IndexingService
from src.services.retrieval_service import RetrievalService
from src.core.graph import GraphBuilder
from src.models.output_models import EvaluatorOutput
from src.core.state import SidekickState
from src.utils.path_utils import normalize_path, validate_directory, get_absolute_path

load_dotenv(override=True)

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "vector_db")


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
            temperature=0
        )
        self.evaluator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0
        )

        # Services
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.indexing_service = IndexingService(embeddings, VECTORSTORE_ROOT)
        self.retrieval_service = RetrievalService()

        # Memory and graph
        self.memory = MemorySaver()
        self.graph = None
        self.tools = []

    async def setup(self):
        """Initializes tools and builds the LangGraph pipeline."""
        
        # ------------------------ Tool definition ------------------------
        @tool
        def search_documents(query: str, k: int = 5) -> str:
            """
            Search for relevant information in indexed documents.

            Args:
                query: Natural-language question or instruction.
                k: Number of top-ranked chunks to retrieve (default: 5).

            Returns:
                Plain-text output summarizing retrieved chunks.
            """
            return self.retrieval_service.search(query, k)

        self.tools = [search_documents]

        # Bind tools and structured outputs
        self.worker_llm_with_tools = self.worker_llm.bind_tools(self.tools)
        self.evaluator_llm_with_output = self.evaluator_llm.with_structured_output(
            EvaluatorOutput
        )

        # Build LangGraph pipeline
        graph_builder = GraphBuilder(
            worker_llm=self.worker_llm_with_tools,
            evaluator_llm=self.evaluator_llm_with_output,
            tools=self.tools,
            memory=self.memory
        )
        self.graph = await graph_builder.build()

    # -------------------- Indexing Operations --------------------

    def index_directory(self, directory: str, force_reindex: bool = False) -> str:
        """
        Synchronously indexes a directory.
        Should be executed inside asyncio.to_thread to avoid blocking.
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
            chunks = self.indexing_service.chunk_documents(valid_docs)
        except Exception as e:
            return f"[ERROR] Error during chunk splitting: {e}"

        # Create vectorstore
        vectorstore, persist_dir = self.indexing_service.create_vectorstore(
            chunks,
            directory
        )
        
        if not vectorstore or not persist_dir:
            return "[ERROR] Failed to create vectorstore"

        # Register retriever
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            self.retrieval_service.register_retriever(folder_key, retriever, persist_dir)
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
        
        return f"🗑️ Removed folder index: {directory}"

    # -------------------- Execution --------------------

    async def run(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Runs a user query through the LangGraph pipeline."""
        if not self.graph:
            await self.setup()
        
        thread_id = thread_id or self.sidekick_id
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = SidekickState(
            messages=[HumanMessage(content=user_input)],
            success_criteria="Answer fully"
        )
        
        result = await self.graph.ainvoke(initial_state, config)
        
        # Extract last assistant message (skipping thoughts)
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
