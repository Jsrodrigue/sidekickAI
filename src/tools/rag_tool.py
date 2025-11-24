# src/tools/rag_tool.py
from langchain.tools import tool
from src.state import SidekickState
from src.rag_manager import RAGManager

def make_search_documents_tool(rag_manager: RAGManager):
    """
    Creates a LangChain tool to search documents using a RAG manager.
    
    The tool is async and reads the current directory from the runtime state.
    """

    @tool("Search documents in the active folder", return_direct=True)
    async def search_documents(query: str, runtime: dict):
        """
        Search documents in the directory currently set in the user's session.

        Args:
            query (str): The search query string provided by the LLM.
            runtime (dict): LangGraph runtime containing the current session state.

        Returns:
            List[Document]: List of LangChain Document objects matching the query.
                            Returns an empty list if no current_directory is set.
        """
        # Get the current session state injected by LangGraph
        state: SidekickState = runtime.get("state")
        current_dir = getattr(state, "current_directory", None)

        if not current_dir:
            return []

        # Perform the search using the RAG manager
        results = await rag_manager.search(current_dir, query)
        return results

    return search_documents
