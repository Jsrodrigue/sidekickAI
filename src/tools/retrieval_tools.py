
from langchain_core.tools import tool


def build_retrieval_tools(retrieval_service):
    """
    Factory that builds all tools related to RAG (Retrieval-Augmented Generation).

    This exposes a single search tool that the LLM can call whenever it needs
    to retrieve information from the user's indexed folders.
    """

    @tool
    def search_documents(query: str, k: int = 0) -> str:
        """
         Search through the user's indexed documents using the RAG system.

        This tool retrieves the most relevant text chunks from the user's
        knowledge base based on semantic search.

        -------------------------
        🔧 PARAMETERS
        -------------------------
        query : str
            The natural language question or text to search for.
        k : int (optional)
            Number of chunks to retrieve.
            - If k == 0 → the system automatically uses `retrieval_service.default_k`.
            - If default_k is not set → uses a fixed fallback of 5.

        -------------------------
         LLM BEHAVIOR GUIDANCE
        -------------------------
        ALWAYS call this tool when:
        - You need factual information from the user's documents.
        - The user asks anything requiring context from indexed folders.
        - You need to reference or quote content from stored files.
        - You need grounding before answering questions.

        DO NOT hallucinate; if the answer depends on document content,
        call this tool first.

        -------------------------
         RETURNS
        -------------------------
        A formatted string containing the retrieved chunks, each with:
        - Document number
        - File name
        - First 500 characters of content

        If no retrievers exist, returns an instructive message.
        """

        # No folders indexed → give the LLM a clear instruction
        if not retrieval_service.retriever_registry:
            return (
                "❌ No indexed knowledge base available.\n"
                "Please index a folder before attempting document search."
            )

        # Determine number of results to fetch
        effective_k = k or getattr(retrieval_service, "default_k", 5)

        # Perform the search
        return retrieval_service.search(query=query, k=effective_k)

    return [search_documents]
