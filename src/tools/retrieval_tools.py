from langchain_core.tools import tool


def build_retrieval_tools(retrieval_service):
    """
    Factory function that creates and returns all retrieval-related tools.

    This pattern allows the tools to use the retrieval_service dependency
    without tightly coupling them to the Sidekick class.
    """

    @tool
    def search_documents(query: str, k: int = 5) -> str:
        """
        Search for relevant text chunks in the currently active folder.
        
        ALWAYS use this tool when:
        - The user asks a question that may relate to their documents.
        - You need context or factual information from indexed files.
        Parameters
        ----------
        query : str
            Natural-language user query to search for.
        k : int, optional (default=5)
            Maximum number of top-ranked chunks to return.

        Behavior
        --------
        - Searches *only* within the currently selected folder.
        - The active folder is set by Sidekick.run() or the UI.
        - Returns concatenated retrieved content in a readable form.
        - If no folder is active or no retriever is available,
          a friendly error message is returned.

        Returns
        -------
        str
            The retrieved text results or an explanatory message.
        """
        return retrieval_service.search(query=query, k=k)

    return [search_documents]
