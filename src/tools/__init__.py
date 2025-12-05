# src/tools/__init__.py

from typing import List, Any

from src.tools.retrieval_tools import build_retrieval_tools
from src.tools.file_tools import build_file_tools
from src.tools.search_tools import build_search_tools
from src.tools.python_tools import build_python_tools
from src.tools.wikipedia_tools import build_wikipedia_tools


def build_all_tools(retrieval_service) -> List[Any]:
    """
    Build and return the complete list of tools used by Sidekick.

    Parameters
    ----------
    retrieval_service
        The retrieval service instance used for RAG operations.

    Returns
    -------
    list
        All tools (RAG, files, web search, Python REPL, Wikipedia).
    """
    tools = []

    # RAG tools
    tools.extend(build_retrieval_tools(retrieval_service))

    # File tools (read/write/list in sandbox)
    tools.extend(build_file_tools())

    # Web search
    tools.extend(build_search_tools())

    # Python REPL
    tools.extend(build_python_tools())

    # Wikipedia
    tools.extend(build_wikipedia_tools())

    return tools
