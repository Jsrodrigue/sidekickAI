# src/tools/__init__.py

from typing import List, Any

from src.tools.retrieval_tools import build_retrieval_tools
from src.tools.file_tools import build_file_tools
from src.tools.search_tools import build_search_tools
from src.tools.python_tools import build_python_tools
from src.tools.wikipedia_tools import build_wikipedia_tools


def _add_tag(tool: Any, tag: str) -> None:
    """
    Safely add a tag to a LangChain tool.

    Many tools (e.g. StructuredTool) already define a `tags` field.
    We rely on that instead of adding custom attributes.
    """
    existing = getattr(tool, "tags", None)
    if existing is None:
        # If tags is not set yet, initialize it
        try:
            tool.tags = [tag]
        except Exception:
            # If for some reason setting tags fails, just ignore tagging
            pass
    else:
        if tag not in existing:
            existing.append(tag)


def build_all_tools(retrieval_service) -> List[Any]:
    """
    Build and return the complete list of tools used by Sidekick.

    Additionally, we tag each tool via its `tags` field so that the UI can
    enable/disable *groups* of tools (rag, files, web_search, python, wikipedia).
    """
    tools: list[Any] = []

    # RAG tools
    for t in build_retrieval_tools(retrieval_service):
        _add_tag(t, "rag")
        tools.append(t)

    # File tools (read/write/list in sandbox)
    for t in build_file_tools():
        _add_tag(t, "files")
        tools.append(t)

    # Web search
    for t in build_search_tools():
        _add_tag(t, "web_search")
        tools.append(t)

    # Python REPL
    for t in build_python_tools():
        _add_tag(t, "python")
        tools.append(t)

    # Wikipedia
    for t in build_wikipedia_tools():
        _add_tag(t, "wikipedia")
        tools.append(t)

    return tools
