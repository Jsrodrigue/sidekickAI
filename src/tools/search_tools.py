
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


def build_search_tools():
    """
    Build web search tools (currently using Serper).
    """

    serper = GoogleSerperAPIWrapper()

    @tool
    def web_search(query: str) -> str:
        """
        Perform a web search using Google Serper.

        Parameters
        ----------
        query : str
            Natural-language query to search for.

        Returns
        -------
        str
            The raw search results as a string (JSON or text) returned by Serper.
        """
        return serper.run(query)

    return [web_search]
