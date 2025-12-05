from langchain_community.agent_toolkits import FileManagementToolkit


def build_file_tools():
    """Returns file management tools (read, write, list, delete)."""
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()
