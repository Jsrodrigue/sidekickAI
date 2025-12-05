from langchain_experimental.tools import PythonREPLTool

def build_python_tools():
    python_repl = PythonREPLTool()
    return [python_repl]
