from langchain_experimental.tools import PythonREPLTool
try:
    # LangChain <= 0.1.x
    from langchain.tools import Tool
except ImportError:
    # LangChain >= 0.2.x
    from langchain_core.tools import Tool


def build_python_tools():
    """
    Builds a wrapped Python REPL tool that:
    - Logs the code being executed.
    - Logs the raw execution output.
    - Returns a well-formatted response back to the LLM.
    - Includes tags=["python"] for your UI tool filtering system.
    """
    base_python_repl = PythonREPLTool()

    def python_repl_verbose(code: str) -> str:
        """
        Wrapper around PythonREPLTool that adds detailed logs
        and returns a formatted output visible to the LLM.
        """
        print("\n======= PYTHON TOOL CALL =======")
        print("Received code:")
        print(code)
        print("===============================\n")

        # Run code through LangChain's built-in REPL
        raw_output = base_python_repl.run(code)

        print("------- PYTHON TOOL OUTPUT ------")
        print(raw_output)
        print("=================================\n")

        # Return formatted output to the agent
        return (
            "🔧 **Python REPL executed**\n\n"
            "📥 **Code:**\n"
            "```python\n"
            f"{code}\n"
            "```\n\n"
            "📤 **Output:**\n"
            "```text\n"
            f"{raw_output}\n"
            "```"
        )

    # Build LangChain Tool wrapper
    python_tool = Tool.from_function(
        name="python_repl_verbose",
        description=(
            "Executes Python code and returns both the executed code and its output. "
            "Use this tool ONLY when code execution is necessary. "
            "If the answer is obvious, do NOT call this tool repeatedly."
            "Always use print() staments if you want to see any result"
        ),
        func=python_repl_verbose,
    )

    # Add group tag so it can be toggled in your UI
    python_tool.tags = ["python"]

    return [python_tool]
