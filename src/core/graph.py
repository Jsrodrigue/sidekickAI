"""
Construction of LangGraph Graph.
"""

from datetime import datetime

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.core.state import SidekickState


class GraphBuilder:
    """Constructor of the Graph."""

    def __init__(self, worker_llm, tools, memory):
        self.worker_llm = worker_llm
        self.tools = tools
        self.memory = memory

    # -------------------- Nodes --------------------

    def worker_node(self, state: SidekickState) -> dict:
        """Worker node: main LLM with tool access."""

        # Build a single string for the system prompt
        success_criteria = (
            state.success_criteria
            if getattr(state, "success_criteria", None)
            else "Provide a clear, correct answer."
        )

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        system_content = (
            "You are a helpful assistant with access to several external tools.\n"
            "Some tools may be *disabled* depending on user settings. Always use tools only when "
            "they are clearly helpful and necessary.\n\n"
            
            "NOTES:\n"
            "- If a tool is disabled for this run, do not attempt to call it.\n"
            "- Use tools sparingly and only when needed to answer the user's question.\n"
            "- When using the Python tool, always use print() to show results. Bare expressions will not display values.\n"
            "- Whenever you write mathematical expressions in LaTeX, you MUST always use block delimiters with double dollar signs $$ ... $$"
            "- In each turn use a tool a maximum of 3 times, if you cannot complete the tasks say so and explain what happened.\n\n"
        
            f"Success criteria: {success_criteria}\n"
            f"Current time: {current_time}\n"
        )

        system_msg = SystemMessage(content=system_content)

        # Prepend system message to the existing conversation
        messages = [system_msg] + state.messages

        # Call the bound LLM (worker_llm already has tools bound)
        response = self.worker_llm.invoke(messages)

        # Return partial state update: add the new assistant message
        return {"messages": [response]}
    # -------------------- Edge conditions --------------------

    def should_continue(self, state: SidekickState) -> str:
        """
        Decide if need to use tools or end
        """
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # -------------------- Construcción --------------------

    async def build(self):
        """Build and compile graph."""
        graph_builder = StateGraph(SidekickState)

        # Nodos
        graph_builder.add_node("worker", self.worker_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        # Edges
        graph_builder.add_edge(START, "worker")

        # worker -> tools (si hay tool_calls) o END (si no)
        graph_builder.add_conditional_edges(
            "worker",
            self.should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # tools -> worker (después de ejecutar tools, volvemos al worker)
        graph_builder.add_edge("tools", "worker")

        return graph_builder.compile(checkpointer=self.memory)
