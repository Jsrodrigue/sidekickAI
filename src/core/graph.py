"""
Construcción del grafo LangGraph.
Define los nodos y edges del flujo de conversación.
"""
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.core.state import SidekickState


class GraphBuilder:
    """Constructor del grafo LangGraph para Sidekick."""

    def __init__(self, worker_llm, evaluator_llm, tools, memory):
        self.worker_llm = worker_llm
        self.evaluator_llm = evaluator_llm
        self.tools = tools
        self.memory = memory

    # -------------------- Nodos --------------------

    def worker_node(self, state: SidekickState) -> dict:
        """Nodo trabajador que procesa consultas del usuario."""
        system_msg = SystemMessage(
            content=(
                f"You are a helpful assistant with tool access.\n\n"
                f"**Success criteria:** {state.success_criteria or 'Provide a clear, correct answer.'}\n\n"
                f"**Available tools:**\n- search_documents: search indexed documents\n\n"
                f"**Current time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                "Use tools when you need external information. Be concise."
            )
        )
        messages = [system_msg] + state.messages
        response = self.worker_llm.invoke(messages)
        return {"messages": [response]}

    def evaluator_node(self, state: SidekickState) -> dict:
        """Nodo evaluador que verifica si se cumplieron los criterios."""
        # Construir contexto de conversación
        conversation = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {getattr(m, 'content', '')}"
            for m in state.messages[-6:]
        ])
        
        last_response = state.messages[-1].content if state.messages else ""
        
        eval_prompt = (
            f"Evaluate the conversation:\n\n{conversation}\n\n"
            f"Last response:\n{last_response}\n\n"
            f"Success criteria:\n{state.success_criteria or 'Provide clear and correct answer.'}\n\n"
            "Answer whether the response meets the criteria, whether more user info is required, "
            "and provide brief feedback."
        )

        eval_result = self.evaluator_llm.invoke([
            SystemMessage(content="You are an objective AI evaluator."),
            HumanMessage(content=eval_prompt)
        ])

        eval_dict = eval_result.model_dump()
        eval_dict["timestamp"] = datetime.now().isoformat()

        return {
            "evaluation_history": [eval_dict],
            "criteria_met": eval_result.success_criteria_met,
            "needs_user_input": eval_result.user_input_needed,
            "messages": [AIMessage(content=f"💭 Evaluation: {eval_result.feedback}")],
        }

    # -------------------- Edge conditions --------------------

    def should_continue(self, state: SidekickState) -> str:
        """Determina si debe ir a herramientas o evaluador."""
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    def should_end(self, state: SidekickState) -> str:
        """Determina si debe terminar o continuar iterando."""
        if state.criteria_met or state.needs_user_input:
            return "end"
        return "worker"

    # -------------------- Construcción --------------------

    async def build(self):
        """Construye y compila el grafo."""
        graph_builder = StateGraph(SidekickState)
        
        # Agregar nodos
        graph_builder.add_node("worker", self.worker_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator_node)
        
        # Agregar edges
        graph_builder.add_edge(START, "worker")
        graph_builder.add_conditional_edges(
            "worker",
            self.should_continue,
            {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator",
            self.should_end,
            {"worker": "worker", "end": END}
        )
        
        return graph_builder.compile(checkpointer=self.memory)