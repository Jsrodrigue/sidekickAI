from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def build_graph(state_class, worker, evaluator, tools, memory):
    g = StateGraph(state_class)
    g.add_node("worker", worker)
    g.add_node("tools", ToolNode(tools=tools))
    g.add_node("evaluator", evaluator)

    g.add_edge(START, "worker")
    g.add_conditional_edges("worker", lambda s: "tools" if hasattr(s.messages[-1], "tool_calls") else "evaluator",
                            {"tools": "tools", "evaluator": "evaluator"})
    g.add_edge("tools", "worker")
    g.add_conditional_edges("evaluator", lambda s: "end" if s.criteria_met or s.needs_user_input else "worker",
                            {"worker": "worker", "end": END})
    return g.compile(checkpointer=memory)
