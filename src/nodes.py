from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def worker_node(llm, tools, state):
    system_msg = SystemMessage(content=f"Eres un asistente útil.\nCriterios: {state.success_criteria}")
    messages = [system_msg] + state.messages
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def evaluator_node(llm, state, evaluator_output_class):
    conversation = "\n".join([
        f"{'👤' if isinstance(m, HumanMessage) else '🤖'}: {getattr(m, 'content', '')}"
        for m in state.messages[-6:]
    ])
    last_response = state.messages[-1].content if state.messages else ""
    prompt = f"Evaluate the conversation:\n{conversation}\n Last response:\n{last_response}\nCriteria: {state.success_criteria}"

    eval_result = llm.with_structured_output(evaluator_output_class).invoke([
        SystemMessage(content="You are an objective evaluator."), 
        HumanMessage(content=prompt)
    ])
    eval_dict = eval_result.model_dump()
    eval_dict["timestamp"] = datetime.now().isoformat()
    return {
        "evaluation_history": [eval_dict],
        "criteria_met": eval_result.success_criteria_met,
        "needs_user_input": eval_result.user_input_needed,
        "messages": [AIMessage(content=f"💭 Evaluación: {eval_result.feedback}")]
    }

def should_continue(state):
    last = state.messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "evaluator"

def should_end(state):
    if state.criteria_met or state.needs_user_input:
        return "end"
    return "worker"
