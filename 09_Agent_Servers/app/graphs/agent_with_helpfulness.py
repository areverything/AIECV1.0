from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from app.models import get_chat_model
from app.tools import get_tool_belt


def call_model(state: MessagesState) -> dict:
    model = get_chat_model().bind_tools(get_tool_belt())
    return {"messages": [model.invoke(state["messages"])]}


def route_to_action_or_helpfulness(state: MessagesState):
    # Tools requested → run them; a plain answer → grade it.
    last = state["messages"][-1]
    return "action" if getattr(last, "tool_calls", None) else "helpfulness"


_helpfulness_prompt = ChatPromptTemplate.from_template(
    "Given an initial query and a final response, determine if the final response "
    "is extremely helpful or not. Respond with only Y or N.\n\n"
    "Initial Query:\n{initial_query}\n\nFinal Response:\n{final_response}"
)


def helpfulness_node(state: MessagesState) -> dict:
    # Never loop more than 10 times, regardless of the assessment.
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]
    chain = _helpfulness_prompt | get_chat_model() | StrOutputParser()
    result = chain.invoke(
        {"initial_query": initial_query.content, "final_response": final_response.content}
    )
    decision = "Y" if result.strip().upper().startswith("Y") else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


def helpfulness_decision(state: MessagesState):
    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if text == "HELPFULNESS:END":
        return END
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


# Building an explicit StateGraph
#
# START → agent
# agent → (asked for tools?)    yes → action → back to agent
#                               no  → helpfulness
# helpfulness → (verdict?)      Y   → END (ship it)
#                               N   → agent (try again)
#                               END → END (the brake)

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("action", ToolNode(get_tool_belt()))
    graph.add_node("helpfulness", helpfulness_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness", helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()