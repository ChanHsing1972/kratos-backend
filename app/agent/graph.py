"""
LangGraph 模板
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END


# Define the state for the Agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Add other state fields as needed, e.g., current_plan, tools_used


def initial_plan(state: AgentState) -> dict:
    """Initial planning node."""
    # Generate initial fitness plan
    return {"messages": []}


def execute_tools(state: AgentState) -> dict:
    """Execute required tools based on current plan."""
    return {"messages": []}


def reflect(state: AgentState) -> dict:
    """Reflection node for checking safety and feasibility."""
    return {"messages": []}


def build_graph() -> StateGraph:
    """Build the LangGraph for the Kratos Fitness Agent."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", initial_plan)
    workflow.add_node("executor", execute_tools)
    workflow.add_node("reflector", reflect)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")
    workflow.add_edge(
        "reflector", END
    )  # Replace with conditional edges for re-planning

    return workflow.compile()


agent_graph = build_graph()
