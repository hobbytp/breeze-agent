"""Define the answer generation workflow graph."""

from langgraph.graph import StateGraph, START, END

from web_research_graph.state import InterviewState
from answers.nodes.search import search_for_context
from answers.nodes.generate import generate_expert_answer

def create_answer_graph() -> StateGraph:
    """Create the answer generation subgraph."""
    builder = StateGraph(InterviewState)

    # Add nodes
    builder.add_node("search_context", search_for_context)
    builder.add_node("generate_answer", generate_expert_answer)

    # Add edges
    builder.add_edge(START, "search_context")
    builder.add_edge("search_context", "generate_answer")
    builder.add_edge("generate_answer", END)

    graph = builder.compile()
    graph.name = "Answer Generator"
    return graph

# Create the graph
answer_graph = create_answer_graph() 