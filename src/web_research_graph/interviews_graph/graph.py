"""Define the interview workflow graph."""

from langgraph.graph import StateGraph, START, END
from langgraph.pregel import RetryPolicy

from web_research_graph.state import InterviewState
from web_research_graph.interviews_graph.nodes.initialize import initialize_interview
from web_research_graph.interviews_graph.nodes.question import generate_question
from web_research_graph.interviews_graph.nodes.next_editor import next_editor
from web_research_graph.interviews_graph.nodes.search_context import search_for_context
from web_research_graph.interviews_graph.nodes.generate_answer import generate_expert_answer
from web_research_graph.interviews_graph.router import route_messages

builder = StateGraph(InterviewState)
        
# Add nodes
builder.add_node("initialize", initialize_interview)
builder.add_node(
    "ask_question", 
    generate_question, 
    retry=RetryPolicy(max_attempts=5)
)
builder.add_node(
    "search_context",
    search_for_context,
    retry=RetryPolicy(max_attempts=3)
)
builder.add_node(
    "generate_answer",
    generate_expert_answer,
    retry=RetryPolicy(max_attempts=5)
)
builder.add_node("next_editor", next_editor)
        
# Add edges
builder.add_edge(START, "initialize")
builder.add_edge("initialize", "ask_question")
builder.add_edge("ask_question", "search_context")
builder.add_edge("search_context", "generate_answer")
builder.add_conditional_edges(
    "generate_answer",
    route_messages,
    {
        "ask_question": "ask_question",
        "next_editor": "next_editor",
        "end": END
    }
)
builder.add_conditional_edges(
    "next_editor",
    lambda x: "end" if x.is_complete else "ask_question",
    {
        "ask_question": "ask_question",
        "end": END
    }
)
        
interview_graph = builder.compile() 
interview_graph.name = "Interview Conductor"
