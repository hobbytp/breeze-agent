"""Define a custom outline generation graph."""

from typing import Dict, List, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.nodes.outline_generator import generate_outline
from react_agent.nodes.topic_expander import expand_topics
from react_agent.nodes.perspectives_generator import generate_perspectives
from react_agent.nodes.interview_conductor import create_interview_graph
from react_agent.nodes.outline_refiner import refine_outline
from react_agent.nodes.article_generator import generate_article
from react_agent.nodes.topic_validator import validate_topic
from react_agent.nodes.topic_input import request_topic


def should_continue(state: State) -> bool:
    """Determine if the graph should continue to the next node."""
    return state.topic['is_valid']

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("validate_topic", validate_topic)
builder.add_node("request_topic", request_topic)
builder.add_node("generate_outline", generate_outline)
builder.add_node("expand_topics", expand_topics)
builder.add_node("generate_perspectives", generate_perspectives)
builder.add_node("refine_outline", refine_outline)
builder.add_node("generate_article", generate_article)

# Add the interview subgraph
interview_graph = create_interview_graph()
builder.add_node("conduct_interviews", interview_graph)

# Set up the flow
builder.add_edge("__start__", "validate_topic")
builder.add_conditional_edges(
    "validate_topic",
    should_continue,
    {
        True: "generate_outline",
        False: "request_topic"
    }
)
builder.add_edge("request_topic", "validate_topic")  # Loop back to validation
builder.add_edge("generate_outline", "expand_topics")
builder.add_edge("expand_topics", "generate_perspectives")
builder.add_edge("generate_perspectives", "conduct_interviews")
builder.add_edge("conduct_interviews", "refine_outline")
builder.add_edge("refine_outline", "generate_article")
builder.add_edge("generate_article", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(interrupt_after=["request_topic"])  # Interrupt after requesting topic
graph.name = "Research and Outline Generator"
