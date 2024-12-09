"""Define a research and content generation workflow graph"""

from langgraph.graph import StateGraph, START, END

from web_research_graph.configuration import Configuration
from web_research_graph.nodes.article_generator import generate_article
from web_research_graph.interviews_graph.graph import interview_graph
from web_research_graph.nodes.outline_generator import generate_outline
from web_research_graph.nodes.outline_refiner import refine_outline
from web_research_graph.nodes.perspectives_generator import generate_perspectives
from web_research_graph.nodes.topic_expander import expand_topics
from web_research_graph.nodes.topic_input import request_topic
from web_research_graph.nodes.topic_validator import validate_topic
from web_research_graph.state import InputState, OutputState, State

def should_continue(state: State) -> bool:
    """Determine if the graph should continue to the next node."""
    return state.topic.is_valid

builder = StateGraph(State, input=InputState, output=OutputState, config_schema=Configuration)

builder.add_node("validate_topic", validate_topic)
builder.add_node("request_topic", request_topic)
builder.add_node("generate_outline", generate_outline)
builder.add_node("expand_topics", expand_topics)
builder.add_node("generate_perspectives", generate_perspectives)
builder.add_node("refine_outline", refine_outline)
builder.add_node("generate_article", generate_article)

builder.add_node("conduct_interviews", interview_graph)

builder.add_edge(START, "validate_topic")
builder.add_conditional_edges(
    "validate_topic",
    should_continue,
    {
        True: "generate_outline",
        False: "request_topic"
    }
)
builder.add_edge("request_topic", "validate_topic") 
builder.add_edge("generate_outline", "expand_topics")
builder.add_edge("expand_topics", "generate_perspectives")
builder.add_edge("generate_perspectives", "conduct_interviews")
builder.add_edge("conduct_interviews", "refine_outline")
builder.add_edge("refine_outline", "generate_article")
builder.add_edge("generate_article", END)

graph = builder.compile(interrupt_after=["request_topic"])
graph.name = "Research and Outline Generator"
