"""Node for conducting interviews between editors and experts."""

from typing import Dict, List, Optional
import re
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.pregel import RetryPolicy

from react_agent.configuration import Configuration
from react_agent.state import InterviewState, Editor, State
from react_agent.prompts import INTERVIEW_QUESTION_PROMPT, INTERVIEW_ANSWER_PROMPT
from react_agent.utils import load_chat_model

MAX_TURNS = 5
EXPERT_NAME = "expert"  # Simplified name for the expert

def sanitize_name(name: str) -> str:
    """Convert a name to a valid format for the API."""
    # Replace spaces and special chars with underscores, keep alphanumeric
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return sanitized

def swap_roles(state: InterviewState, name: str):
    """Convert messages to appropriate roles for the current speaker."""
    converted = []
    for message in state.messages:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.dict(exclude={"type"}))
        converted.append(message)
    return InterviewState(messages=converted, editor=state.editor, references=state.references)

async def generate_question(state: InterviewState, config: RunnableConfig):
    """Generate a question from the editor's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state. Make sure to set the editor before starting the interview.")
    
    editor = state.editor
    editor_name = sanitize_name(editor.name)
    swapped = swap_roles(state, editor_name)
    
    chain = INTERVIEW_QUESTION_PROMPT | model
    
    # Get the raw string content from the result
    result = await chain.ainvoke(
        {"messages": swapped.messages, "persona": editor.persona},
        config
    )
    
    # Extract the string content if result is a message
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=[AIMessage(content=content, name=editor_name)],
        editor=state.editor,
        references=state.references
    )

async def generate_answer(state: InterviewState, config: RunnableConfig):
    """Generate an answer from the expert's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state. Make sure to set the editor before starting the interview.")
    
    swapped = swap_roles(state, EXPERT_NAME)
    chain = INTERVIEW_ANSWER_PROMPT | model
    
    # Get the raw string content from the result
    result = await chain.ainvoke({"messages": swapped.messages}, config)
    
    # Extract the string content if result is a message
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=[AIMessage(content=content, name=EXPERT_NAME)],
        editor=state.editor,
        references=state.references
    )

def route_messages(state: InterviewState) -> str:
    """Determine whether to continue the interview or end it."""
    if not state.messages:
        return END
        
    messages = state.messages
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == EXPERT_NAME]
    )
    if num_responses >= MAX_TURNS:
        return END
        
    if len(messages) < 2:
        return "ask_question"
        
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"

async def initialize_interview(state: State, config: RunnableConfig) -> InterviewState:
    """Initialize the interview state with an editor from perspectives."""
    print("Current state:", state)
    print("Perspectives:", state.perspectives)
    
    # Get editors from perspectives
    if not state.perspectives:
        raise ValueError("No perspectives found in state")
        
    perspectives = state.perspectives
    if isinstance(perspectives, dict):
        editors = perspectives.get("editors", [])
    else:
        editors = perspectives.editors
        
    if not editors:
        raise ValueError("No editors found in perspectives")
    
    # Get the first editor
    editor_data = editors[0]
    editor = Editor(**editor_data) if isinstance(editor_data, dict) else editor_data
    print("Selected editor:", editor)
    
    # Create initial message from expert
    initial_message = AIMessage(
        content=f"So you said you were writing an article on {state.outline.page_title if state.outline else 'this topic'}?",
        name=EXPERT_NAME
    )
    
    # Create and return interview state
    return InterviewState(
        messages=[initial_message],
        editor=editor,
        references={}
    )

def create_interview_graph() -> StateGraph:
    """Create the interview subgraph."""
    builder = StateGraph(InterviewState)
    
    # Add initialization node
    builder.add_node("initialize", initialize_interview)
    
    # Add interview nodes with retry policies
    builder.add_node(
        "ask_question", 
        generate_question, 
        retry=RetryPolicy(max_attempts=5)
    )
    builder.add_node(
        "answer_question", 
        generate_answer, 
        retry=RetryPolicy(max_attempts=5)
    )
    
    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "ask_question")
    builder.add_conditional_edges("answer_question", route_messages)
    builder.add_edge("ask_question", "answer_question")
    
    return builder.compile() 