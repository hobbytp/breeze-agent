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
from react_agent.nodes.answer_generator import create_answer_graph

MAX_TURNS = 3
EXPERT_NAME = "expert"

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
    return InterviewState(
        messages=converted, 
        editor=state.editor,
        references=state.references,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    )

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
    
    result = await chain.ainvoke(
        {"messages": swapped.messages, "persona": editor.persona},
        config
    )
    
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=state.messages + [AIMessage(content=content, name=editor_name)],
        editor=state.editor,
        references=state.references,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    )

answer_graph = create_answer_graph()

async def generate_answer(state: InterviewState, config: RunnableConfig):
    """Generate an answer from the expert's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state. Make sure to set the editor before starting the interview.")
    
    swapped = swap_roles(state, EXPERT_NAME)
    chain = INTERVIEW_ANSWER_PROMPT | model
    
    result = await chain.ainvoke({"messages": swapped.messages}, config)
    
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=state.messages + [AIMessage(content=content, name=EXPERT_NAME)],
        editor=state.editor,
        references=state.references,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    )

def route_messages(state: InterviewState) -> str:
    """Determine whether to continue the interview or end it."""
    if not state.messages:
        return "end"
        
    messages = state.messages
    current_editor_name = sanitize_name(state.editor.name)
    
    # Find where the current editor's conversation started
    conversation_start = 0
    for i, m in enumerate(messages):
        if (isinstance(m, AIMessage) and 
            m.name == "system" and 
            current_editor_name in m.content):
            conversation_start = i
            break
    
    # Only look at messages after the conversation start
    current_messages = messages[conversation_start:]
    
    # Count responses only for current conversation
    current_responses = len([
        m for i, m in enumerate(current_messages) 
        if isinstance(m, AIMessage) 
        and m.name == EXPERT_NAME
        and any(
            prev.name == current_editor_name 
            for prev in current_messages[max(0, i-1):i]
        )
    ])
    
    if current_responses >= MAX_TURNS:
        return "next_editor"
        
    if len(current_messages) < 2:
        return "ask_question"
    
    # Look for thank you message from current editor
    editor_messages = [
        m for m in current_messages 
        if isinstance(m, AIMessage) 
        and m.name == current_editor_name
    ]
    
    if editor_messages and editor_messages[-1].content.endswith("Thank you so much for your help!"):
        return "next_editor"
    
    return "ask_question"

async def initialize_interview(state: State, config: RunnableConfig) -> InterviewState:
    """Initialize the interview state with editors from perspectives."""
    
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
    
    # Convert editors to proper Editor objects if needed
    editors_list = [
        Editor(**editor) if isinstance(editor, dict) else editor 
        for editor in editors
    ]
    
    # Start with the first editor
    initial_message = AIMessage(
        content=f"So you said you were writing an article on {state.outline.page_title if state.outline else 'this topic'}?",
        name=EXPERT_NAME
    )
    
    return InterviewState(
        messages=[initial_message],
        editor=editors_list[0],
        references={},
        editors=editors_list,
        current_editor_index=0
    )

async def next_editor(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Move to the next editor or end if all editors are done."""
    next_index = state.current_editor_index + 1
    
    if next_index >= len(state.editors):
        return InterviewState(
            messages=state.messages,
            editor=state.editor,
            references=state.references,
            editors=state.editors,
            current_editor_index=next_index,
            is_complete=True
        )
        
    # Add a separator message to mark the start of a new conversation
    separator = AIMessage(
        content=f"\n--- Starting interview with {state.editors[next_index].name} ---\n",
        name="system"
    )
    
    # Start fresh conversation with next editor while keeping history
    initial_message = AIMessage(
        content=f"So you said you were writing an article on this topic?",
        name=EXPERT_NAME
    )
    
    return InterviewState(
        messages=state.messages + [separator, initial_message],
        editor=state.editors[next_index],
        references=state.references,
        editors=state.editors,
        current_editor_index=next_index
    )

def create_interview_graph() -> StateGraph:
    """Create the interview subgraph."""
    builder = StateGraph(InterviewState)
    
    # Add nodes
    builder.add_node("initialize", initialize_interview)
    builder.add_node(
        "ask_question", 
        generate_question, 
        retry=RetryPolicy(max_attempts=5)
    )
    builder.add_node(
        "answer_question",
        answer_graph,
        retry=RetryPolicy(max_attempts=5)
    )
    builder.add_node("next_editor", next_editor)
    
    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "ask_question")
    builder.add_conditional_edges(
        "answer_question",
        route_messages,
        {
            "ask_question": "ask_question",
            "next_editor": "next_editor",
            "end": END
        }
    )
    builder.add_edge("ask_question", "answer_question")
    builder.add_conditional_edges(
        "next_editor",
        lambda x: "end" if x.is_complete else "ask_question",
        {
            "ask_question": "ask_question",
            "end": END
        }
    )
    
    return builder.compile() 