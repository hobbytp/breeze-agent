"""Utility functions for the interview process."""

import re
from langchain_core.messages import AIMessage, HumanMessage

from web_research_graph.state import InterviewState

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