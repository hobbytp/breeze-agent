"""Router functions for managing interview flow."""

from langchain_core.messages import AIMessage

from web_research_graph.state import InterviewState
from web_research_graph.utils import sanitize_name

MAX_TURNS = 3
EXPERT_NAME = "expert"

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
    
    # Debug: Print current conversation state
    print(f"[DEBUG] Current editor: {current_editor_name}")
    print(f"[DEBUG] Total messages in conversation: {len(current_messages)}")
    
    # Get the last message
    if current_messages:
        last_message = current_messages[-1]
        print(f"[DEBUG] Last message from: {last_message.name if hasattr(last_message, 'name') else 'unknown'}")
        
        # Since route_messages is called AFTER answer_question, 
        # the last message is almost always from expert
        if isinstance(last_message, AIMessage) and last_message.name == EXPERT_NAME:
            # Check if the previous message (from editor) wanted to end the conversation
            if len(current_messages) >= 2:
                prev_message = current_messages[-2]
                print(f"[DEBUG] Previous message from: {prev_message.name if hasattr(prev_message, 'name') else 'unknown'}")
                
                if (isinstance(prev_message, AIMessage) and 
                    prev_message.name == current_editor_name):
                    
                    # Check for structured output metadata
                    wants_to_end = False
                    end_reason = None
                    
                    if hasattr(prev_message, 'additional_kwargs') and prev_message.additional_kwargs:
                        wants_to_end = prev_message.additional_kwargs.get("wants_to_end", False)
                        end_reason = prev_message.additional_kwargs.get("end_reason")
                        print(f"[DEBUG] Editor wants to end: {wants_to_end}, reason: {end_reason}")
                    else:
                        print("[DEBUG] No structured metadata found in editor message")
                    
                    if wants_to_end:
                        print("[DEBUG] Editor wanted to end conversation - ending now")
                        return "next_editor"
            
            # Count expert responses in this conversation
            expert_responses = len([
                m for m in current_messages 
                if isinstance(m, AIMessage) and m.name == EXPERT_NAME
            ])
            
            print(f"[DEBUG] Expert responses so far: {expert_responses}")
            
            # Check if we've reached max turns
            if expert_responses >= MAX_TURNS:
                print("[DEBUG] Max turns reached - ending conversation")
                return "next_editor"
            
            print("[DEBUG] Continuing conversation - editor should ask next question")
            return "ask_question"
            
        # If the last message was from the editor (rare case, but handle it)
        if isinstance(last_message, AIMessage) and last_message.name == current_editor_name:
            print("[DEBUG] Last message from editor - expert should answer")
            return "ask_question"
    
    # If we're just starting, ask a question
    print("[DEBUG] Starting conversation")
    return "ask_question" 