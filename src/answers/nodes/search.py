"""Node for searching relevant context for answers."""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from web_research_graph.state import InterviewState
from web_research_graph.tools import search
from web_research_graph.utils import swap_roles

EXPERT_NAME = "expert"

async def search_for_context(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Search for relevant information to answer the question."""
    print("\n=== Debug: search_for_context START ===")
    print(f"Initial state messages: {len(state.messages)}")
    print(f"Initial references: {len(state.references or {})}")
    print(f"Editor: {state.editor.name if state.editor else 'None'}")
    
    if state.editor is None:
        print("No editor found!")
        raise ValueError("Editor not found in state")
    
    # Swap roles to get the correct perspective
    swapped_state = swap_roles(state, EXPERT_NAME)
    print(f"Messages after swap: {len(swapped_state.messages)}")
    
    # Get the last question (now as HumanMessage after swap)
    last_question = next(
        (msg for msg in reversed(swapped_state.messages) 
         if isinstance(msg, HumanMessage)),
        None
    )
    
    if not last_question:
        print("No question found after role swap")
        print(f"Message types: {[type(m) for m in swapped_state.messages]}")
        print("=== Debug: search_for_context END (no question) ===\n")
        return state

    print(f"Found question: {last_question.content[:100]}...")
    
    # Perform search
    search_results = await search(last_question.content, config=config)
    
    print(f"Search returned {len(search_results) if search_results else 0} results")
    
    # Store results in references
    if search_results:
        references = state.references or {}
        for result in search_results:
            if isinstance(result, dict):
                references[result.get("url", "unknown")] = result.get("content", "")
            elif isinstance(result, str):
                references[f"source_{len(references)}"] = result
            
        print(f"Updated references: {len(references)}")
        print("=== Debug: search_for_context END (with results) ===\n")
        return InterviewState(
            messages=state.messages,
            references=references,
            editor=state.editor,
            editors=state.editors,
            current_editor_index=state.current_editor_index
        )
    
    print("No results found")
    print("=== Debug: search_for_context END (no results) ===\n")
    return state 