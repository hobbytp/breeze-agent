"""Node for searching relevant context for answers."""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from web_research_graph.state import InterviewState
from web_research_graph.tools import search

async def search_for_context(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Search for relevant information to answer the question."""
    if state.editor is None:
        raise ValueError("Editor not found in state")
        
    # Get the last question from the editor
    last_question = next(
        (msg for msg in reversed(state.messages) 
         if isinstance(msg, HumanMessage)),
        None
    )
    
    if not last_question:
        # If no question is found, just return the state unchanged
        return state

    # Perform search
    search_results = await search(last_question.content, config=config)
    
    # Store results in references
    if search_results:
        references = state.references or {}
        # Handle search results whether they're strings or dictionaries
        for result in search_results:
            if isinstance(result, dict):
                # Handle dictionary format
                references[result.get("url", "unknown")] = result.get("content", "")
            elif isinstance(result, str):
                # Handle string format
                references[f"source_{len(references)}"] = result
            
        return InterviewState(
            messages=state.messages,
            references=references,
            editor=state.editor,
            editors=state.editors,
            current_editor_index=state.current_editor_index
        )
    
    return state 