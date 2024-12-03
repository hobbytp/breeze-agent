"""Node for generating expert answers during interviews."""

from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from react_agent.configuration import Configuration
from react_agent.state import InterviewState
from react_agent.utils import load_chat_model
from react_agent.tools import search
from react_agent.prompts import INTERVIEW_ANSWER_PROMPT

EXPERT_NAME = "expert"

def swap_roles(state: InterviewState, name: str) -> InterviewState:
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

async def search_for_context(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Search for relevant information to answer the question."""
    if state.editor is None:
        raise ValueError("Editor not found in state")
        
    # Swap roles to get the correct perspective
    swapped_state = swap_roles(state, EXPERT_NAME)
    
    # Get the last question from the editor (now as HumanMessage after swap)
    last_question = next(
        (msg for msg in reversed(swapped_state.messages) 
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

async def generate_expert_answer(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Generate an expert answer using the gathered information."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state")
    
    # Swap roles for the expert's perspective
    swapped_state = swap_roles(state, EXPERT_NAME)
    
    # Format references for the prompt
    references_text = ""
    if state.references:
        references_text = "\n\n".join(
            f"Source: {url}\nContent: {content}" 
            for url, content in state.references.items()
        )
    
    # Create the chain
    chain = INTERVIEW_ANSWER_PROMPT | model
    
    # Generate answer
    result = await chain.ainvoke(
        {
            "messages": swapped_state.messages, 
            "references": references_text
        },
        config
    )
    
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=state.messages + [AIMessage(content=content, name=EXPERT_NAME)],
        references=state.references,
        editor=state.editor,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    )

def create_answer_graph() -> StateGraph:
    """Create the answer generation subgraph."""
    builder = StateGraph(InterviewState)
    
    # Add nodes
    builder.add_node("search_context", search_for_context)
    builder.add_node("generate_answer", generate_expert_answer)
    
    # Add edges
    builder.add_edge("__start__", "search_context")
    builder.add_edge("search_context", "generate_answer")
    builder.add_edge("generate_answer", "__end__")
    
    return builder.compile() 