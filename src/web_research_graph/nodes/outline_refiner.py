"""Node for refining the outline based on interview results."""

from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.state import State, Outline, Section, Subsection
from web_research_graph.utils import load_chat_model, get_message_text, dict_to_outline
from web_research_graph.prompts import REFINE_OUTLINE_PROMPT

async def refine_outline(
    state: State, 
    config: Optional[RunnableConfig] = None
) -> State:
    """Refine the outline based on interview results."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if not state.outline:
        raise ValueError("No initial outline found in state")
        
    # Convert dictionary outline to Outline object if needed
    current_outline = state.outline if isinstance(state.outline, Outline) else dict_to_outline(state.outline)
        
    # Format conversations from the state's messages
    conversations = "\n\n".join(
        f"### {m.name}\n\n{get_message_text(m)}" 
        for m in state.messages
    )
    
    # Create the chain with structured output
    chain = REFINE_OUTLINE_PROMPT | model.with_structured_output(Outline)
    
    # Generate refined outline with explicit structure validation
    refined_outline = await chain.ainvoke(
        {
            "topic": current_outline.page_title,
            "old_outline": current_outline.as_str,
            "conversations": conversations,
        },
        config
    )
    
    # Validate that the refined outline has sections
    if not refined_outline.sections:
        raise ValueError("Refined outline was generated without sections")
    
    # Return updated state with new outline
    return State(
        messages=state.messages,
        outline=refined_outline,
        related_topics=state.related_topics,
        perspectives=state.perspectives,
        is_last_step=state.is_last_step
    ) 