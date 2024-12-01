"""Node for refining the outline based on interview results."""

from typing import Optional, Dict, Any
from langchain_core.runnables import RunnableConfig

from react_agent.configuration import Configuration
from react_agent.state import State, Outline, Section, Subsection
from react_agent.utils import load_chat_model, get_message_text
from react_agent.prompts import REFINE_OUTLINE_PROMPT

def dict_to_outline(outline_dict: Dict[str, Any]) -> Outline:
    """Convert a dictionary to an Outline object."""
    sections = []
    for section_data in outline_dict.get("sections", []):
        subsections = []
        for subsection_data in section_data.get("subsections", []) or []:
            subsections.append(Subsection(
                subsection_title=subsection_data["subsection_title"],
                description=subsection_data["description"]
            ))
        
        sections.append(Section(
            section_title=section_data["section_title"],
            description=section_data["description"],
            subsections=subsections
        ))
    
    return Outline(
        page_title=outline_dict["page_title"],
        sections=sections
    )

async def refine_outline(
    state: State, 
    config: Optional[RunnableConfig] = None
) -> State:
    """Refine the outline based on interview results."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.long_context_model)
    
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
    
    # Generate refined outline
    refined_outline = await chain.ainvoke(
        {
            "topic": current_outline.page_title,
            "old_outline": current_outline.as_str,
            "conversations": conversations,
        },
        config
    )
    
    # Return updated state with new outline
    return State(
        messages=state.messages,
        outline=refined_outline,
        related_topics=state.related_topics,
        perspectives=state.perspectives,
        is_last_step=state.is_last_step
    ) 