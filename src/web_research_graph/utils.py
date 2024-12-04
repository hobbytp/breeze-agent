"""Utility & helper functions."""

from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from web_research_graph.state import Outline, Section, Subsection
import re
from langchain_core.messages import AIMessage, HumanMessage

from web_research_graph.state import InterviewState

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


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


def dict_to_section(section_dict: Dict[str, Any]) -> Section:
    """Convert a dictionary to a Section object."""
    subsections = []
    for subsection_data in section_dict.get("subsections", []) or []:
        subsections.append(Subsection(
            subsection_title=subsection_data["subsection_title"],
            description=subsection_data["description"]
        ))
    
    return Section(
        section_title=section_dict["section_title"],
        description=section_dict["description"],
        subsections=subsections,
        citations=section_dict.get("citations", [])
    )

def sanitize_name(name: str) -> str:
    """Convert a name to a valid format for the API."""
    # Replace spaces and special chars with underscores, keep alphanumeric
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return sanitized

def swap_roles(state: InterviewState, name: str):
    """Convert messages to appropriate roles for the current speaker."""
    
    converted = []
    for i, message in enumerate(state.messages):
        
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
