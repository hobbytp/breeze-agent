"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class Editor:
    """Represents a Wikipedia editor with specific expertise."""
    
    affiliation: str
    name: str
    role: str
    description: str

    @property
    def persona(self) -> str:
        """Return a formatted string representation of the editor."""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


@dataclass
class Perspectives:
    """Represents a group of editors with different perspectives."""
    
    editors: List[Editor] = field(default_factory=list)


@dataclass
class RelatedTopics:
    """Represents related topics for research."""
    topics: List[str] = field(default_factory=list)


@dataclass
class Subsection:
    """Represents a subsection in a Wikipedia-style outline."""

    subsection_title: str
    description: str

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the subsection."""
        return f"### {self.subsection_title}\n\n{self.description}".strip()


@dataclass
class Section:
    """Represents a section in a Wikipedia-style outline."""

    section_title: str
    description: str
    subsections: Optional[List[Subsection]] = None

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the section."""
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


@dataclass
class Outline:
    """Represents a complete Wikipedia-style outline."""

    page_title: str
    sections: List[Section] = field(default_factory=list)

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the outline."""
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


@dataclass
class InputState:
    """Defines the input state for the agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)



@dataclass
class State(InputState):
    """Represents the complete state of the agent."""

    is_last_step: IsLastStep = field(default=False)
    outline: Optional[Outline] = field(default=None)
    related_topics: Optional[RelatedTopics] = field(default=None)
    perspectives: Optional[Perspectives] = field(default=None)


@dataclass
class InterviewState:
    """State for the interview process between editors and experts."""
    
    messages: Annotated[List[AnyMessage], add_messages] = field(default_factory=list)
    references: Annotated[Optional[dict], field(default=None)] = None
    editor: Annotated[Optional[Editor], field(default=None)] = None
    perspectives: Optional[Perspectives] = field(default=None)
