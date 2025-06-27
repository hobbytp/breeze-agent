"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field
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


class RelatedTopics(BaseModel):
    """Represents related topics for research."""
    
    topics: List[str] = Field(
        description="List of related topics that are relevant to the main research subject"
    )


class Subsection(BaseModel):
    """Represents a subsection in a Wikipedia article."""
    
    subsection_title: str = Field(
        description="The title of the subsection"
    )
    description: str = Field(
        description="The detailed content of the subsection"
    )

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the subsection."""
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    """Represents a section in a Wikipedia article."""
    
    section_title: str = Field(
        description="The title of the section"
    )
    description: str = Field(
        description="The main content/summary of the section"
    )
    subsections: List[Subsection] = Field(
        description="List of subsections within this section"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of citations supporting the section content"
    )

    @property
    def as_str(self) -> str:
        """Return a formatted string representation of the section."""
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f"[{i+1}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


class Outline(BaseModel):
    """Represents a complete Wikipedia-style outline."""

    page_title: str = Field(
        description="The main title of the Wikipedia article"
    )
    sections: List[Section] = Field(
        description="List of sections that make up the article"
    )

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
class OutputState:
    """Defines the output state for the agent."""

    article: Optional[str] = field(default=None)

class TopicValidation(BaseModel):
    """Structured output for topic validation."""
    
    is_valid: bool = Field(
        description="Indicates whether the topic is valid for article generation"
    )
    topic: Optional[str] = Field(
        description="The validated and possibly reformulated topic"
    )
    message: Optional[str] = Field(
        description="Feedback message about the topic validation result"
    )

class EditorResponse(BaseModel):
    """Structured output for editor responses during interviews."""
    
    message: str = Field(
        description="The editor's question or response content"
    )
    wants_to_end: bool = Field(
        description="Whether the editor wants to end the conversation"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional reason for ending the conversation"
    )

def default_topic_validation() -> TopicValidation:
    """Create a default TopicValidation instance."""
    return TopicValidation(is_valid=False, topic=None, message=None)

@dataclass
class State(InputState, OutputState):
    """Represents the complete state of the agent."""

    is_last_step: IsLastStep = field(default=False)
    outline: Optional[Outline] = field(default=None)
    related_topics: Optional[RelatedTopics] = field(default=None)
    perspectives: Optional[Perspectives] = field(default=None)
    article: Optional[str] = field(default=None)
    references: Annotated[Optional[dict], field(default=None)] = None
    topic: TopicValidation = field(default_factory=default_topic_validation)
    all_conversations: Annotated[Optional[dict], field(default=None)] = None

@dataclass
class InterviewState:
    """State for the interview process between editors and experts."""
    
    messages: Annotated[List[AnyMessage], add_messages] = field(default_factory=list)
    references: Annotated[Optional[dict], field(default=None)] = None
    editor: Annotated[Optional[Editor], field(default=None)] = None
    editors: List[Editor] = field(default_factory=list)
    current_editor_index: int = field(default=0)
    is_complete: bool = field(default=False)
    perspectives: Optional[Perspectives] = field(default=None)

def extract_conversations_by_editor(state: State) -> dict:
    """
    从State中提取按编辑器组织的对话。
    
    优先使用all_conversations，如果不存在则从messages中解析。
    返回格式: {"editor_name": [messages_for_this_editor]}
    """
    from typing import Dict
    from langchain_core.messages import AIMessage
    
    # 如果存在结构化对话，直接使用
    if state.all_conversations:
        return state.all_conversations
    
    # 否则从messages中解析（向后兼容）
    if not state.messages or not state.perspectives:
        return {}
    
    # 处理perspectives可能是dict的情况
    if isinstance(state.perspectives, dict):
        if 'editors' not in state.perspectives or not state.perspectives['editors']:
            return {}
        editors = state.perspectives['editors']
        # 如果editors中的项目是dict，需要转换为Editor对象
        if editors and isinstance(editors[0], dict):
            editors = [Editor(**editor_dict) for editor_dict in editors]
    else:
        if not state.perspectives.editors:
            return {}
        editors = state.perspectives.editors
    
    conversations = {}
    current_editor = None
    current_conversation = []
    
    for message in state.messages:
        if isinstance(message, AIMessage):
            # 检查是否是分隔符消息
            if (message.name == "system" and 
                "Interview with" in message.content and 
                "---" in message.content):
                # 保存上一个编辑器的对话
                if current_editor and current_conversation:
                    conversations[current_editor] = current_conversation.copy()
                
                # 提取新编辑器名称
                for editor in editors:
                    editor_name = editor.name if hasattr(editor, 'name') else editor.get('name', '')
                    if editor_name in message.content:
                        current_editor = editor_name
                        current_conversation = []
                        break
            else:
                # 普通消息，添加到当前对话
                if current_editor:
                    current_conversation.append(message)
    
    # 保存最后一个编辑器的对话
    if current_editor and current_conversation:
        conversations[current_editor] = current_conversation
    
    return conversations


def format_conversations_for_outline(state: State) -> str:
    """
    按编辑器顺序格式化对话，用于outline refinement。
    
    确保按照State中perspectives.editors的顺序来组织对话。
    """
    if not state.perspectives:
        return ""
    
    # 处理perspectives可能是dict的情况（LangGraph序列化）
    if isinstance(state.perspectives, dict):
        if 'editors' not in state.perspectives or not state.perspectives['editors']:
            return ""
        editors = state.perspectives['editors']
        # 如果editors中的项目是dict，需要转换为Editor对象
        if editors and isinstance(editors[0], dict):
            editors = [Editor(**editor_dict) for editor_dict in editors]
    else:
        # perspectives是Perspectives对象
        if not state.perspectives.editors:
            return ""
        editors = state.perspectives.editors
    
    conversations = extract_conversations_by_editor(state)
    formatted_parts = []
    
    # 按照原始编辑器顺序组织对话
    for editor in editors:
        editor_name = editor.name
        if editor_name in conversations and conversations[editor_name]:
            # 格式化这个编辑器的对话
            editor_conversations = []
            for message in conversations[editor_name]:
                if hasattr(message, 'name') and hasattr(message, 'content'):
                    # 获取消息内容
                    content = message.content if hasattr(message, 'content') else str(message)
                    if content and content.strip():
                        editor_conversations.append(f"**{message.name}**: {content}")
            
            if editor_conversations:
                formatted_parts.append(
                    f"### Interview with {editor_name} ({editor.role})\n\n" + 
                    "\n\n".join(editor_conversations)
                )
    
    return "\n\n---\n\n".join(formatted_parts)
