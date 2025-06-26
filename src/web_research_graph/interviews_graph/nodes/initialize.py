"""Node for initializing the interview process."""

from typing import Union, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import InterviewState, Editor, State, Perspectives

EXPERT_NAME = "expert"

def _extract_editors_from_perspectives(perspectives: Union[Perspectives, dict, None]) -> List[Editor]:
    """从perspectives中提取editors，处理不同的数据类型，返回Editor对象列表"""
    if not perspectives:
        raise ValueError("No perspectives found in state")
    
    # 如果是Perspectives对象
    print(f"!!! === In initialize, perspectives is type: {type(perspectives)}")
    if isinstance(perspectives, Perspectives):
        if not perspectives.editors:
            raise ValueError("No editors found in perspectives")
        return perspectives.editors # type: ignore
    else:
        raise ValueError(f"Invalid perspectives type: {type(perspectives)}")

async def initialize_interview(state: State, config: RunnableConfig) -> InterviewState:
    """Initialize the interview state with editors from perspectives."""
    
    # 提取editors，处理不同的数据类型
    editors_list = _extract_editors_from_perspectives(state.perspectives)
    
    # Start with the first editor
    initial_message = AIMessage(
        content=f"So you said you were writing an article on {state.outline.page_title if state.outline else 'this topic'}?",
        name=EXPERT_NAME
    )
    
    return InterviewState(
        messages=[initial_message],
        editor=editors_list[0],
        references={},
        editors=editors_list,
        current_editor_index=0
    ) 