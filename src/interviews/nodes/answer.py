"""Node for generating expert answers during interviews."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.state import InterviewState
from web_research_graph.prompts import INTERVIEW_ANSWER_PROMPT
from web_research_graph.utils import load_chat_model
from answers.graph import answer_graph
from interviews.utils import swap_roles

EXPERT_NAME = "expert"

async def generate_answer(state: InterviewState, config: RunnableConfig):
    """Generate an answer from the expert's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state. Make sure to set the editor before starting the interview.")
    
    swapped = swap_roles(state, EXPERT_NAME)
    chain = INTERVIEW_ANSWER_PROMPT | model
    
    result = await chain.ainvoke({"messages": swapped.messages}, config)
    
    content = result.content if hasattr(result, 'content') else str(result)
    
    return InterviewState(
        messages=state.messages + [AIMessage(content=content, name=EXPERT_NAME)],
        editor=state.editor,
        references=state.references,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    ) 