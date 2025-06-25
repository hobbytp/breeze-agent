"""Node for generating interview questions from editors."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import json

from web_research_graph.configuration import Configuration
from web_research_graph.state import InterviewState, EditorResponse
from web_research_graph.prompts import INTERVIEW_QUESTION_PROMPT
from web_research_graph.utils import load_chat_model
from web_research_graph.utils import sanitize_name, swap_roles

async def generate_question(state: InterviewState, config: RunnableConfig):
    """Generate a question from the editor's perspective."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state. Make sure to set the editor before starting the interview.")
    
    editor = state.editor
    editor_name = sanitize_name(editor.name)
    swapped = swap_roles(state, editor_name)
    
    # Use structured output
    chain = INTERVIEW_QUESTION_PROMPT | model.with_structured_output(EditorResponse)
    
    try:
        result = await chain.ainvoke(
            {"messages": swapped.messages, "persona": editor.persona},
            config
        )
        
        # Extract message content and store end conversation intent in state
        content = result.message
        
        # Add metadata to the message to indicate conversation end intent
        message = AIMessage(
            content=content, 
            name=editor_name,
            additional_kwargs={
                "wants_to_end": result.wants_to_end,
                "end_reason": result.reason
            }
        )
        
    except Exception as e:
        # Fallback to regular text output if structured output fails
        print(f"[WARNING] Structured output failed, falling back to text: {e}")
        chain = INTERVIEW_QUESTION_PROMPT | model
        result = await chain.ainvoke(
            {"messages": swapped.messages, "persona": editor.persona},
            config
        )
        content = result.content if hasattr(result, 'content') else str(result)
        message = AIMessage(content=content, name=editor_name)
    
    return InterviewState(
        messages=state.messages + [message],
        editor=state.editor,
        references=state.references,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    ) 