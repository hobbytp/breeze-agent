"""Node for generating expert answers."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.configuration import Configuration
from web_research_graph.state import InterviewState
from web_research_graph.prompts import INTERVIEW_ANSWER_PROMPT
from web_research_graph.utils import load_chat_model

EXPERT_NAME = "expert"

async def generate_expert_answer(state: InterviewState, config: RunnableConfig) -> InterviewState:
    """Generate an expert answer using the gathered information."""
    print("\n=== Debug: generate_expert_answer ===")
    print(f"Messages count: {len(state.messages)}")
    print(f"References count: {len(state.references or {})}")
    
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    if state.editor is None:
        raise ValueError("Editor not found in state")
    
    # Format references for the prompt
    references_text = ""
    if state.references:
        references_text = "\n\n".join(
            f"Source: {url}\nContent: {content}" 
            for url, content in state.references.items()
        )
        print(f"References text length: {len(references_text)}")
    else:
        print("No references available")
    
    # Create the chain
    chain = INTERVIEW_ANSWER_PROMPT | model
    
    # Generate answer
    result = await chain.ainvoke(
        {
            "messages": state.messages, 
            "references": references_text
        },
        config
    )
    
    content = result.content if hasattr(result, 'content') else str(result)
    print(f"Generated content length: {len(content)}")
    
    if not content:
        print("Warning: Empty content generated, returning original state")
        return state
        
    print("=== End Debug ===\n")
    
    return InterviewState(
        messages=state.messages + [AIMessage(content=content, name=EXPERT_NAME)],
        references=state.references,
        editor=state.editor,
        editors=state.editors,
        current_editor_index=state.current_editor_index
    ) 