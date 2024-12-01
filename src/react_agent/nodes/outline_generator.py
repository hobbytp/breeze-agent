"""Node for generating Wikipedia-style outlines."""

from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from react_agent.configuration import Configuration
from react_agent.state import Outline, State
from react_agent.utils import load_chat_model

# Define the prompt template
OUTLINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific.",
        ),
        ("user", "{topic}"),
    ]
)


async def generate_outline(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Generate a Wikipedia-style outline for a given topic.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message and the generated outline.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the fast LLM for outline generation
    model = load_chat_model(configuration.fast_llm_model)

    # Get the topic from the last user message
    last_user_message = next(
        (msg for msg in reversed(state.messages) if msg.type == "human"),
        None,
    )
    if not last_user_message:
        raise ValueError("No user message found in state")

    # Create the chain for outline generation with structured output
    chain = OUTLINE_PROMPT | model.with_structured_output(Outline)

    # Generate the outline
    response = await chain.ainvoke({"topic": last_user_message.content}, config)
    # Create a response message
   

    return {
        "outline": response,
    } 