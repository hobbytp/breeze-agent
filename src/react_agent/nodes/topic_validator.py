"""Node for validating and extracting the topic from user input."""

from langchain_core.runnables import RunnableConfig
from typing import Dict

from react_agent.configuration import Configuration
from react_agent.state import State, TopicValidation
from react_agent.utils import load_chat_model
from react_agent.prompts import TOPIC_VALIDATOR_PROMPT

async def validate_topic(state: State, config: RunnableConfig) -> Dict:
    """Validate and extract the topic from user input."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.fast_llm_model)
    
    # Get the last user message
    last_user_message = next(
        (msg for msg in reversed(state.messages) if msg.type == "human"),
        None,
    )
    if not last_user_message:
        raise ValueError("No user message found in state")
    
    # Validate the topic using structured output
    chain = TOPIC_VALIDATOR_PROMPT | model.with_structured_output(TopicValidation)
    response = await chain.ainvoke(
        {"input": last_user_message.content},
        config=config,
    )

    message = []
    if not response['is_valid']:
        message = response['message']
    
    return {
        "topic": response,
        "message": message,
    }