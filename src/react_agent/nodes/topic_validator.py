"""Node for validating and extracting the topic from user input."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from typing import Dict

from react_agent.configuration import Configuration
from react_agent.state import State, TopicValidation
from react_agent.utils import load_chat_model

TOPIC_VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant whose job is to ensure the user provides a clear topic for research.
        Analyze the user's input and determine if it contains a clear research topic.
        
        Example valid topics:
        - "Artificial Intelligence"
        - "The French Revolution"
        - "Quantum Computing"
        
        Return a structured response with:
        - is_valid: true if a clear topic is provided, false otherwise
        - topic: the extracted topic if valid, null otherwise
        - message: a helpful message if the input is invalid, null otherwise
        
        For invalid inputs or small talk, provide a polite message asking for a specific topic."""
    ),
    ("user", "{input}"),
])

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