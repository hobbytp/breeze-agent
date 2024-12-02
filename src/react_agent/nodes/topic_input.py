"""Node for handling invalid topics and waiting for user input."""

from typing import Dict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.state import State

async def request_topic(state: State, config: RunnableConfig) -> Dict:
    """Request a new topic from the user.
    
    This node simply adds a message requesting a topic and returns the state,
    effectively pausing for user input.
    """
    # Get the validation message from the previous node
    message = state.topic.get('message', 'Please provide a specific topic for research.')
    
    # Add the request message to the state
    new_message = AIMessage(content=message)
    new_messages = state.messages + [new_message]
    
    return {
        "messages": new_messages
    } 