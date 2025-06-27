"""Node for handling invalid topics and waiting for user input."""

from typing import Dict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import State

async def request_topic(state: State, config: RunnableConfig) -> Dict:
    """Request a new topic from the user."""
    # 处理TopicValidation对象或字典
    print(f"!!! === state.topic is type: {type(state.topic)}")
    if hasattr(state.topic, 'message'):
        print(f"!!! === state.topic hasattr message")
        # TopicValidation对象
        message = state.topic.message or 'Please provide a specific topic for research.'
    elif isinstance(state.topic, dict):
        print(f"!!! === state.topic is dict")
        # 字典格式
        message = state.topic.get('message', 'Please provide a specific topic for research.')
    else:
        print(f"!!! === state.topic is default")        # 默认消息
        message = 'Please provide a specific topic for research.'
    
    new_message = AIMessage(content=message)
    new_messages = state.messages + [new_message]
    
    return {
        "messages": new_messages
    } 