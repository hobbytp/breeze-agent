# 智能体实现: 主题验证器
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field
from typing import Optional

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

async def validate_topic(topic: str, model_client: OpenAIChatCompletionClient) -> TopicValidation:
    """
    Validates the user's topic.

    Args:
        topic: The user's input topic.
        model_client: The OpenAI model client.

    Returns:
        A TopicValidation object.
    """
    validator_agent = AssistantAgent(
        name="topic_validator",
        system_message="""You are a helpful assistant whose job is to ensure the user provides a clear topic for research.
        Analyze the user's input and determine if it contains a clear research topic.
        
        Example valid topics:
        - "Artificial Intelligence"
        - "The French Revolution"
        - "Quantum Computing"
        
        Return a JSON object with the following keys:
        - is_valid: true if a clear topic is provided, false otherwise
        - topic: the extracted topic if valid, null otherwise
        - message: a helpful message if the input is invalid, null otherwise
        
        For invalid inputs or small talk, provide a polite message asking for a specific topic.""",
        model_client=model_client,
    )

    task_result = await validator_agent.run(task=f"Topic: {topic}")
    response_content = task_result.messages[-1].content
    
    # Manually parse the JSON response
    try:
        response_dict = json.loads(response_content)
        return TopicValidation(**response_dict)
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the response is not valid JSON
        return TopicValidation(is_valid=False, topic=topic, message="The model did not return a valid JSON response.")


if __name__ == '__main__':
    async def test():
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        valid_topic = await validate_topic("Artificial Intelligence", model_client)
        print(f"Valid topic validation: {valid_topic}")
        invalid_topic = await validate_topic("hey how are you", model_client)
        print(f"Invalid topic validation: {invalid_topic}")
        await model_client.close()

    asyncio.run(test())