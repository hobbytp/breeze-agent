# 智能体实现: 主题扩展器
import asyncio
import json
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field
from typing import List

class RelatedTopics(BaseModel):
    """Represents related topics for research."""
    
    topics: List[str] = Field(
        description="List of related topics that are relevant to the main research subject"
    )

def _extract_json(text: str) -> dict:
    """从可能包含额外文本的字符串中提取JSON对象。"""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果没有找到代码块或者解析失败，尝试直接解析整个文本
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to decode JSON from response: {text}")
            return {}

async def expand_topics(topic: str, model_client: OpenAIChatCompletionClient) -> RelatedTopics:
    """
    Expands a given topic to find related topics.

    Args:
        topic: The topic to expand.
        model_client: The OpenAI model client.

    Returns:
        A RelatedTopics object.
    """
    topic_expander_agent = AssistantAgent(
        name="topic_expander",
        system_message="""I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

Please return a JSON object with a single key "topics" which is a list of strings. Ensure the output is only the raw JSON object.""",
        model_client=model_client,
    )

    # 使用 on_messages 进行非阻塞调用
    message = TextMessage(content=f"Topic of interest: {topic}", source="user")
    response_message = await topic_expander_agent.on_messages(
        messages=[message],
        cancellation_token=CancellationToken()
    )
    response_content = response_message.chat_message.to_text()
    
    # 使用更健壮��JSON提取和解析
    response_dict = _extract_json(response_content)
    if response_dict:
        return RelatedTopics(**response_dict)
    else:
        return RelatedTopics(topics=[])


if __name__ == '__main__':
    async def test():
        # 需要设置OPENAI_API_KEY环境变量
        # from dotenv import load_dotenv
        # load_dotenv()
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        related_topics = await expand_topics("Artificial Intelligence", model_client)
        print(f"Related Topics: {related_topics}")
        await model_client.close()

    asyncio.run(test())
