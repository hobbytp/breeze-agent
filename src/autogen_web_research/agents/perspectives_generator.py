# 智能体实现: 视角生成器
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field
from typing import List
from .wikipedia_search import search_wikipedia_examples

class Editor(BaseModel):
    """Represents a Wikipedia editor with specific expertise."""
    
    affiliation: str
    name: str
    role: str
    description: str

    @property
    def persona(self) -> str:
        """Return a formatted string representation of the editor."""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    """Represents a group of editors with different perspectives."""
    
    editors: List[Editor] = Field(default_factory=list)

async def generate_perspectives(topic: str, related_topics: List[str], max_editors: int, model_client: OpenAIChatCompletionClient) -> Perspectives:
    """
    Generates different perspectives for a given topic.

    Args:
        topic: The topic to generate perspectives for.
        related_topics: List of related topics to search for Wikipedia examples.
        max_editors: The maximum number of editors to generate.
        model_client: The OpenAI model client.

    Returns:
        A Perspectives object.
    """
    # 搜索相关主题的Wikipedia内容作为示例
    print(f"Searching Wikipedia for related topics: {related_topics}")
    examples = await search_wikipedia_examples(related_topics)
    print(f"Found {len(examples)} characters of Wikipedia content")
    
    perspectives_agent = AssistantAgent(
        name="perspectives_generator",
        system_message=f"""You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.
        You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on. Select up to {max_editors} editors.

        Wiki page outlines of related topics for inspiration:
        {examples}
        
        IMPORTANT: You MUST respond with ONLY a valid JSON object in the following format. Do not include any explanations, markdown formatting, or additional text:

        {{
            "editors": [
                {{
                    "name": "Editor Name",
                    "role": "Their role/expertise",
                    "affiliation": "Their affiliation/organization", 
                    "description": "What they will focus on"
                }}
            ]
        }}
        
        Return only the JSON object, nothing else.""",
        model_client=model_client,
    )

    task_result = await perspectives_agent.run(task=f"Topic of interest: {topic}")
    response_content = task_result.messages[-1].content
    
    # 调试：打印AI的原始响应
    print(f"AI Response for perspectives generation:")
    print(f"Response length: {len(response_content)}")
    print(f"Response content: {response_content[:500]}...")  # 只显示前500字符
    
    # Manually parse the JSON response
    try:
        # 尝试找到JSON部分
        if "```json" in response_content:
            # 提取JSON代码块
            json_start = response_content.find("```json") + 7
            json_end = response_content.find("```", json_start)
            json_content = response_content[json_start:json_end].strip()
        elif "{" in response_content:
            # 尝试提取JSON对象
            json_start = response_content.find("{")
            json_end = response_content.rfind("}") + 1
            json_content = response_content[json_start:json_end]
        else:
            json_content = response_content
            
        print(f"Extracted JSON: {json_content[:300]}...")
        
        response_dict = json.loads(json_content)
        result = Perspectives(**response_dict)
        print(f"Successfully parsed {len(result.editors)} editors")
        return result
    except (json.JSONDecodeError, TypeError) as e:
        # Handle cases where the response is not valid JSON
        print(f"JSON parsing failed: {e}")
        print(f"Failed to parse content: {response_content}")
        return Perspectives(editors=[])


if __name__ == '__main__':
    async def test():
        import httpx
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            http_client=httpx.AsyncClient(verify=False)
        )
        perspectives = await generate_perspectives(
            topic="Artificial Intelligence",
            related_topics=["Machine Learning", "Deep Learning", "Natural Language Processing"],
            max_editors=3,
            model_client=model_client
        )
        print(f"Generated Perspectives: {perspectives}")
        await model_client.close()

    asyncio.run(test())
