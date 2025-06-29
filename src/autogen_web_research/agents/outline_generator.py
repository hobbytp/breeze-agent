# 智能体实现: 大纲生成器
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field
from typing import List

class Subsection(BaseModel):
    """Represents a subsection in a Wikipedia article."""
    
    subsection_title: str = Field(
        description="The title of the subsection"
    )
    description: str = Field(
        description="The detailed content of the subsection"
    )

class Section(BaseModel):
    """Represents a section in a Wikipedia article."""
    
    section_title: str = Field(
        description="The title of the section"
    )
    description: str = Field(
        description="The main content/summary of the section"
    )
    subsections: List[Subsection] = Field(
        default_factory=list,
        description="List of subsections within this section"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of citations supporting the section content"
    )

class Outline(BaseModel):
    """Represents a complete Wikipedia-style outline."""

    page_title: str = Field(
        description="The main title of the Wikipedia article"
    )
    sections: List[Section] = Field(
        default_factory=list,
        description="List of sections that make up the article"
    )

async def generate_outline(topic: str, model_client: OpenAIChatCompletionClient) -> Outline:
    """
    Generates an outline for a given topic.

    Args:
        topic: The topic to generate an outline for.
        model_client: The OpenAI model client.

    Returns:
        An Outline object.
    """
    outline_agent = AssistantAgent(
        name="outline_generator",
        system_message="""You are a Wikipedia writer. Create a comprehensive outline for a Wikipedia page about the given topic.

Your output must be a JSON object that follows this structure:
- page_title: The main topic title
- sections: A list of sections where each section has:
  - section_title: The section heading
  - description: A detailed description of what the section will cover
  - subsections: A list of subsections where each has:
    - subsection_title: The subsection heading
    - description: Detailed description of the subsection content
  - citations: A list of citation URLs (can be empty for initial outline)

Make sure to include at least 3-5 main sections with relevant subsections.""",
        model_client=model_client,
    )

    task_result = await outline_agent.run(task=f"Create a Wikipedia outline for: {topic}")
    response_content = task_result.messages[-1].content
    
    # Manually parse the JSON response
    try:
        response_dict = json.loads(response_content)
        return Outline(**response_dict)
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the response is not valid JSON
        return Outline(page_title=topic, sections=[])


if __name__ == '__main__':
    async def test():
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        outline = await generate_outline("Artificial Intelligence", model_client)
        print(f"Generated Outline: {outline}")
        await model_client.close()

    asyncio.run(test())