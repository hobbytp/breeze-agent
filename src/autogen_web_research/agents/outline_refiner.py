# 智能体实现: 大纲优化器
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

async def refine_outline(topic: str, old_outline: str, conversations: str, model_client: OpenAIChatCompletionClient) -> Outline:
    """
    Refines an outline based on conversations with experts.

    Args:
        topic: The topic of the outline.
        old_outline: The old outline.
        conversations: The conversations with experts.
        model_client: The OpenAI model client.

    Returns:
        An Outline object.
    """
    outline_refiner_agent = AssistantAgent(
        name="outline_refiner",
        system_message=f"""You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
You need to make sure that the outline is comprehensive and specific. \
Topic you are writing about: {topic} 

Your output must be a JSON object that follows this structure:
- page_title: The main topic title
- sections: A list of sections where each section has:
  - section_title: The section heading
  - description: The section's main content
  - subsections: A list of subsections (optional) where each has:
    - subsection_title: The subsection heading
    - description: The subsection's content
  - citations: A list of citation URLs

Use the old outline as a base, enhancing it with new information from the conversations. Do not remove existing sections or subsections.

Old outline:

{old_outline}""",
        model_client=model_client,
    )

    task_result = await outline_refiner_agent.run(
        task=f"Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nProvide the refined outline following the required JSON structure."
    )
    response_content = task_result.messages[-1].content
    
    # Manually parse the JSON response
    try:
        response_dict = json.loads(response_content)
        return Outline(**response_dict)
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the response is not valid JSON
        return Outline.parse_raw(old_outline)


if __name__ == '__main__':
    async def test():
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        refined_outline = await refine_outline(
            topic="Artificial Intelligence",
            old_outline='{"page_title": "Artificial Intelligence", "sections": []}',
            conversations="Conversations with experts",
            model_client=model_client
        )
        print(f"Refined Outline: {refined_outline}")
        await model_client.close()

    asyncio.run(test())
