# 智能体实现: 文章生成器
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def generate_article(topic: str, draft: str, model_client: OpenAIChatCompletionClient) -> str:
    """
    Generates an article from a draft.

    Args:
        topic: The topic of the article.
        draft: The draft of the article.
        model_client: The OpenAI model client.

    Returns:
        The generated article as a string.
    """
    article_writer_agent = AssistantAgent(
        name="article_generator",
        system_message=f"""You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:


{draft}


CRITICAL REQUIREMENTS:
1. You MUST write the COMPLETE article including ALL sections from the drafts
2. Do NOT truncate, summarize, or skip any sections
3. Do NOT use placeholders or "[Continue with...]" statements
4. Write the article in its entirety, no matter how long
5. Piece together all section drafts into a cohesive article
6. Make minor adjustments only for flow and transitions between sections
7. Ensure all specific details, examples, and depth from the original drafts are preserved
8. Maintain the comprehensive nature of each section while creating a unified whole

Strictly follow Wikipedia format guidelines.""",
        model_client=model_client,
    )

    task_result = await article_writer_agent.run(
        task='Write the COMPLETE Wiki article using markdown format. Include ALL sections. Organize citations using footnotes like "[1]",'
        " avoiding duplicates in the footer. Include URLs in the footer.",
    )
    
    return task_result.messages[-1].content

if __name__ == '__main__':
    async def test():
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        article = await generate_article(
            topic="Artificial Intelligence",
            draft="Draft content",
            model_client=model_client
        )
        print(f"Generated Article: {article}")
        await model_client.close()

    asyncio.run(test())