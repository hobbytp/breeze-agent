# 主程序入口
import asyncio
import os
import json
from typing import List
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents.topic_validator import validate_topic, TopicValidation
from agents.outline_generator import generate_outline, Outline
from agents.topic_expander import expand_topics, RelatedTopics
from agents.perspectives_generator import generate_perspectives, Perspectives
from agents.interviewer import conduct_interviews, InterviewResult
from agents.outline_refiner import refine_outline
from agents.article_generator import generate_article

# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path='F:/AI/src/breeze-agent/src/autogen_web_research/.env')

class AppState:
    """
    Represents the state of the application.
    """
    def __init__(self):
        self.topic: str = ""
        self.topic_validation: TopicValidation = None
        self.outline: Outline = None
        self.related_topics: RelatedTopics = None
        self.perspectives: Perspectives = None
        self.interviews: List[InterviewResult] = []
        self.refined_outline: Outline = None
        self.article: str = ""
        
        # 创建 httpx.AsyncClient 时禁用 SSL 验证
        import httpx
        self.model_client = OpenAIChatCompletionClient(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="gpt-4o-mini",
            http_client=httpx.AsyncClient(verify=False) # Workaround for SSL issue
        )

app_state = AppState()

async def main():
    """
    主函数，用于演示AutoGen 0.4的基本用法。
    """
    async def run_chat():
        app_state.topic = "The Impact of AI on Modern Software Development"
        
        print(f"Starting research process for topic: '{app_state.topic}'")
        
        # 1. Validate Topic
        print("\n--- Step 1: Validating Topic ---")
        app_state.topic_validation = await validate_topic(app_state.topic, app_state.model_client)
        if not app_state.topic_validation.is_valid:
            print(f"Invalid topic: {app_state.topic_validation.message}")
            return
        print(f"Topic '{app_state.topic}' is valid.")

        # 2. Generate Outline
        print("\n--- Step 2: Generating Initial Outline ---")
        app_state.outline = await generate_outline(app_state.topic_validation.topic, app_state.model_client)
        print("Initial outline generated.")
        
        # 3. Expand Topics
        print("\n--- Step 3: Expanding Topics ---")
        app_state.related_topics = await expand_topics(app_state.topic_validation.topic, app_state.model_client)
        print(f"Related topics expanded to {app_state.related_topics.topics} topics.")
        
        # 4. Generate Perspectives
        print("\n--- Step 4: Generating Perspectives ---")
        app_state.perspectives = await generate_perspectives(
            topic=app_state.topic_validation.topic,
            related_topics=app_state.related_topics.topics,
            max_editors=3,
            model_client=app_state.model_client
        )
        print(f"Generated {len(app_state.perspectives.editors)} perspectives.")
        if len(app_state.perspectives.editors) == 0:
            print("No perspectives generated. Exiting.")
            return
        
        # 5. Conduct Interviews
        print("\n--- Step 5: Conducting Interviews ---")
        app_state.interviews = await conduct_interviews(
            perspectives=app_state.perspectives.editors,
            outline=app_state.outline.model_dump_json(indent=2),
            topic=app_state.topic_validation.topic,
            model_client=app_state.model_client,
            max_turns=2 # Keep it short for demonstration
        )
        
        # For debugging: print interview results
        print("\n--- Interview Results ---")
        for interview in app_state.interviews:
            print(f"Interview with {interview.editor_name} ({interview.persona}):")
            for turn in interview.interview_history:
                print(f"  Q: {turn.question}")
                print(f"  A: {turn.answer[:150]}...")
        
        # 6. Refine Outline
        print("\n--- Step 6: Refining Outline ---")
        # Convert interview results to a JSON string for the prompt
        conversations_str = json.dumps([res.model_dump() for res in app_state.interviews], indent=2)
        
        app_state.refined_outline = await refine_outline(
            topic=app_state.topic_validation.topic,
            old_outline=app_state.outline.model_dump_json(),
            conversations=conversations_str,
            model_client=app_state.model_client
        )
        print("Outline refined based on interviews.")
        
        # 7. Generate Article
        print("\n--- Step 7: Generating Final Article ---")
        app_state.article = await generate_article(
            topic=app_state.topic_validation.topic,
            draft=app_state.refined_outline.model_dump_json(),
            model_client=app_state.model_client
        )
        
        print("\n\n--- FINAL ARTICLE ---")
        print(app_state.article)
        print("--- END OF ARTICLE ---")

    await run_chat()
    
    # 关闭连接
    await app_state.model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
