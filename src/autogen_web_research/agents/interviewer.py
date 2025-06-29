#包含使用 AutoGen 实现并行访谈逻辑的模块。
import asyncio
from typing import List, Dict

from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from .perspectives_generator import Editor
# 假设我们有一个搜索工具
# from ..tools import search 

# --- 数据结构 ---

class InterviewTurn(BaseModel):
    """记录一轮问答。"""
    question: str = Field(description="访谈者提出的问题")
    answer: str = Field(description="专家的回答")
    references: Dict[str, str] = Field(default_factory=dict, description="回答所引用的资料")

class InterviewResult(BaseModel):
    """记录单个专家的完整访谈结果。"""
    editor_name: str = Field(description="专家的姓名或标识")
    persona: str = Field(description="专家的角色设定")
    interview_history: List[InterviewTurn] = Field(default_factory=list, description="访谈的完整历史记录")

# --- 核心函数 ---

async def _run_single_interview(
    editor_agent: AssistantAgent,
    expert_agent: AssistantAgent,
    outline: str,
    topic: str,
    max_turns: int,
    original_editor: Editor
) -> InterviewResult:
    """
    与单个专家（editor）进行完整的访谈。
    """
    print(f"--- Starting interview with {editor_agent.name} ---")
    
    # 使用原始编辑器的信息
    persona = original_editor.persona
    
    interview_result = InterviewResult(editor_name=original_editor.name, persona=persona)
    
    # 构建初始对话历史，为提问提供上下文
    conversation_history = f"You are a journalist with the persona: '{persona}'.\n"
    conversation_history += f"You are interviewing an expert about the topic: '{topic}'.\n"
    conversation_history += f"Here is the article outline you should base your questions on:\n{outline}\n\n"
    conversation_history += "Ask insightful questions to uncover unique perspectives for the article. Ask one question at a time."

    for turn in range(max_turns):
        print(f"  - {editor_agent.name} | Turn {turn + 1}/{max_turns}")
        
        # 1. 编辑提出问题
        # 为了让提问更有针对性，我们将完整的历史传递给它
        question_task = f"Based on the conversation so far, ask your next single, specific question.\n\nFull Conversation History:\n{conversation_history}"
        
        question_response = await editor_agent.run(task=question_task)
        question = question_response.messages[-1].content
        
        # 更新对话历史
        conversation_history += f"\n\nQuestion from {editor_agent.name}: {question}"
        print(f"    > Question: {question}")

        # 2. TODO: 实现搜索逻辑
        # search_results = await search(question)
        # context = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in search_results])
        # references = {r['url']: r['content'] for r in search_results}
        context = "No search context available in this version."
        references = {}

        # 3. 专家回答问题
        answer_task = f"Please answer the following question based on your expertise and the provided context.\n\nQuestion: {question}\n\nContext:\n{context}"
        answer_response = await expert_agent.run(task=answer_task)
        answer = answer_response.messages[-1].content

        # 更新对话历史
        conversation_history += f"\n\nAnswer from Expert: {answer}"
        print(f"    > Answer: {answer[:100]}...")

        # 4. 记录这一轮的问答
        interview_result.interview_history.append(
            InterviewTurn(question=question, answer=answer, references=references)
        )

    print(f"--- Finished interview with {editor_agent.name} ---")
    return interview_result


async def conduct_interviews(
    perspectives: List[Editor],
    outline: str,
    topic: str,
    model_client: OpenAIChatCompletionClient,
    max_turns: int = 3,
    max_parallel_interviews: int = 3,
) -> List[InterviewResult]:
    """
    使用 AutoGen 并行执行多视角访谈。

    Args:
        perspectives: 从 AppState.perspectives.editors 传入的编辑者对象列表。
        outline: 文章大纲。
        topic: 主题。
        model_client: OpenAI 模型客户端。
        max_turns: 每个访谈的最大轮次。
        max_parallel_interviews: 最大并行访谈数。

    Returns:
        一个包含所有访谈结果的列表。
    """
    # 1. 创建一个固定的“回答者”智能体
    expert_agent = AssistantAgent(
        name="Expert_Answerer",
        system_message="You are a world-class researcher and expert on any topic. Answer the questions based on your expertise and any provided context. Be concise, clear, and insightful.",
        model_client=model_client,
    )

    # 2. 为每个视角创建一个“提问者”智能体
    editor_agents = [
        AssistantAgent(
            name=p.name.replace(' ', '_').replace('.', '').replace('-', '_'),
            system_message=p.persona, # 将 persona 直接作为 system_message
            model_client=model_client,
        ) for p in perspectives
    ]

    # 3. 定义带信号量的单个访谈任务
    semaphore = asyncio.Semaphore(max_parallel_interviews)
    async def _run_with_semaphore(editor_agent: AssistantAgent, original_editor: Editor):
        async with semaphore:
            return await _run_single_interview(
                editor_agent=editor_agent,
                expert_agent=expert_agent,
                outline=outline,
                topic=topic,
                max_turns=max_turns,
                original_editor=original_editor
            )

    # 4. 并行执行所有访谈
    print(f"\nConducting {len(editor_agents)} interviews in parallel (max {max_parallel_interviews})...")
    tasks = [_run_with_semaphore(agent, original_editor) for agent, original_editor in zip(editor_agents, perspectives)]
    all_results = await asyncio.gather(*tasks)
    print("\nAll interviews complete.")

    return all_results
