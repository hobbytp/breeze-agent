"""Parallel interview conductor for running multiple editor interviews simultaneously."""

import asyncio
from typing import List, Union, Dict, Any
from langchain_core.runnables import RunnableConfig

from web_research_graph.state import State, Editor, Perspectives
from web_research_graph.interviews_graph.graph import interview_graph
from web_research_graph.configuration import Configuration

async def _run_single_editor_interview(
    base_state: State, 
    editor: Editor, 
    config: RunnableConfig
) -> Dict[str, Any]:
    """为单个editor运行完整的访谈流程"""
    # 创建只包含单个editor的perspectives
    single_editor_perspectives = Perspectives(editors=[editor])
    
    # 创建单editor的状态，让interview_graph自动处理状态转换
    single_editor_state = State(
        messages=base_state.messages.copy(),
        outline=base_state.outline,
        related_topics=base_state.related_topics,
        perspectives=single_editor_perspectives,  # 只包含当前editor
        article=base_state.article,
        references=base_state.references.copy() if base_state.references else {},
        topic=base_state.topic
    )
    
    # 运行访谈图，让LangGraph自动处理State到InterviewState的转换
    return await interview_graph.ainvoke(single_editor_state, config)


def _extract_editors_from_perspectives(perspectives: Union[Perspectives, dict, None]) -> List[Editor]:
    """从perspectives中提取editors，处理不同的数据类型"""
    if not perspectives:
        raise ValueError("No perspectives found in state")
    
    # 如果是Perspectives对象， 注意，这里perspectives会一直是dict类型。
    print(f"!!! === In parallel_conduct_interviews, perspectives is type: {type(perspectives)}")
    if isinstance(perspectives, Perspectives):        
        if not perspectives.editors:
            raise ValueError("No editors found in perspectives")
        return perspectives.editors
    
    # 如果是字典
    elif isinstance(perspectives, dict):
        editors_data = perspectives.get("editors", [])
        if not editors_data:
            raise ValueError("No editors found in perspectives")
        
        # 如果editors是Editor对象列表
        if editors_data and isinstance(editors_data[0], Editor):
            return editors_data
        # 如果editors是字典列表，需要转换为Editor对象
        elif editors_data and isinstance(editors_data[0], dict):
            return [Editor(**editor_dict) for editor_dict in editors_data]
        else:
            raise ValueError("Invalid editors format in perspectives")
    
    else:
        raise ValueError(f"Invalid perspectives type: {type(perspectives)}")

async def parallel_conduct_interviews(state: State, config: RunnableConfig = None) -> State:
    """并行执行所有editor的访谈"""
    configuration = Configuration.from_runnable_config(config)
    
    # 提取editors，处理不同的数据类型
    editors: List[Editor] = _extract_editors_from_perspectives(state.perspectives)
    
    # 使用信号量控制并发数量
    semaphore = asyncio.Semaphore(configuration.max_parallel_interviews)
    
    async def _run_with_semaphore(editor: Editor):
        async with semaphore:
            return await _run_single_editor_interview(state, editor, config)
    
    # 并发执行所有访谈
    tasks = [_run_with_semaphore(editor) for editor in editors]
    results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
    
    # 汇总结果
    merged_messages = state.messages.copy()
    merged_references = state.references.copy() if state.references else {}
    
    for i, result in enumerate(results):
        # editor_name = editors[i]["name"]
        editor_name = editors[i].name
        # 添加分隔符标识不同editor的对话
        from langchain_core.messages import AIMessage
        separator = AIMessage(
            content=f"\n--- Interview with {editor_name} ---\n",
            name="system"
        )
        merged_messages.append(separator)
        
        # 正确访问字典中的messages
        if "messages" in result:
            merged_messages.extend(result["messages"])
        
        # 合并参考资料，避免URL冲突
        if "references" in result and result["references"]:
            for url, content in result["references"].items():
                # 如果URL已存在，用editor名称作为前缀避免冲突
                final_url = url if url not in merged_references else f"{editor_name}_{url}"
                merged_references[final_url] = content
    
    return State(
        messages=merged_messages,
        outline=state.outline,
        related_topics=state.related_topics,
        perspectives=state.perspectives,  # 保持原始的完整perspectives
        article=state.article,
        references=merged_references,
        topic=state.topic
    ) 