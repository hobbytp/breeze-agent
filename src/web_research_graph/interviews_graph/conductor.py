"""Smart interview conductor that chooses between serial and parallel interview modes."""

from langchain_core.runnables import RunnableConfig

from web_research_graph.state import State
from web_research_graph.configuration import Configuration
from web_research_graph.interviews_graph.graph import interview_graph
from web_research_graph.interviews_graph.parallel_conductor import parallel_conduct_interviews

async def conduct_interviews(state: State, config: RunnableConfig = None) -> State:
    """根据配置选择串行或并行访谈模式"""
    configuration = Configuration.from_runnable_config(config)
    
    if configuration.parallel_interviews:
        # 并行模式
        return await parallel_conduct_interviews(state, config)
    else:
        # 串行模式（原有逻辑）
        interview_state = await interview_graph.ainvoke(state, config)
        
        # 将InterviewState转换回State
        return State(
            messages=state.messages + interview_state.messages,
            outline=state.outline,
            related_topics=state.related_topics,
            perspectives=state.perspectives,
            article=state.article,
            references={**(state.references or {}), **(interview_state.references or {})},
            topic=state.topic,
            all_conversations=None  # 串行模式下为None，让工具函数从messages解析
        ) 