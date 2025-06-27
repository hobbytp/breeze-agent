"""
演示并行访谈功能的示例

这个示例展示了如何使用配置选项来控制访谈模式：
- 串行模式（默认）：编辑器依次进行访谈
- 并行模式：所有编辑器同时进行访谈，提升性能
"""

import asyncio
import sys
import os
import traceback

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_research_graph.graph import graph
from web_research_graph.state import State, TopicValidation
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from IPython.display import display, Markdown

async def demo_serial_interviews():
    """演示串行访谈模式（默认）"""
    print("=== 串行访谈模式演示 ===")
    
    # 串行模式配置（默认）
    config = {
        "configurable": {
            "parallel_interviews": False,  # 串行模式
            "max_turns": 2,  # 每个编辑器最多2轮对话
        }
    }
    
    # 输入状态
    input_state = {
        "messages": [],
        "topic": TopicValidation(
            is_valid=True, 
            topic="人工智能的发展历史", 
            message="有效的主题"
        ).model_dump()
    }
    
    print(f"主题: {input_state['topic']['topic']}")
    print(f"访谈模式: 串行")
    print("开始执行...")
    
    # 注意：这只是演示代码结构，实际运行需要完整的LLM配置
    # result = await graph.ainvoke(input_state, config=config)
    print("串行模式：编辑器将依次进行访谈")

async def demo_parallel_interviews():
    """演示并行访谈模式"""
    print("\n=== 并行访谈模式演示 ===")
    
    my_custom_params = {
        "fast_llm_model": "openai/gpt-4o-mini",
        "long_context_model": "openai/gpt-4o",
        "max_editors": 2,
        "max_search_results": 1,  # 你也可以覆盖非模型参数
        "parallel_interviews": True,  # 并行模式
        "max_parallel_interviews": 3,  # 最多3个并发访谈
        "max_turns": 2,  # 每个编辑器最多2轮对话
        "system_prompt": "You are a helpful assistant that can help with research tasks."  # 甚至可以覆盖系统提示
    }

    # 创建一个 RunnableConfig 实例
    run_config = RunnableConfig(
        configurable=my_custom_params
    )
    
    # 使用正确的LangChain消息格式
    topic_text = "please study the latest developments regarding Multiple Agent in artificial intelligence."
    input_data = {
        "messages": [HumanMessage(content=topic_text)]
    }
    print("正在异步调用网络研究图...")
    
    print(f"主题: {topic_text}")
    print(f"访谈模式: 并行")
    print(f"最大并发数: {my_custom_params['max_parallel_interviews']}")
    print("开始执行...")

    try:
        result = await graph.ainvoke(input=input_data, config=run_config)
        print("\n--- 研究完成 ---")
        
        # 首先检查result是否为None
        if result is None:
            print("图执行返回了None结果")
            print("这可能是因为工作流程在某个节点中断了")
            return
            
        # 检查 'article' 是否在结果中并显示它
        if "article" in result and result["article"]:
            print("\n--- 研究结果 ---")
            print("生成的文章:")
            try:
                display(Markdown(result["article"]))
            except:
                # 如果display失败，直接打印
                print(result["article"])
        else:
            print("未能生成文章。完整的返回结果如下:")
            print(f"结果类型: {type(result)}")
            print(f"结果内容: {result}")
            
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("错误详情:")
        traceback.print_exc()
        print("这是一个演示程序，实际运行需要配置API密钥等环境变量")

def compare_modes():
    """对比两种模式的特点"""
    print("\n=== 模式对比 ===")
    print("串行模式:")
    print("  ✓ 向后兼容，稳定可靠")
    print("  ✓ 资源消耗较低")
    print("  ✗ 耗时较长（编辑器依次访谈）")
    
    print("\n并行模式:")
    print("  ✓ 性能显著提升（3-5倍速度提升）")
    print("  ✓ 所有编辑器同时工作")
    print("  ✓ 可配置并发数量控制API限额")
    print("  ✗ 资源消耗较高")
    print("  ✗ 需要注意API速率限制")

async def main():
    """主函数"""
    print("🚀 并行访谈功能演示")
    print("=" * 50)
    
    # await demo_serial_interviews()
    await demo_parallel_interviews()
    # compare_modes()
    
    print("\n=== 使用建议 ===")
    print("1. 开发和测试阶段：使用串行模式（默认）")
    print("2. 生产环境需要高性能：启用并行模式")
    print("3. 根据API限额调整 max_parallel_interviews")
    print("4. 监控API使用情况，避免超出限额")

if __name__ == "__main__":
    asyncio.run(main()) 