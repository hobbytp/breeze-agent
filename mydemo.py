import asyncio
import os
from dotenv import load_dotenv
from IPython.display import display, Markdown

from src.web_research_graph.graph import graph
from langchain_core.runnables import RunnableConfig

# 从 .env 文件加载环境变量
# 请确保您在项目根目录下创建了 .env 文件并填入了必要的 API 密钥
# 例如: ANTHROPIC_API_KEY="sk-..." 或 OPENAI_API_KEY="sk-..."
load_dotenv(override=True)
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")

async def main():
    """
    异步主函数，用于运行网络研究图。
    """
    # 为本次运行定义自定义参数
    # 这些将覆盖 configuration.py 中的默认设置
    my_custom_params = {
        "fast_llm_model": "openai/gpt-4o-mini",
        "long_context_model": "openai/gpt-4o",
        "max_editors": 6,
        "max_search_results": 3, # 你也可以覆盖非模型参数
        "max_turns": 5, # 每个编辑与专家的最大对话轮数
        "system_prompt": "You are a helpful assistant that can help with research tasks." # 甚至可以覆盖系统提示
    }

    # 创建一个 RunnableConfig 实例
    run_config = RunnableConfig(
        configurable=my_custom_params
    )

    # 定义图的输入
    # input_data = {"messages": [("user", "你好，请研究一下关于人工智能Multiple Agent的最新发展。")]}
    input_data = {"messages": [("user", "please study the latest developments regarding Multiple Agent in artificial intelligence.")]}
    print("正在异步调用网络研究图...")

    # 使用 'await' 等待图的异步调用完成
    # 'ainvoke' 用于异步执行
    result = await graph.ainvoke(input_data, config=run_config)

    print("\n--- 研究完成 ---")
    
    # 检查 'article' 是否在结果中并显示它
    if result and "article" in result and result["article"]:
        print("生成的文章:")
        # 在脚本中，display 可能不会渲染 Markdown，但会打印其表示
        # 在 Jupyter Notebook 中，这将呈现为格式化文本
        display(Markdown(result["article"]))
    else:
        print("未能生成文章。完整的返回结果如下:")
        print(result)

if __name__ == "__main__":
    # 运行异步主函数
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # 这是为Jupyter等可能已有正在运行的事件循环的环境准备的回退方案
        if "cannot run loop while another loop is running" in str(e):
            print("在已有事件循环的环境中运行。")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise e
