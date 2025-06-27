#!/usr/bin/env python3
"""
可配置编辑器数量功能演示

这个脚本演示如何使用新的 max_editors 配置来控制生成的编辑器数量。
"""

import asyncio
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_research_graph.graph import graph
from web_research_graph.configuration import Configuration


async def demo_different_editor_counts():
    """演示不同编辑器数量的配置"""
    
    print("🚀 可配置编辑器数量功能演示")
    print("=" * 60)
    
    # 测试主题
    topic = "Renewable Energy Technologies"
    
    # 不同的编辑器数量配置
    test_configs = [
        {"max_editors": 1, "description": "单一视角（1个编辑器）"},
        {"max_editors": 2, "description": "双重视角（2个编辑器）"}, 
        {"max_editors": 4, "description": "多元视角（4个编辑器）"},
        {"max_editors": 5, "description": "丰富视角（5个编辑器）"}
    ]
    
    for i, config_info in enumerate(test_configs, 1):
        print(f"\n📋 测试 {i}: {config_info['description']}")
        print("-" * 50)
        
        # 创建运行配置
        run_config = {
            "configurable": {
                "max_editors": config_info["max_editors"],
                "parallel_interviews": False,  # 使用串行模式便于观察
                "max_turns": 2,  # 减少轮次以加快演示
                "fast_llm_model": "anthropic/claude-3-5-haiku-20241022"
            }
        }
        
        # 准备输入数据
        input_data = {
            "messages": [{"role": "user", "content": topic}]
        }
        
        try:
            print(f"🔧 配置: max_editors={config_info['max_editors']}")
            print(f"🎯 主题: {topic}")
            print("⏳ 生成编辑器视角...")
            
            # 运行到生成编辑器视角的步骤
            result = await graph.ainvoke(
                input=input_data, 
                config=run_config
            )
            
            # 检查生成的编辑器
            if result.get("perspectives") and hasattr(result["perspectives"], "editors"):
                editors = result["perspectives"].editors
                print(f"✅ 成功生成 {len(editors)} 个编辑器:")
                
                for j, editor in enumerate(editors, 1):
                    print(f"   {j}. {editor.name} - {editor.role}")
                    print(f"      隶属: {editor.affiliation}")
                    print(f"      描述: {editor.description[:80]}...")
                    print()
                
                # 验证编辑器数量是否符合配置
                expected_count = config_info["max_editors"]
                actual_count = len(editors)
                
                if actual_count <= expected_count:
                    print(f"✅ 编辑器数量验证通过: {actual_count} <= {expected_count}")
                else:
                    print(f"⚠️  编辑器数量超出预期: {actual_count} > {expected_count}")
            else:
                print("❌ 未能生成编辑器视角")
                
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {str(e)}")
            print("这可能是由于API配置或网络问题导致的")
            
        print("\n" + "="*60)


async def demo_configuration_usage():
    """演示配置使用的不同方式"""
    
    print("\n🛠️  配置使用方式演示")
    print("=" * 60)
    
    print("📝 方式1: 使用默认配置")
    default_config = Configuration()
    print(f"   默认编辑器数量: {default_config.max_editors}")
    
    print("\n📝 方式2: 从字典创建配置")
    custom_config = Configuration.from_runnable_config({
        "configurable": {
            "max_editors": 6,
            "parallel_interviews": True,
            "max_turns": 5
        }
    })
    print(f"   自定义编辑器数量: {custom_config.max_editors}")
    print(f"   并行访谈: {custom_config.parallel_interviews}")
    print(f"   最大轮次: {custom_config.max_turns}")
    
    print("\n📝 方式3: 在图调用中直接配置")
    example_config = {
        "configurable": {
            "max_editors": 2,
            "fast_llm_model": "anthropic/claude-3-5-haiku-20241022",
            "parallel_interviews": False
        }
    }
    print(f"   运行时配置示例: {example_config}")
    
    print("\n📝 方式4: 不同场景的推荐配置")
    scenarios = [
        {
            "name": "快速原型",
            "config": {"max_editors": 1, "max_turns": 2},
            "description": "单一视角，快速验证想法"
        },
        {
            "name": "平衡质量",
            "config": {"max_editors": 3, "max_turns": 3},
            "description": "默认配置，平衡质量和效率"
        },
        {
            "name": "深度研究",
            "config": {"max_editors": 5, "max_turns": 4, "parallel_interviews": True},
            "description": "多视角深度分析，适合复杂主题"
        },
        {
            "name": "全面覆盖",
            "config": {"max_editors": 8, "max_turns": 5, "parallel_interviews": True},
            "description": "最全面的视角覆盖，适合重要主题"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n   🎯 {scenario['name']}:")
        print(f"      配置: {scenario['config']}")
        print(f"      说明: {scenario['description']}")


def main():
    """主函数"""
    print("🌟 欢迎使用可配置编辑器数量功能！")
    print("这个功能允许你根据需要调整生成的编辑器数量，")
    print("从而控制研究的深度和广度。\n")
    
    # 选择演示模式
    print("请选择演示模式:")
    print("1. 完整演示（需要API配置）")
    print("2. 仅配置演示（无需API）")
    
    try:
        choice = input("\n请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            print("\n🚀 开始完整演示...")
            asyncio.run(demo_different_editor_counts())
        elif choice == "2":
            print("\n🛠️  开始配置演示...")
            asyncio.run(demo_configuration_usage())
        else:
            print("❌ 无效选择，运行配置演示...")
            asyncio.run(demo_configuration_usage())
            
    except KeyboardInterrupt:
        print("\n\n👋 演示已取消")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("\n🛠️  运行配置演示作为备选...")
        asyncio.run(demo_configuration_usage())


if __name__ == "__main__":
    main() 