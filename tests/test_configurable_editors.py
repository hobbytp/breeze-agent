#!/usr/bin/env python3
"""
测试可配置编辑器数量功能

验证 max_editors 配置项是否正常工作：
1. 测试默认值（3个编辑器）
2. 测试自定义值（1个、2个、5个编辑器）
3. 验证配置正确传递到提示模板
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_research_graph.configuration import Configuration
from web_research_graph.prompts import PERSPECTIVES_PROMPT

def test_configuration_defaults():
    """测试配置的默认值"""
    print("🔧 测试1: 配置默认值")
    print("=" * 50)
    
    config = Configuration()
    print(f"✅ 默认 max_editors: {config.max_editors}")
    print(f"✅ 默认 max_turns: {config.max_turns}")
    print(f"✅ 默认 max_search_results: {config.max_search_results}")
    
    assert config.max_editors == 3, f"Expected max_editors=3, got {config.max_editors}"
    print("✅ 默认值验证通过")

def test_configuration_from_runnable_config():
    """测试从RunnableConfig创建配置"""
    print("\n🔧 测试2: 从RunnableConfig创建配置")
    print("=" * 50)
    
    # 测试自定义配置
    runnable_configs = [
        {"configurable": {"max_editors": 1}},
        {"configurable": {"max_editors": 2}},
        {"configurable": {"max_editors": 5}},
        {"configurable": {"max_editors": 1, "max_turns": 5, "parallel_interviews": True}},
    ]
    
    for i, runnable_config in enumerate(runnable_configs, 1):
        config = Configuration.from_runnable_config(runnable_config)
        expected_editors = runnable_config["configurable"]["max_editors"]
        
        print(f"✅ 测试配置 {i}: max_editors={expected_editors}")
        print(f"   实际值: {config.max_editors}")
        
        assert config.max_editors == expected_editors, \
            f"Expected max_editors={expected_editors}, got {config.max_editors}"
    
    print("✅ RunnableConfig配置验证通过")

def test_prompt_template():
    """测试提示模板是否正确使用max_editors参数"""
    print("\n🔧 测试3: 提示模板参数传递")
    print("=" * 50)
    
    # 测试不同的编辑器数量
    test_cases = [1, 2, 3, 5, 10]
    
    for max_editors in test_cases:
        # 格式化提示模板
        formatted_prompt = PERSPECTIVES_PROMPT.format_messages(
            examples="Sample Wikipedia content...",
            topic="Artificial Intelligence",
            max_editors=max_editors
        )
        
        # 检查系统消息中是否包含正确的编辑器数量
        system_message = formatted_prompt[0].content
        expected_text = f"Select up to {max_editors} editors"
        
        print(f"✅ 测试 max_editors={max_editors}")
        print(f"   提示中包含: '{expected_text}'")
        
        assert expected_text in system_message, \
            f"Expected '{expected_text}' in prompt, but got: {system_message[:200]}..."
    
    print("✅ 提示模板参数传递验证通过")

def test_edge_cases():
    """测试边界情况"""
    print("\n🔧 测试4: 边界情况")
    print("=" * 50)
    
    # 测试边界值
    edge_cases = [
        {"max_editors": 0, "description": "零个编辑器"},
        {"max_editors": 1, "description": "一个编辑器"},
        {"max_editors": 100, "description": "大量编辑器"},
    ]
    
    for case in edge_cases:
        config = Configuration.from_runnable_config({
            "configurable": {"max_editors": case["max_editors"]}
        })
        
        print(f"✅ {case['description']}: max_editors={config.max_editors}")
        assert config.max_editors == case["max_editors"]
    
    print("✅ 边界情况验证通过")

def demo_configuration_usage():
    """演示配置使用示例"""
    print("\n🚀 配置使用示例")
    print("=" * 50)
    
    print("📝 示例1: 默认配置（3个编辑器）")
    default_config = Configuration()
    print(f"   max_editors: {default_config.max_editors}")
    
    print("\n📝 示例2: 自定义配置（5个编辑器）")
    custom_config = Configuration.from_runnable_config({
        "configurable": {
            "max_editors": 5,
            "parallel_interviews": True,
            "max_turns": 4
        }
    })
    print(f"   max_editors: {custom_config.max_editors}")
    print(f"   parallel_interviews: {custom_config.parallel_interviews}")
    print(f"   max_turns: {custom_config.max_turns}")
    
    print("\n📝 示例3: 提示模板格式化")
    formatted = PERSPECTIVES_PROMPT.format_messages(
        examples="Example content...",
        topic="Climate Change",
        max_editors=custom_config.max_editors
    )
    print(f"   系统提示片段: {formatted[0].content[:100]}...")

def main():
    """运行所有测试"""
    print("🚀 开始测试可配置编辑器数量功能\n")
    
    try:
        test_configuration_defaults()
        test_configuration_from_runnable_config()
        test_prompt_template()
        test_edge_cases()
        demo_configuration_usage()
        
        print("\n" + "="*70)
        print("🎉 所有测试通过！可配置编辑器数量功能正常工作")
        print("="*70)
        
        print("\n📊 功能总结:")
        print("✅ 添加了 max_editors 配置项（默认值：3）")
        print("✅ 支持通过 RunnableConfig 动态配置编辑器数量")
        print("✅ 提示模板正确使用 max_editors 参数")
        print("✅ 配置验证和边界情况处理正常")
        print("✅ 保持向后兼容性（默认行为不变）")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    main() 