#!/usr/bin/env python3
"""
测试Section模型验证修复

验证Section和Outline模型在缺少某些字段时不会抛出验证错误
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_research_graph.state import Section, Subsection, Outline

def test_section_creation():
    """测试Section对象的创建"""
    print("🧪 测试1: Section对象创建")
    print("=" * 50)
    
    # 测试完整Section创建
    print("📝 测试完整Section创建...")
    complete_section = Section(
        section_title="Introduction",
        description="This is the introduction section.",
        subsections=[
            Subsection(
                subsection_title="Background",
                description="Background information"
            )
        ],
        citations=["https://example.com"]
    )
    print(f"✅ 完整Section创建成功: {complete_section.section_title}")
    
    # 测试缺少subsections的Section创建（修复前会失败）
    print("\n📝 测试缺少subsections的Section创建...")
    try:
        minimal_section = Section(
            section_title="Conclusion",
            description="This is the conclusion section."
            # 注意：没有提供subsections和citations
        )
        print(f"✅ 最小Section创建成功: {minimal_section.section_title}")
        print(f"   默认subsections: {minimal_section.subsections}")
        print(f"   默认citations: {minimal_section.citations}")
    except Exception as e:
        print(f"❌ 最小Section创建失败: {e}")
        return False
    
    # 测试从字典创建Section（模拟LLM输出）
    print("\n📝 测试从字典创建Section...")
    try:
        # 模拟LLM可能返回的不完整字典
        section_dict = {
            "section_title": "Methods",
            "description": "Research methodology section."
            # 缺少subsections字段
        }
        
        dict_section = Section(**section_dict)
        print(f"✅ 从字典创建Section成功: {dict_section.section_title}")
        print(f"   自动补充的subsections: {dict_section.subsections}")
    except Exception as e:
        print(f"❌ 从字典创建Section失败: {e}")
        return False
    
    return True

def test_outline_creation():
    """测试Outline对象的创建"""
    print("\n🧪 测试2: Outline对象创建")
    print("=" * 50)
    
    # 测试完整Outline创建
    print("📝 测试完整Outline创建...")
    complete_outline = Outline(
        page_title="Test Article",
        sections=[
            Section(
                section_title="Introduction",
                description="Introduction section"
            )
        ]
    )
    print(f"✅ 完整Outline创建成功: {complete_outline.page_title}")
    
    # 测试缺少sections的Outline创建
    print("\n📝 测试缺少sections的Outline创建...")
    try:
        minimal_outline = Outline(
            page_title="Empty Article"
            # 没有提供sections
        )
        print(f"✅ 最小Outline创建成功: {minimal_outline.page_title}")
        print(f"   默认sections: {minimal_outline.sections}")
    except Exception as e:
        print(f"❌ 最小Outline创建失败: {e}")
        return False
    
    return True

def test_structured_output_simulation():
    """模拟LLM结构化输出的情况"""
    print("\n🧪 测试3: 模拟LLM结构化输出")
    print("=" * 50)
    
    # 模拟LLM可能返回的各种不完整数据
    test_cases = [
        {
            "name": "完全缺少subsections",
            "data": {
                "section_title": "Analysis",
                "description": "Data analysis section."
            }
        },
        {
            "name": "空的subsections",
            "data": {
                "section_title": "Results", 
                "description": "Research results.",
                "subsections": []
            }
        },
        {
            "name": "只有citations",
            "data": {
                "section_title": "References",
                "description": "Reference list.",
                "citations": ["https://example1.com", "https://example2.com"]
            }
        },
        {
            "name": "完全最小",
            "data": {
                "section_title": "Summary",
                "description": "Article summary."
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        try:
            section = Section(**test_case['data'])
            print(f"   ✅ 创建成功: {section.section_title}")
            print(f"   📊 subsections数量: {len(section.subsections)}")
            print(f"   📊 citations数量: {len(section.citations)}")
        except Exception as e:
            print(f"   ❌ 创建失败: {e}")
            return False
    
    return True

def test_as_str_methods():
    """测试字符串转换方法"""
    print("\n🧪 测试4: 字符串转换方法")
    print("=" * 50)
    
    # 创建测试对象
    section = Section(
        section_title="Test Section",
        description="This is a test section."
    )
    
    outline = Outline(
        page_title="Test Article",
        sections=[section]
    )
    
    print("📝 测试Section.as_str...")
    section_str = section.as_str
    print(f"✅ Section字符串转换成功 ({len(section_str)} 字符)")
    print(f"   预览: {section_str[:100]}...")
    
    print("\n📝 测试Outline.as_str...")
    outline_str = outline.as_str
    print(f"✅ Outline字符串转换成功 ({len(outline_str)} 字符)")
    print(f"   预览: {outline_str[:100]}...")
    
    return True

def main():
    """运行所有测试"""
    print("🚀 开始测试Section模型验证修复\n")
    
    tests = [
        test_section_creation,
        test_outline_creation,
        test_structured_output_simulation,
        test_as_str_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ 测试 {test_func.__name__} 失败")
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "="*70)
    if passed == total:
        print("🎉 所有测试通过！Section模型验证修复成功")
        print("="*70)
        
        print("\n📊 修复总结:")
        print("✅ Section.subsections 现在有默认值 (空列表)")
        print("✅ Section.citations 已有默认值 (空列表)")
        print("✅ Outline.sections 现在有默认值 (空列表)")
        print("✅ 支持从不完整的字典创建对象")
        print("✅ 兼容LLM结构化输出的各种情况")
        print("✅ 字符串转换方法正常工作")
        
        print("\n🔧 这个修复解决了:")
        print("- LLM生成结构化输出时缺少某些字段的验证错误")
        print("- Pydantic模型在部分数据输入时的失败")
        print("- 提高了系统对不完整输入的容错性")
        
    else:
        print(f"❌ {total - passed}/{total} 测试失败")
        print("="*70)

if __name__ == "__main__":
    main() 