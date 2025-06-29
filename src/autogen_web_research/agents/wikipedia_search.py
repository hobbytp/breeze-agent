# Wikipedia搜索工具
import aiohttp
import asyncio
from typing import List, Dict, Any
from urllib.parse import quote

class WikipediaDocument:
    """表示一个Wikipedia文档"""
    
    def __init__(self, title: str, content: str, categories: List[str] = None):
        self.title = title
        self.content = content
        self.categories = categories or []
        self.metadata = {
            "title": title,
            "categories": categories or []
        }

class WikipediaSearcher:
    """Wikipedia搜索器"""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    
    async def search_topic(self, topic: str) -> WikipediaDocument:
        """搜索单个主题"""
        url = self.base_url + quote(topic.replace(" ", "_"))
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        title = data.get("title", topic)
                        extract = data.get("extract", "")
                        
                        # 获取类别信息（这里简化处理）
                        categories = []
                        
                        return WikipediaDocument(
                            title=title,
                            content=extract,
                            categories=categories
                        )
                    else:
                        # 如果搜索失败，返回空文档
                        return WikipediaDocument(
                            title=topic,
                            content=f"Wikipedia article on {topic} not found.",
                            categories=[]
                        )
            except Exception as e:
                # 错误处理，返回空文档
                return WikipediaDocument(
                    title=topic,
                    content=f"Error searching for {topic}: {str(e)}",
                    categories=[]
                )
    
    async def search_topics(self, topics: List[str]) -> List[WikipediaDocument]:
        """并行搜索多个主题"""
        tasks = [self.search_topic(topic) for topic in topics]
        return await asyncio.gather(*tasks)

def format_doc(doc: WikipediaDocument, max_length: int = 1000) -> str:
    """格式化Wikipedia文档"""
    related = "- ".join(doc.categories) if doc.categories else "No categories"
    formatted = f"### {doc.title}\n\nSummary: {doc.content}\n\nRelated\n{related}"
    return formatted[:max_length]

def format_docs(docs: List[WikipediaDocument]) -> str:
    """格式化多个Wikipedia文档"""
    return "\n\n".join(format_doc(doc) for doc in docs)

async def search_wikipedia_examples(topics: List[str]) -> str:
    """搜索Wikipedia示例并格式化"""
    searcher = WikipediaSearcher()
    docs = await searcher.search_topics(topics)
    return format_docs(docs) 