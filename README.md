# BREEZE (Balanced Research and Expert Engagement for Zonal Exploration)

A streamlined research system that generates comprehensive Wikipedia-style articles through multi-perspective expert engagement and focused topic exploration.

## Overview

BREEZE is inspired by STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) developed by Shao et al. While STORM focuses on broad research capabilities, BREEZE refines this approach specifically for Wikipedia-style article generation with:

- Streamlined architecture focused on article generation
- Enhanced topic validation and scoping
- Structured expert interview process
- Robust citation handling and fact-checking

## Features

- **Balanced Multi-Perspective Research**: 
  - Simulates conversations between diverse subject matter experts
  - Ensures comprehensive coverage of different viewpoints
  - Maintains neutrality in topic exploration

- **Expert Interview System**: 
  - Conducts focused interviews with AI experts
  - Gathers detailed information and citations
  - Validates information through cross-referencing

- **Structured Article Generation**: 
  - Creates well-organized articles with:
    - Clear section outlines
    - Proper citations and references
    - Consistent writing style
    - Wikipedia-style formatting

- **Zonal Topic Exploration**: 
  - Efficiently scopes and defines research boundaries
  - Maintains focus on relevant subject areas
  - Ensures appropriate depth of coverage

## How It Works

1. Topic Input and Validation
   - Submit your research topic
   - System validates and scopes the subject area
   - Establishes clear research boundaries

2. Research and Synthesis
   - Generates structured outline
   - Creates expert personas for different perspectives
   - Conducts targeted expert interviews
   - Refines outline based on gathered insights

3. Article Generation
   - Writes section drafts
   - Integrates expert insights
   - Adds proper citations
   - Delivers polished final article

## Usage

1. **Installation**
   ```bash
   pip install breeze-agent
   ```

2. **API Keys Setup**
   
   BREEZE requires the following API keys to be set as environment variables:
   - `ANTHROPIC_API_KEY`: For outline generations, expert interview interactions and section writing
   - `OPENAI_API_KEY`: For embedding model access
   - `TAVILY_API_KEY`: For web search capabilities

   You can set these using environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   export OPENAI_API_KEY="your-key-here"
   export TAVILY_API_KEY="your-key-here"
   ```

3. **Basic Usage**
   ```python
   from web_research_graph.graph import graph
   
   # Initialize research query
   result = await graph.ainvoke({
       "messages": "Your research topic here"
   })
   
   # Display results using markdown
   from IPython.display import Markdown, display
   display(Markdown(result["article"]))
   ```

4. **Output**
   The system will generate a well-structured Wikipedia-style article based on your research topic, complete with citations and multiple expert perspectives.

## Example Topics

- Technical: "Impact of Large Language Models on Software Development"
- Business: "The Rise of AI-Powered Customer Service"
- General: "History and Evolution of Electric Vehicles"

## Limitations

- Quality depends on available online sources
- May require topic refinement for very broad subjects
- Citations limited to publicly accessible sources

## Credits

BREEZE builds upon the innovative foundation laid by STORM (Shao et al.), as documented in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/tutorials/storm/storm/). We've refined their approach of outline-driven research and multi-perspective conversations while adding specialized enhancements for Wikipedia-style article generation.