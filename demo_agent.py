# A paper search agent (Demo)
# Prerequisites: 
# 1. Install LangChain: pip install langchain
# 2. Install arxiv: pip install arxiv
# 3. Set up OpenAI API key: export OPENAI_API_KEY=your_api_key

import arxiv
from dataclasses import dataclass
from typing import List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy

# 1. Define tools
@tool
def search_arxiv_papers(query: str, max_results: int = 3) -> str:
    """Search for papers on arXiv based on a query string."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for paper in client.results(search):
        details = (
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(author.name for author in paper.authors)}\n"
            f"ID: {paper.entry_id.split('/')[-1]}\n"
            f"Summary: {paper.summary[:300]}...\n"
        )
        results.append(details)
    
    return "\n---\n".join(results) if results else "No papers found."

# 2. Initialize LLMs inside the agent
model = init_chat_model(
    "gpt-4o", 
    temperature=0
)

# 3. Define the system prompt for the LLM
SYSTEM_PROMPT = """You are a highly skilled academic research assistant. 
Your goal is to help users find relevant scientific papers on arXiv.

When searching:
1. Use 'search_arxiv_papers' for general keyword searches.
2. Prioritize papers that are most impactful.
3. Provide a concise summary of each paper."""


# 4. Define the response format
@dataclass
class PaperSummary:
    title: str
    arxiv_id: str
    key_takeaway: str

@dataclass
class ResponseFormat:
    assistant_message: str
    found_papers: List[PaperSummary]

# 5. Create the agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_arxiv_papers],
    response_format=ToolStrategy(ResponseFormat),
)

# --- Running the Agent ---
if __name__ == "__main__":  
    user_query = input("What do you want to search? ")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
    )

    print(f"Assistant: {response['structured_response'].assistant_message}\n")
    for paper in response['structured_response'].found_papers:
        print(f"- {paper.title} ({paper.arxiv_id})")
        print(f"  Takeaway: {paper.key_takeaway}\n")
