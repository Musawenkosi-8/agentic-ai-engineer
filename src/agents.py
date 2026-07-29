from langchain_groq import ChatGroq

from src.tools import (
    search_tool,
    save_research_note,
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

tools = [
    search_tool,
    save_research_note,
]

llm_with_tools = llm.bind_tools(tools)