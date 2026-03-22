from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage


def web_fallback(question: str, llm) -> str:
    search_tool = DuckDuckGoSearchRun(num_results=5)
    try:
        search_results = search_tool.run(question)
        prompt = [
            SystemMessage(
                content=(
                    "You are summarizing external web results. Make it explicit this answer uses external sources, not internal knowledge base evidence. "
                    "Provide a concise summary and add a caution that local policy and official manuals should be checked."
                )
            ),
            HumanMessage(content=f"Question: {question}\n\nExternal search results:\n{search_results}"),
        ]
        return llm.invoke(prompt).content
    except Exception:
        return llm.invoke(
            [
                SystemMessage(content="Provide a cautious general answer without claiming internal evidence."),
                HumanMessage(content=question),
            ]
        ).content
