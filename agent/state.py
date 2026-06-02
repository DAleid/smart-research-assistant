from typing import TypedDict, List


class ResearchState(TypedDict):
    question: str
    search_queries: List[str]
    search_results: List[str]
    analysis: str
    reflection: str
    needs_more_research: bool
    final_report: str
    iteration: int