from langgraph.graph import StateGraph, END
from .state import ResearchState
from .nodes import planner, searcher, analyzer, reflect, synthesizer


def _route(state: ResearchState) -> str:
    if state.get("needs_more_research") and state.get("iteration", 0) < 2:
        return "searcher"
    return "synthesizer"


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner)
    workflow.add_node("searcher", searcher)
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("reflect", reflect)
    workflow.add_node("synthesizer", synthesizer)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "analyzer")
    workflow.add_edge("analyzer", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        _route,
        {"searcher": "searcher", "synthesizer": "synthesizer"},
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()