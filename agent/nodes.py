import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from .state import ResearchState

_llm = None
_search = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def get_search() -> TavilySearchResults:
    global _search
    if _search is None:
        _search = TavilySearchResults(max_results=4)
    return _search


def planner(state: ResearchState) -> ResearchState:
    response = get_llm().invoke([
        SystemMessage(content=(
            "You are a research planner. Given a research question, generate exactly 3 "
            "specific, targeted search queries that together provide comprehensive coverage. "
            "Return ONLY a valid JSON array of 3 strings, nothing else."
        )),
        HumanMessage(content=f"Research question: {state['question']}"),
    ])
    try:
        queries = json.loads(response.content.strip())
        if not isinstance(queries, list):
            raise ValueError
    except Exception:
        queries = [state["question"]]
    return {**state, "search_queries": queries[:3]}


def searcher(state: ResearchState) -> ResearchState:
    queries = state["search_queries"]

    if state.get("iteration", 0) > 0 and state.get("reflection"):
        response = get_llm().invoke([
            SystemMessage(content=(
                "Generate 2 focused search queries to fill the identified research gaps. "
                "Return ONLY a valid JSON array of 2 strings."
            )),
            HumanMessage(
                content=f"Original question: {state['question']}\nGaps: {state['reflection']}"
            ),
        ])
        try:
            queries = json.loads(response.content.strip())[:2]
        except Exception:
            queries = [state["question"]]

    results = list(state.get("search_results", []))
    for query in queries:
        try:
            raw = get_search().invoke(query)
            for r in raw:
                snippet = f"[{r.get('url', '')}]\n{r.get('content', '')[:600]}"
                results.append(snippet)
        except Exception:
            pass

    return {
        **state,
        "search_results": results,
        "search_queries": queries,
        "iteration": state.get("iteration", 0) + 1,
    }


def analyzer(state: ResearchState) -> ResearchState:
    snippets = state.get("search_results", [])
    combined = "\n\n---\n\n".join(snippets[:10])
    if len(combined) > 8000:
        combined = combined[:8000] + "\n...[truncated]"

    response = get_llm().invoke([
        SystemMessage(content=(
            "You are a research analyst. Analyze the search results and extract key facts, "
            "insights, and information relevant to the research question. Be thorough and objective."
        )),
        HumanMessage(content=f"Question: {state['question']}\n\nSearch results:\n{combined}"),
    ])
    return {**state, "analysis": response.content}


def reflect(state: ResearchState) -> ResearchState:
    if state.get("iteration", 0) >= 2:
        return {**state, "reflection": "", "needs_more_research": False}

    response = get_llm().invoke([
        SystemMessage(content=(
            "Evaluate whether the analysis sufficiently answers the research question. "
            "Respond ONLY with valid JSON:\n"
            '{"sufficient": true, "gaps": ""}\n'
            "Set sufficient=false and describe gaps only if important information is clearly missing."
        )),
        HumanMessage(
            content=f"Question: {state['question']}\n\nAnalysis:\n{state['analysis']}"
        ),
    ])
    try:
        data = json.loads(response.content.strip())
        sufficient = bool(data.get("sufficient", True))
        gaps = str(data.get("gaps", ""))
    except Exception:
        sufficient = True
        gaps = ""

    return {**state, "reflection": gaps, "needs_more_research": not sufficient}


def synthesizer(state: ResearchState) -> ResearchState:
    response = get_llm().invoke([
        SystemMessage(content=(
            "You are a professional research writer. Write a comprehensive, well-structured "
            "research report using this exact format:\n\n"
            "# [Descriptive Title]\n\n"
            "## Executive Summary\n"
            "[2-3 sentence overview]\n\n"
            "## Key Findings\n"
            "[Bullet points of main findings]\n\n"
            "## Detailed Analysis\n"
            "[In-depth discussion with subsections if needed]\n\n"
            "## Conclusion\n"
            "[Summary and implications]"
        )),
        HumanMessage(
            content=f"Question: {state['question']}\n\nAnalysis:\n{state['analysis']}"
        ),
    ])
    return {**state, "final_report": response.content}