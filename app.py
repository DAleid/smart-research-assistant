import os
import streamlit as st
from dotenv import load_dotenv
from agent import build_graph

load_dotenv()

st.set_page_config(
    page_title="Smart Research Assistant",
    layout="wide",
)

st.title("Smart Research Assistant")
st.caption("Stateful multi-agent research — LangGraph · Groq LLaMA 3.3 · Tavily Search")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
**Agent graph:**

```
Question
   |
Planner       generates search queries
   |
Searcher      fetches real-time results
   |
Analyzer      extracts key insights
   |
Reflect       checks completeness
   |
sufficient? ---> Synthesizer --> Report
not enough? ---> Searcher (loop, max 2x)
```
    """)
    st.divider()
    st.markdown("**Models**")
    st.markdown("- LLM: Groq LLaMA 3.3 70B\n- Search: Tavily real-time web")

question = st.text_area(
    "Research question",
    placeholder="e.g. What are the latest advances in multimodal AI models?",
    height=90,
)

if st.button("Research", type="primary", disabled=not question.strip()):
    missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        st.error(f"Missing environment variables: {', '.join(missing)}. Check your .env file.")
        st.stop()

    graph = build_graph()
    initial_state = {
        "question": question.strip(),
        "search_queries": [],
        "search_results": [],
        "analysis": "",
        "reflection": "",
        "needs_more_research": False,
        "final_report": "",
        "iteration": 0,
    }

    node_labels = {
        "planner": "Planning search strategy",
        "searcher": "Searching the web",
        "analyzer": "Analyzing results",
        "reflect": "Reflecting on completeness",
        "synthesizer": "Writing final report",
    }

    progress = st.progress(0)
    status = st.empty()
    steps_done = []
    final_state = initial_state

    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        final_state = step[node_name]
        steps_done.append(node_name)
        status.info(f"{node_labels.get(node_name, node_name)}...")
        progress.progress(min(len(steps_done) / 6, 0.95))

    progress.progress(1.0)
    status.success("Research complete.")

    st.divider()
    st.markdown(final_state.get("final_report", "No report generated."))

    with st.expander("Search queries used"):
        for q in final_state.get("search_queries", []):
            st.markdown(f"- {q}")

    with st.expander("Raw analysis"):
        st.write(final_state.get("analysis", ""))

    with st.expander("Reflection"):
        r = final_state.get("reflection", "")
        st.write(r if r else "Analysis was sufficient on first pass.")