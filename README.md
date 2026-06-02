# Smart Research Assistant

A stateful multi-agent research system built with LangGraph. Ask any question — the agent plans a search strategy, gathers real-time web information, reflects on gaps, and produces a structured report.

## How It Works

```
Question --> Planner --> Searcher --> Analyzer --> Reflect
                                                     |
                                         sufficient? --> Synthesizer --> Report
                                         gaps found? --> Searcher (loop, max 2x)
```

The **reflection loop** is what makes this different from a simple RAG chain. If the analysis does not fully answer the question, the agent identifies the gaps, generates new targeted queries, and researches again automatically.

## Agent Nodes

| Node | Role |
|------|------|
| **Planner** | Generates 3 targeted search queries from the question |
| **Searcher** | Fetches real-time web results via Tavily |
| **Analyzer** | Extracts key facts and insights from results |
| **Reflect** | Evaluates completeness, decides to loop or continue |
| **Synthesizer** | Writes the final structured research report |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph |
| LLM | Groq LLaMA 3.3 70B |
| Search | Tavily Search API |
| UI | Streamlit |
| Language | Python 3.11 |

## Getting Started

### Prerequisites

- Groq API key — [console.groq.com](https://console.groq.com) (free)
- Tavily API key — [tavily.com](https://tavily.com) (free, 1000 searches/month)

### Installation

```bash
git clone https://github.com/DAleid/smart-research-assistant.git
cd smart-research-assistant
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`:

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Project Structure

```
smart-research-assistant/
├── agent/
│   ├── graph.py      # LangGraph graph — nodes, edges, routing logic
│   ├── nodes.py      # Planner, Searcher, Analyzer, Reflect, Synthesizer
│   └── state.py      # Shared ResearchState TypedDict
├── app.py            # Streamlit interface
├── requirements.txt
└── .env.example
```

## Example Questions

- What are the latest advances in multimodal AI models?
- How does Retrieval-Augmented Generation improve LLM accuracy?
- What are the key differences between LangChain and LangGraph?

## License

MIT License