# 🤖 AI Agent Learning

A hands-on project exploring AI agents from basic concepts to production-grade multi-agent systems using LangChain, LangGraph, and AWS Bedrock.

## 📋 Overview

This repository demonstrates the evolution of AI agent architectures:

1. **Agent Loop Basics** - Understanding the fundamental agent loop with raw AWS Bedrock
2. **LangChain Agent** - Single agent with tools using LangChain's `create_agent()`
3. **LangGraph Multi-Agent** - Multi-agent orchestration with supervisor pattern
4. **Production Chatbot** - Full-featured Streamlit app with Knowledge Graph integration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query (Streamlit)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     🎯 SUPERVISOR (Claude Haiku)                     │
│              Analyzes query → Routes to specialist agents            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ ⚡ Energy      │ │ 🔗 Knowledge  │ │ 📦 Order      │
        │    Agent      │ │    Graph      │ │    Agent      │
        │               │ │    Agent      │ │               │
        │ • get_price   │ │ • kg_markets  │ │ • get_status  │
        │ • forecast    │ │ • kg_plants   │ │ • track       │
        └───────────────┘ └───────────────┘ └───────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   📝 FINAL RESPONSE SYNTHESIZER                      │
│                  Combines results into natural language              │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai-agent-learning/
├── agent_loop_basics/
│   └── agent_loop_basis.py      # Raw Bedrock agent loop implementation
├── langchain_agent/
│   ├── langchain_agent.py       # Single LangChain agent with tools
│   └── langchain_multi_agent.py # Basic multi-agent exploration
├── langraph_agent/
│   └── langraph_agent.py        # LangGraph multi-agent with supervisor
├── chatbot/
│   ├── agent.py                 # Production multi-agent system
│   └── app.py                   # Streamlit UI
├── docs/
│   ├── langchain-vs-langgraph.svg
│   └── multi-agent-architecture.svg
├── .env                         # Environment variables
├── settings.py                  # Configuration
├── pyproject.toml               # Dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- AWS credentials with Bedrock access
- Neo4j database (optional, for Knowledge Graph features)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai-agent-learning

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Environment Setup

Create a `.env` file:

```env
AWS_PROFILE=your_aws_profile
AWS_REGION=us-east-1

# Optional: Neo4j for Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Running the Chatbot

```bash
# Start the Streamlit app
uv run streamlit run chatbot/app.py
```

Open http://localhost:8501 in your browser.

## 🧪 Learning Modules

### 1. Agent Loop Basics (`agent_loop_basics/`)

Understand how AI agents work at the fundamental level:
- Direct AWS Bedrock API calls
- Manual tool configuration and execution
- The complete agent loop: LLM → Tool Call → Result → LLM

```bash
python agent_loop_basics/agent_loop_basis.py
```

### 2. LangChain Agent (`langchain_agent/`)

Simplify agent creation with LangChain:
- `@tool` decorator for function tools
- `create_agent()` wrapper
- Streaming responses

```bash
python langchain_agent/langchain_agent.py
```

### 3. LangGraph Multi-Agent (`langraph_agent/`)

Orchestrate multiple specialists:
- `StateGraph` for workflow definition
- Supervisor pattern for routing
- Parallel execution with `Send()` API

```bash
python langraph_agent/langraph_agent.py
```

### 4. Production Chatbot (`chatbot/`)

Full-featured application with:
- Streamlit UI with chat interface
- Knowledge Graph integration (Neo4j)
- Colored logging for debugging
- Tool call transparency

## 🔧 Available Agents & Tools

### ⚡ Energy Agent
| Tool | Description |
|------|-------------|
| `get_energy_price(market)` | Get current price for ERCOT, PJM, MISO, CAISO, SPP, NYISO |
| `get_market_forecast(market, days)` | Get price forecast with trend, average, and range |

### 🔗 Knowledge Graph Agent
| Tool | Description |
|------|-------------|
| `kg_list_markets()` | List all energy markets |
| `kg_fetch_market_hubs(market)` | Get hubs for a market |
| `kg_fetch_hub_regions(hub)` | Get regions for a hub |
| `kg_fetch_fuel_types(market)` | Get fuel types used in a market |
| `kg_fetch_power_plants(market)` | List power plants with capacity |
| `kg_fetch_power_plants_by_fuel(fuel)` | Find plants by fuel type |
| `kg_fetch_power_plants_by_hub(hub)` | Find plants supplying a hub |

### 📦 Order Agent
| Tool | Description |
|------|-------------|
| `get_order_status(order_id)` | Get status of an order |
| `track_shipment(tracking_number)` | Track a shipment |

## 📊 Demo Scenarios

### Single Agent Queries
```
"What is the ERCOT price?"              → Energy Agent
"List all energy markets"               → Knowledge Graph Agent
"What's the status of order 12345?"     → Order Agent
"What is 2+2?"                          → Unknown Agent (graceful handling)
```

### Multi-Agent Queries
```
"ERCOT price and order 12345 status"
    → Energy Agent + Order Agent (parallel execution)

"Wind plants in ERCOT and price forecast"
    → Knowledge Graph Agent + Energy Agent

"Compare ERCOT and PJM prices, check order 12346"
    → Energy Agent + Order Agent
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Orchestration | LangGraph (StateGraph) |
| Agents | LangChain `create_agent()` |
| LLM | AWS Bedrock - Claude Haiku |
| Knowledge Graph | Neo4j |
| Data | Mock data + Real Neo4j queries |

## 📚 Key Concepts

### LangChain vs LangGraph

| Aspect | LangChain Agent | LangGraph |
|--------|-----------------|-----------|
| Architecture | Single agent | Multi-agent |
| Execution | Sequential | Parallel possible |
| Complexity | Simple | Moderate |
| Use Case | Single domain | Multi-domain orchestration |
| Scalability | Limited | High |

### Supervisor Pattern

The supervisor analyzes each query and routes to appropriate specialists:

```python
class RouteDecision(BaseModel):
    tasks: list[AgentTask]

supervisor_llm = llm.with_structured_output(RouteDecision)
```

### Parallel Execution

Multiple agents can run simultaneously using LangGraph's `Send()` API:

```python
def route_agents(state):
    return [
        Send(task["agent"], {"question": task["task"]})
        for task in state["tasks"]
    ]
```

## 🎓 Key Learnings

**What Works Well:**
- Supervisor pattern for clean routing logic
- Parallel execution for faster multi-domain queries
- Structured output for reliable routing decisions
- LangChain agents as self-contained building blocks

**Challenges:**
- Model selection and deprecation
- Controlling LLM hallucination in response synthesis
- Error handling across distributed agents
- State aggregation from parallel results

**Best Practices:**
- Start simple (LangChain) → graduate to complex (LangGraph)
- Write clear tool descriptions (LLM uses these for decisions)
- Use structured output over text parsing
- Handle out-of-scope queries gracefully

## 📖 Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Confluence Page: AI Agents - LangChain vs LangGraph](https://woodmac.atlassian.net/wiki/spaces/~712020a1b1aebdf10249079dfee9d9a19f4311/pages/3090874388)

## 📄 License

Internal use only.
