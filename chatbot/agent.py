"""
Multi-Agent System for the Chatbot
Wraps the LangGraph supervisor + specialist agents
Includes Energy Knowledge Graph integration via direct Neo4j queries
"""

import os
import sys
import operator
import logging
from datetime import datetime
from typing import TypedDict, Annotated, Literal, Callable
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel

load_dotenv()

# Add energy-knowledge-graph to path for importing services
ENERGY_KG_PATH = r"c:\Users\I42881\Documents\projects\energy-knowledge-graph"
if ENERGY_KG_PATH not in sys.path:
    sys.path.insert(0, ENERGY_KG_PATH)

# ============ LOGGING SETUP ============

# Custom formatter with colors for terminal
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
        'BLUE': '\033[34m',      # Blue
        'MAGENTA': '\033[35m',   # Magenta
    }

    def format(self, record):
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        bold = self.COLORS['BOLD']
        
        # Format the message
        formatted = f"{color}[{timestamp}] {bold}{record.levelname}{reset} {record.msg}"
        return formatted

# Setup logger
logger = logging.getLogger("MultiAgentSystem")
logger.setLevel(logging.DEBUG)

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# Prevent duplicate logs
logger.propagate = False


# ============ TOOLS ============

@tool
def get_energy_price(market: str) -> str:
    """Get the current energy price for a power market."""
    prices = {
        "ERCOT": 38.5,
        "MISO": 42.0,
        "PJM": 45.2,
        "CAISO": 51.1,
        "SPP": 35.8,
        "NYISO": 48.3,
    }
    price = prices.get(market.upper())
    if price is None:
        available = ", ".join(prices.keys())
        return f"Market '{market}' not found. Available markets: {available}"
    return f"{market.upper()} price is {price} USD/MWh"


@tool
def get_market_forecast(market: str, days: int = 7) -> str:
    """Get energy price forecast for a market for the next N days."""
    forecasts = {
        "ERCOT": {"trend": "rising", "avg": 41.2, "high": 52.0, "low": 35.0},
        "MISO": {"trend": "stable", "avg": 43.5, "high": 48.0, "low": 38.0},
        "PJM": {"trend": "falling", "avg": 42.0, "high": 46.0, "low": 39.0},
        "CAISO": {"trend": "rising", "avg": 55.0, "high": 68.0, "low": 45.0},
    }
    forecast = forecasts.get(market.upper())
    if not forecast:
        return f"No forecast available for {market}"
    return (
        f"{market.upper()} {days}-day forecast: "
        f"Trend {forecast['trend']}, "
        f"Average ${forecast['avg']}/MWh, "
        f"Range ${forecast['low']}-${forecast['high']}/MWh"
    )


@tool
def get_order_status(order_id: str) -> str:
    """Get the status of an order using the order ID."""
    statuses = {
        "12345": {"status": "shipped", "carrier": "FedEx", "eta": "Sep 26, 2026"},
        "12346": {"status": "delivered", "delivered_on": "Sep 24, 2026"},
        "12347": {"status": "processing", "estimated_ship": "Sep 27, 2026"},
        "99999": {"status": "cancelled", "reason": "Customer request"},
    }
    order = statuses.get(order_id)
    if order is None:
        return f"Order {order_id} not found in our system"
    
    status = order["status"]
    if status == "shipped":
        return f"Order {order_id}: Shipped via {order['carrier']}, ETA: {order['eta']}"
    elif status == "delivered":
        return f"Order {order_id}: Delivered on {order['delivered_on']}"
    elif status == "processing":
        return f"Order {order_id}: Processing, estimated ship date: {order['estimated_ship']}"
    else:
        return f"Order {order_id}: {status.capitalize()} - {order.get('reason', '')}"


@tool
def track_shipment(tracking_number: str) -> str:
    """Track a shipment using the tracking number."""
    shipments = {
        "FX123456": {"location": "Memphis, TN", "status": "In transit", "last_update": "2 hours ago"},
        "FX789012": {"location": "Delivered", "status": "Delivered", "last_update": "Sep 24, 2026"},
    }
    shipment = shipments.get(tracking_number.upper())
    if not shipment:
        return f"Tracking number {tracking_number} not found"
    return f"Shipment {tracking_number}: {shipment['status']} - {shipment['location']} (Updated: {shipment['last_update']})"


# ============ KNOWLEDGE GRAPH TOOLS ============
# These directly query the Energy Knowledge Graph (Neo4j)

# Lazy import - only connect to Neo4j when tools are actually called
KG_IMPORT_ERROR = None
kg_services = None

def get_kg_services():
    """Lazy load the KG services module."""
    global kg_services, KG_IMPORT_ERROR
    if kg_services is not None:
        return kg_services
    if KG_IMPORT_ERROR is not None:
        return None
    try:
        import services as _kg_services
        kg_services = _kg_services
        logger.info("✅ Energy Knowledge Graph service loaded (Neo4j)")
        return kg_services
    except Exception as e:
        KG_IMPORT_ERROR = str(e)
        logger.warning(f"⚠️ Energy Knowledge Graph not available: {e}")
        return None


@tool
def kg_list_markets() -> str:
    """List all available energy markets in the knowledge graph (e.g., ERCOT, NYISO, PJM)."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Make sure Neo4j is running. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_all_markets()
        markets = result.get("markets", [])
        # Markets can be either a list of strings or list of dicts
        if markets and isinstance(markets[0], dict):
            market_names = [m.get("name", "Unknown") for m in markets]
        else:
            market_names = markets  # Already a list of strings
        return f"Available markets ({len(market_names)}): {', '.join(market_names)}"
    except Exception as e:
        return f"Error listing markets: {str(e)}"


@tool
def kg_fetch_market_hubs(market_name: str) -> str:
    """Get all hubs for a specific energy market (e.g., ERCOT has HB_NORTH, HB_SOUTH, etc.)."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_market_hubs(market_name)
        hubs = result.get("hubs", [])
        # Hubs can be either a list of strings or list of dicts
        if hubs and isinstance(hubs[0], dict):
            hub_names = [h.get("name", "Unknown") for h in hubs]
        else:
            hub_names = hubs  # Already a list of strings
        if not hub_names:
            return f"No hubs found for market '{market_name}'"
        return f"Hubs in {market_name} ({len(hub_names)} total): {', '.join(hub_names)}"
    except Exception as e:
        return f"Error fetching hubs: {str(e)}"


@tool
def kg_fetch_hub_regions(hub_name: str) -> str:
    """Get all regions associated with a specific hub."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_hub_region(hub_name)
        regions = result.get("region", [])
        # Regions can be either a list of strings or list of dicts
        if regions and isinstance(regions[0], dict):
            region_names = [r.get("name", "Unknown") for r in regions]
        else:
            region_names = regions  # Already a list of strings
        if not region_names:
            return f"No regions found for hub '{hub_name}'"
        return f"Regions in {hub_name}: {', '.join(region_names)}"
    except Exception as e:
        return f"Error fetching regions: {str(e)}"


@tool
def kg_fetch_market_hubs_with_regions(market_name: str) -> str:
    """Get all hubs for a market along with their associated regions."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.fetch_market_hubs_with_regions(market_name)
        hubs_with_regions = result.get("hubs_with_regions", [])
        if not hubs_with_regions:
            return f"No hubs found for market '{market_name}'"
        
        lines = [f"Hubs and regions for {market_name}:"]
        for item in hubs_with_regions:
            hub = item.get("hub", "Unknown")
            regions = item.get("regions", [])
            regions_str = ", ".join(regions) if regions else "No regions"
            lines.append(f"• {hub}: {regions_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching hubs with regions: {str(e)}"


@tool
def kg_fetch_fuel_types(market_name: str) -> str:
    """Get all fuel types used by power plants in a specific market (e.g., Wind, Solar, Natural Gas)."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_fuel_types_for_market(market_name)
        fuel_types = result.get("fuel_types", [])
        if not fuel_types:
            return f"No fuel types found for market '{market_name}'"
        return f"Fuel types in {market_name}: {', '.join(fuel_types)}"
    except Exception as e:
        return f"Error fetching fuel types: {str(e)}"


@tool
def kg_fetch_power_plants(market_name: str) -> str:
    """Get all power plants in a specific market with their capacity and fuel type."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_power_plants_for_market(market_name)
        plants = result.get("power_plants", [])
        count = result.get("count", 0)
        if not plants:
            return f"No power plants found for market '{market_name}'"
        
        lines = [f"Power plants in {market_name} ({count} total, showing first 10):"]
        for p in plants[:10]:
            name = p.get("name", "Unknown")
            capacity = p.get("capacity_mw", "N/A")
            fuel = p.get("fuel_type", "Unknown")
            lines.append(f"• {name} ({capacity} MW, {fuel})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching power plants: {str(e)}"


@tool
def kg_fetch_power_plants_by_fuel(fuel_type: str) -> str:
    """Get all power plants that use a specific fuel type (e.g., Wind, Solar, Coal, Natural Gas)."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_power_plants_by_fuel(fuel_type)
        plants = result.get("power_plants", [])
        count = result.get("count", 0)
        if not plants:
            return f"No power plants found using fuel type '{fuel_type}'"
        
        lines = [f"{fuel_type} power plants ({count} total, showing first 10):"]
        for p in plants[:10]:
            name = p.get("name", "Unknown")
            capacity = p.get("capacity_mw", "N/A")
            hub = p.get("hub", "Unknown hub")
            lines.append(f"• {name} ({capacity} MW, supplies {hub})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching power plants by fuel: {str(e)}"


@tool
def kg_fetch_power_plants_by_hub(hub_name: str) -> str:
    """Get all power plants that supply a specific hub with their capacity and fuel type."""
    services = get_kg_services()
    if services is None:
        return f"Knowledge Graph service not available. Error: {KG_IMPORT_ERROR}"
    try:
        result = services.get_power_plants_by_hub(hub_name)
        plants = result.get("power_plants", [])
        count = result.get("count", 0)
        total_capacity = result.get("total_capacity_mw", 0)
        if not plants:
            return f"No power plants found for hub '{hub_name}'"
        
        lines = [f"Power plants supplying {hub_name} ({count} total, {total_capacity} MW total capacity):"]
        for p in plants[:10]:
            name = p.get("name", "Unknown")
            capacity = p.get("capacity_mw", "N/A")
            fuel = p.get("fuel_type", "Unknown")
            lines.append(f"• {name} ({capacity} MW, {fuel})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching power plants by hub: {str(e)}"


# ============ STATE ============

class AgentState(TypedDict):
    question: str
    tasks: list[dict]
    agent_results: Annotated[list[str], operator.add]
    agents_used: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[dict], operator.add]
    answer: str


class AgentTask(BaseModel):
    agent: Literal["energy_agent", "order_agent", "knowledge_graph_agent", "unknown_agent"]
    task: str


class RouteDecision(BaseModel):
    tasks: list[AgentTask]


# ============ MULTI-AGENT SYSTEM ============

class MultiAgentSystem:
    def __init__(self):
        # Initialize the LLM
        self.llm = ChatBedrockConverse(
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            temperature=0,
            max_tokens=3000,
            credentials_profile_name=os.environ.get("AWS_PROFILE"),
        )
        
        # Create specialist agents
        self.energy_agent = create_agent(
            model=self.llm,
            tools=[get_energy_price, get_market_forecast],
            system_prompt="You are an energy market assistant. Always use tools for energy questions."
        )
        
        self.order_agent = create_agent(
            model=self.llm,
            tools=[get_order_status, track_shipment],
            system_prompt="You are an order delivery assistant. Always use tools for order queries."
        )
        
        # Knowledge Graph agent (queries Neo4j energy data)
        kg_tools = [
            kg_list_markets,
            kg_fetch_market_hubs,
            kg_fetch_hub_regions,
            kg_fetch_market_hubs_with_regions,
            kg_fetch_fuel_types,
            kg_fetch_power_plants,
            kg_fetch_power_plants_by_fuel,
            kg_fetch_power_plants_by_hub,
        ]
        self.knowledge_graph_agent = create_agent(
            model=self.llm,
            tools=kg_tools,
            system_prompt="""You are an energy knowledge graph assistant with access to a Neo4j database.
You can query information about:
- Energy markets (ERCOT, NYISO, PJM, etc.)
- Market hubs and their regions
- Power plants, their capacity, and fuel types
- Fuel type distributions across markets

Always use the appropriate kg_* tools to answer questions about market structure, 
power plants, hubs, regions, and fuel types."""
        )
        
        # Supervisor LLM with structured output
        self.supervisor_llm = self.llm.with_structured_output(RouteDecision)
        
        # Build the graph
        self.app = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("energy_agent", self._energy_agent)
        graph.add_node("order_agent", self._order_agent)
        graph.add_node("knowledge_graph_agent", self._knowledge_graph_agent)
        graph.add_node("unknown_agent", self._unknown_agent)
        graph.add_node("final_response", self._final_response)
        
        # Add edges
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._route_agents)
        graph.add_edge("energy_agent", "final_response")
        graph.add_edge("order_agent", "final_response")
        graph.add_edge("knowledge_graph_agent", "final_response")
        graph.add_edge("unknown_agent", "final_response")
        graph.add_edge("final_response", END)
        
        return graph.compile()
    
    def _supervisor(self, state: AgentState):
        question = state["question"]
        logger.info("=" * 60)
        logger.info(f"🎯 SUPERVISOR: Analyzing question...")
        logger.debug(f"   Question: \"{question}\"")
        
        prompt = f"""You are a supervisor responsible for routing user questions.

Choose one or more agents required to answer the user question:

energy_agent:
Handles current energy prices, price forecasts, and real-time market data.
Use for questions like "What is the ERCOT price?" or "Give me a 7-day forecast for PJM"

knowledge_graph_agent:
Handles structural queries about the energy domain from a knowledge graph database.
Use for questions about:
- Market structure (what markets exist, what hubs are in a market)
- Power plants (list plants, plants by fuel type, plants supplying a hub)
- Fuel types (what fuels are used in a market)
- Regions and hubs (what regions are in a hub, region data)
Use for questions like "What power plants are in ERCOT?", "List all wind plants", 
"What hubs are in NYISO?", "Show me fuel types in PJM"

order_agent:
Handles orders, order IDs, shipping, delivery, tracking numbers and logistics.

unknown_agent:
Handles anything that does not belong to the above categories.

User question:
{question}"""

        decision = self.supervisor_llm.invoke(prompt)
        
        tasks = [{"agent": task.agent, "task": task.task} for task in decision.tasks]
        
        logger.info(f"📋 ROUTING DECISION: {len(tasks)} agent(s) selected")
        for task in tasks:
            logger.info(f"   → {task['agent']}: \"{task['task']}\"")
        
        return {"tasks": tasks}
    
    def _route_agents(self, state):
        return [
            Send(task["agent"], {"question": task["task"]})
            for task in state["tasks"]
        ]
    
    def _extract_tool_calls(self, result) -> list[dict]:
        """Extract tool call information from agent result."""
        tool_calls = []
        for message in result["messages"]:
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "tool": tc["name"],
                        "input": tc["args"]
                    })
        return tool_calls
    
    def _energy_agent(self, state: AgentState):
        logger.info(f"⚡ ENERGY AGENT: Processing...")
        logger.debug(f"   Task: \"{state['question']}\"")
        
        result = self.energy_agent.invoke({
            "messages": [{"role": "user", "content": state["question"]}]
        })
        
        tool_calls = self._extract_tool_calls(result)
        for tc in tool_calls:
            logger.info(f"   🔧 Tool: {tc['tool']}({tc['input']})")
        
        answer = result["messages"][-1].content
        logger.info(f"   ✅ Result: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
        return {
            "agent_results": [answer],
            "agents_used": ["⚡ Energy Agent"],
            "tool_calls": tool_calls
        }
    
    def _order_agent(self, state: AgentState):
        logger.info(f"📦 ORDER AGENT: Processing...")
        logger.debug(f"   Task: \"{state['question']}\"")
        
        result = self.order_agent.invoke({
            "messages": [{"role": "user", "content": state["question"]}]
        })
        
        tool_calls = self._extract_tool_calls(result)
        for tc in tool_calls:
            logger.info(f"   🔧 Tool: {tc['tool']}({tc['input']})")
        
        answer = result["messages"][-1].content
        logger.info(f"   ✅ Result: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
        return {
            "agent_results": [answer],
            "agents_used": ["📦 Order Agent"],
            "tool_calls": tool_calls
        }
    
    def _knowledge_graph_agent(self, state: AgentState):
        logger.info(f"🔗 KNOWLEDGE GRAPH AGENT: Processing...")
        logger.debug(f"   Task: \"{state['question']}\"")
        
        result = self.knowledge_graph_agent.invoke({
            "messages": [{"role": "user", "content": state["question"]}]
        })
        
        tool_calls = self._extract_tool_calls(result)
        for tc in tool_calls:
            logger.info(f"   🔧 KG Tool: {tc['tool']}({tc['input']})")
        
        answer = result["messages"][-1].content
        logger.info(f"   ✅ Result: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
        return {
            "agent_results": [answer],
            "agents_used": ["🔗 Knowledge Graph Agent"],
            "tool_calls": tool_calls
        }
    
    def _unknown_agent(self, state: AgentState):
        logger.info(f"❓ UNKNOWN AGENT: No specialized handler")
        logger.debug(f"   Task: \"{state['question']}\"")
        
        helpful_message = """I can help with:
• **Energy market prices and forecasts** (ERCOT, PJM, MISO, CAISO, SPP, NYISO)
• **Power plants, hubs, and fuel types** from the Knowledge Graph
• **Order status and shipment tracking**

Your question is outside my area of expertise. Try asking about energy markets or orders!"""
        
        return {
            "agent_results": [helpful_message],
            "agents_used": ["❓ General Agent"],
            "tool_calls": []
        }
    
    def _final_response(self, state: AgentState):
        logger.info(f"📝 FINAL RESPONSE: Synthesizing {len(state['agent_results'])} result(s)...")
        
        prompt = f"""Answer the user's question using ONLY the agent results below.

STRICT RULES:
- Use ONLY information from the agent results. Do not add any external knowledge.
- Do not mention agents, tools, routing, or internal processing.
- Do not invent, elaborate, or add information not in the results.
- If the agent result says the question is outside scope, repeat that message without adding context.
- Preserve factual values exactly as given.
- Keep the response concise and natural.
- Use markdown formatting for clarity.

Original question:
{state["question"]}

Agent results:
{state["agent_results"]}"""

        response = self.llm.invoke(prompt)
        
        logger.info(f"✨ COMPLETE: Response generated")
        logger.info("=" * 60)
        
        return {"answer": response.content}
    
    def run(self, question: str, on_step: Callable[[dict], None] = None) -> dict:
        """
        Run the multi-agent system.
        
        Args:
            question: The user's question
            on_step: Optional callback function called with state updates (for UI streaming)
        
        Returns:
            dict with 'answer', 'agents_used', and 'tool_calls'
        """
        logger.info("")
        logger.info("🚀 NEW REQUEST")
        logger.info(f"📨 Question: \"{question}\"")
        
        initial_state = {
            "question": question,
            "tasks": [],
            "agent_results": [],
            "agents_used": [],
            "tool_calls": [],
            "answer": ""
        }

        result = initial_state

        for snapshot in self.app.stream(initial_state, stream_mode="values"):
            result = snapshot
            if on_step:
                on_step(snapshot)

        return {
            "answer": result["answer"],
            "agents_used": result.get("agents_used", []),
            "tool_calls": result.get("tool_calls", [])
        }


# For testing
if __name__ == "__main__":
    agent = MultiAgentSystem()
    
    print("\n" + "="*60)
    print("TEST 1: Single agent query")
    print("="*60)
    result = agent.run("What is the ERCOT price?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Agents: {result['agents_used']}")
    print(f"Tools: {result['tool_calls']}")
    
    print("\n" + "="*60)
    print("TEST 2: Multi-agent query")
    print("="*60)
    result = agent.run("What's the ERCOT price and status of order 12345?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Agents: {result['agents_used']}")
    print(f"Tools: {result['tool_calls']}")
