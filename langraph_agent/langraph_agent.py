from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent

from langchain.tools import tool
from langchain.agents.middleware import ToolErrorMiddleware
from pydantic import BaseModel
from typing import Literal


@tool
def get_energy_price(market: str) -> str:
    """Get the current energy price for a power market."""

    prices = {
        "ERCOT": 38.5,
        "MISO": 42.0,
        "PJM": 45.2
    }

    price = prices.get(market.upper())

    if price is None:
        return f"Price not found for {market}"

    return f"{market.upper()} price is {price} USD/MWh"

@tool
def get_order_status(order_id: str) -> str:
    """Get the status of an order using the order ID."""

    statuses = {
        "12345": "shipped",
        "12346": "delivered",
        "12347": "processing"
    }
    if order_id == "99999":
        raise RuntimeError("Order service is unavailable")

    status = statuses.get(order_id)

    if status is None:
        return f"Order {order_id} not found"

    return f"Order {order_id} status is {status}"

llm = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=3000,
    credentials_profile_name="woodmac_ai_non_prod",

)


def handle_tool_error(exc, request):
    if isinstance(exc, RuntimeError):
        return "The service is currently unavailable. Please try again later."

    if isinstance(exc, KeyError):
        return "The requested item was not found."

    return None

energy_langchain_agent  = create_agent(
    model=llm,
    tools=[
        get_energy_price,
    ],
     middleware=[
        ToolErrorMiddleware(handle_tool_error)
    ],
    system_prompt="""You are an energy market assistant.
    For factual energy-market questions, always use the available tools.
    """
    )

order_langchain_agent  = create_agent(
    model=llm,
    tools=[
        get_order_status,
    ],
     middleware=[
        ToolErrorMiddleware(handle_tool_error)
    ],
    system_prompt="""You are an Order delivery assistant.
    For order status related always use the available tools.
    """
    )

import operator
from typing import Annotated


class AgentState(TypedDict):
    question: str
    tasks: list[dict]
    agent_results: Annotated[
        list[str],
        operator.add
    ]

    answer: str


class AgentTask(BaseModel):
    agent: Literal[
        "energy_agent",
        "order_agent",
        "unknown_agent"
    ]
    task: str

class RouteDecision(BaseModel):
        tasks: list[AgentTask]

def energy_agent(state: AgentState):
    result = energy_langchain_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": state["question"]
                    }
                ]
            }
        )
    print("ENERGY AGENT RESULT:")
    print(result)

    final_message = result["messages"][-1]

    return {
        "agent_results": [
            final_message.content
        ]
    }


def order_agent(state: AgentState):
    result = order_langchain_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": state["question"]
                    }
                ]
            }
        )
    print("ENERGY AGENT RESULT:")
    print(result)

    final_message = result["messages"][-1]

    return {
        "agent_results": [
            final_message.content
        ]
    }

def unknown_agent(state: AgentState):
    return {
        "answer": "I don't know which agent should handle this request."
    }



supervisor_llm = llm.with_structured_output(RouteDecision)


def supervisor(state: AgentState):
    question = state["question"]
    prompt  = f"""You are a supervisor responsible for routing user questions.

    Choose one or more agents required to answer the user question:

    energy_agent:
    Handles energy markets, electricity prices, load, generation,
    ERCOT, MISO, PJM, CAISO and similar energy-market questions.

    order_agent:
    Handles orders, order IDs, shipping, delivery and tracking.

    unknown_agent:
    Handles anything that does not belong to the above categories.

    User question:
    {question}"""

    decision = supervisor_llm.invoke(prompt)
    return {
        "tasks": [
            {
                "agent": task.agent,
                "task": task.task
            }
            for task in decision.tasks
        ]
    }


def route_agents(state):
    return [
        Send(
            task["agent"],
            {
                "question": task["task"]
            }
        )
        for task in state["tasks"]
    ]


def final_response(state: AgentState):

    results = state["agent_results"]

    prompt = f"""
        Answer the user's question using the agent results below.

        Rules:
        - Give only the useful final answer.
        - Do not mention agents, tools, routing, or internal processing.
        - Do not invent information.
        - Preserve factual values exactly.
        - Keep the response concise and natural.
        - Preserve numbers, units, IDs, and statuses exactly as returned.

        Original question:
        {state["question"]}

        Agent results:
        {state["agent_results"]}
        """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }



graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor)
graph.add_node("energy_agent", energy_agent)
graph.add_node("order_agent", order_agent)
graph.add_node("unknown_agent", unknown_agent)
graph.add_node("final_response", final_response)


graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_agents
)

graph.add_edge("energy_agent", "final_response")
graph.add_edge("order_agent", "final_response")
graph.add_edge("unknown_agent", "final_response")

graph.add_edge("final_response", END)


app = graph.compile()

result = app.invoke({
    "question":
        "What is the ERCOT price and what is the status of order 12346?",

    "agent_results": [],
    "answer": ""
})

print(result)
