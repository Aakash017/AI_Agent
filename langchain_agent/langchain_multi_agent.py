from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent

from langchain.tools import tool
from langchain.agents.middleware import ToolErrorMiddleware

energy_prices = {
    "ERCOT": 38.50,
    "PJM": 42.20,
    "CAISO": 51.10,
}

market_region={
    "ERCOT":"TEXAS",
    "PJM":"NEWYORK",
    "CAISO":"CALIFORNIA"
}

market_capacity={
    "ERCOT":"23MWH",
    "PJM":"466MWH",
    "CAISO":"100MWH"
}







@tool
def get_energy_price(market: str) -> float | str:
    """Get the energy price for an electricity market."""
    ...
    return {"market":market, "energy_price":energy_prices[market.upper()]}

@tool
def get_market_region(market: str) -> str:
    """Get the geographic region for an electricity market."""
    ...
    return {"market":market, "region":market_region.get(market.upper(), "NOT FOUND")}

@tool
def get_generation_capacity(market: str) -> float | str:
    """Get the generation capacity of an electricity market in MW."""
    ...
    return {"market":market,
    "generation_capacity":market_capacity.get(market.upper(), "NOT FOUND")}


def handle_tool_error(exc, request):
    if isinstance(exc, KeyError):
        return "The requested market was not found."

    return None


model = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=3000,
    credentials_profile_name="woodmac_ai_non_prod",

)


agent = create_agent(
    model=model,
    tools=[
        get_energy_price,
        get_market_region,
        get_generation_capacity,
    ],
     middleware=[
        ToolErrorMiddleware(handle_tool_error)
    ],
    system_prompt="""You are an energy market assistant.

For factual energy-market questions, always use the available tools.

Treat tool results as the source of truth.
Do not add factual information from your own knowledge.

If the user asks for multiple pieces of information:
- Gather all required tool results first.
- Do not provide a partial answer while tool calls are still required.
- After all necessary tools have completed, provide one consolidated final answer."""
)

user_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is MISO's energy price and which region does it operate in?"
        }
    ]
}

for message_chunk in agent.stream(
    user_input,
    stream_mode="updates"
):
    # if isinstance(message_chunk.content, list):
    #     for block in message_chunk.content:
    #         if block.get("type") == "text":
    #             print(block["text"], end="", flush=True)
    print(message_chunk)

