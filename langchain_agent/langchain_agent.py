from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent

from langchain.tools import tool


energy_prices = {
    "ERCOT": 38.50,
    "PJM": 42.20,
    "CAISO": 51.10,
}

@tool
def get_energy_price(market: str) -> float | str:
    """Get the energy price for a given electricity market."""
    return energy_prices.get(market.upper(), "NOT Found")



model = ChatBedrockConverse(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=3000,
    credentials_profile_name="woodmac_ai_non_prod",

)

agent = create_agent(
    model=model,
    tools=[get_energy_price], # basically replacing your manually maintained Bedrock tool registry.
    system_prompt="You are a helpful energy market assistant. Only answer question related to energy market"
)


# response = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is the energy price for ERCOT?"
#             }
#         ]
#     }
# )

# print(response)


user_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of INDIA?"
        }
    ]
}

for message_chunk, metadata in agent.stream(
    user_input,
    stream_mode="messages"
):
    if isinstance(message_chunk.content, list):
        for block in message_chunk.content:
            if block.get("type") == "text":
                print(block["text"], end="", flush=True)


