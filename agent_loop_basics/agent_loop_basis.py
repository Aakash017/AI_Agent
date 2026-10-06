import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
bedrock = session.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

energy_prices = {
    "ERCOT": 38.50,
    "PJM": 42.20,
    "CAISO": 51.10,
}

# Tool implementations
def get_energy_price(market: str):
    return energy_prices.get(market, "NOT Found")

# Map tool names to functions
available_tools = {
    "get_energy_price": get_energy_price,
}

tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_energy_price",
                "description": "Get the energy price for a market",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "market": {
                                "type": "string",
                                "description": "The energy market name (e.g., ERCOT, PJM, CAISO)"
                            }
                        },
                        "required": ["market"]
                    }
                }
            }
        }
    ]
}


def run_agent(user_message: str):
    """Run the agent loop until completion."""
    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}]
        }
    ]
    
    while True:
        # Call the model
        response = bedrock.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            toolConfig=tool_config
        )
        
        stop_reason = response["stopReason"]
        assistant_message = response["output"]["message"]
        
        # Add assistant response to conversation
        messages.append(assistant_message)
        
        # If the model is done, return the final response
        if stop_reason == "end_turn":
            # Extract the text response
            for block in assistant_message["content"]:
                if "text" in block:
                    return block["text"]
            return None
        
        # If the model wants to use tools
        if stop_reason == "tool_use":
            tool_results = []
            
            for block in assistant_message["content"]:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_name = tool_use["name"]
                    tool_input = tool_use["input"]
                    tool_use_id = tool_use["toolUseId"]
                    
                    print(f"[Tool Call] {tool_name}({tool_input})")
                    
                    # Execute the tool
                    if tool_name in available_tools:
                        result = available_tools[tool_name](**tool_input)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                    
                    print(f"[Tool Result] {result}")
                    
                    tool_results.append({
                        "toolUseId": tool_use_id,
                        "content": [{"text": str(result)}]
                    })
            
            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results]
            })
        else:
            # Unexpected stop reason
            print(f"Unexpected stop reason: {stop_reason}")
            break
    
    return None


# Run the agent
if __name__ == "__main__":
    result = run_agent("What is the energy price for ERCOT?")
    print(f"\n[Final Answer] {result}")





