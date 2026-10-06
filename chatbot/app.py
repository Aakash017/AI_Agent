import streamlit as st
from agent import MultiAgentSystem

st.set_page_config(
    page_title="Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 14px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .energy-badge { background-color: #fef3c7; color: #92400e; }
    .order-badge { background-color: #dbeafe; color: #1e40af; }
    .tool-call { 
        background-color: #f3f4f6; 
        padding: 8px 12px; 
        border-radius: 8px; 
        font-family: monospace;
        font-size: 13px;
        margin: 4px 0;
    }
    .stChatMessage { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Assistant")
st.caption("Powered by LangGraph + AWS Bedrock + Neo4j Knowledge Graph | Energy Markets, Power Plants & Orders")

# Initialize the agent system
@st.cache_resource
def get_agent():
    return MultiAgentSystem()

agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("🎯 Demo Scenarios")
    
    st.subheader("⚡ Energy Prices")
    if st.button("What is the ERCOT price?", key="demo1"):
        st.session_state.demo_query = "What is the ERCOT price?"
    if st.button("Compare ERCOT and PJM prices", key="demo2"):
        st.session_state.demo_query = "Compare the prices for ERCOT and PJM"
    if st.button("ERCOT 7-day forecast", key="demo3"):
        st.session_state.demo_query = "What's the forecast for ERCOT?"
    
    st.subheader("🔗 Knowledge Graph")
    if st.button("List all markets", key="kg1"):
        st.session_state.demo_query = "What energy markets are available?"
    if st.button("ERCOT hubs", key="kg2"):
        st.session_state.demo_query = "What hubs are in the ERCOT market?"
    if st.button("Power plants in ERCOT", key="kg3"):
        st.session_state.demo_query = "Show me power plants in ERCOT"
    if st.button("All wind power plants", key="kg4"):
        st.session_state.demo_query = "List all wind power plants"
    if st.button("Fuel types in ERCOT", key="kg5"):
        st.session_state.demo_query = "What fuel types are used in ERCOT?"
    
    st.subheader("📦 Order Tracking")
    if st.button("Check order 12345", key="demo4"):
        st.session_state.demo_query = "What's the status of order 12345?"
    if st.button("Track shipment FX123456", key="demo5"):
        st.session_state.demo_query = "Track shipment FX123456"
    
    st.subheader("🚀 Multi-Agent Queries")
    if st.button("Price + KG + Order", key="demo6"):
        st.session_state.demo_query = "What's the ERCOT price, how many power plants are in ERCOT, and status of order 12345?"
    if st.button("Market deep dive", key="demo7"):
        st.session_state.demo_query = "Give me ERCOT price forecast and list all the wind plants in that market"
    
    st.divider()
    
    st.subheader("📊 Available Data")
    with st.expander("Energy Markets"):
        st.code("ERCOT, MISO, PJM, CAISO, SPP, NYISO")
    with st.expander("Knowledge Graph"):
        st.markdown("""
        **Markets**: ERCOT, NYISO, etc.
        **Hubs**: HB_NORTH, HB_SOUTH, etc.
        **Fuel Types**: Wind, Solar, Natural Gas, Coal, Nuclear
        **Power Plants**: 100+ plants with capacity data
        """)
    with st.expander("Order IDs"):
        st.code("12345 (shipped)\n12346 (delivered)\n12347 (processing)\n99999 (cancelled)")
    with st.expander("Tracking Numbers"):
        st.code("FX123456 (in transit)\nFX789012 (delivered)")
    
    st.divider()
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show agent info for assistant messages
        if message["role"] == "assistant" and "agents_used" in message:
            cols = st.columns([3, 1])
            with cols[0]:
                if message["agents_used"]:
                    st.caption(f"Handled by: {' • '.join(message['agents_used'])}")
            with cols[1]:
                if message.get("tool_calls"):
                    with st.expander("🔧 Tools Used"):
                        for tc in message["tool_calls"]:
                            st.code(f"{tc['tool']}({tc['input']})")

# Handle demo button clicks
if "demo_query" in st.session_state and st.session_state.demo_query:
    prompt = st.session_state.demo_query
    st.session_state.demo_query = None  # Clear it
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Routing to agents..."):
            result = agent.run(prompt)
        
        st.markdown(result["answer"])
        
        # Show metadata
        cols = st.columns([3, 1])
        with cols[0]:
            if result["agents_used"]:
                st.caption(f"Handled by: {' • '.join(result['agents_used'])}")
        with cols[1]:
            if result["tool_calls"]:
                with st.expander("🔧 Tools Used"):
                    for tc in result["tool_calls"]:
                        st.code(f"{tc['tool']}({tc['input']})")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "agents_used": result.get("agents_used", []),
        "tool_calls": result.get("tool_calls", [])
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about energy prices, power plants, markets, or orders..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Routing to agents..."):
            result = agent.run(prompt)
        
        st.markdown(result["answer"])
        
        # Show metadata
        cols = st.columns([3, 1])
        with cols[0]:
            if result["agents_used"]:
                st.caption(f"Handled by: {' • '.join(result['agents_used'])}")
        with cols[1]:
            if result["tool_calls"]:
                with st.expander("🔧 Tools Used"):
                    for tc in result["tool_calls"]:
                        st.code(f"{tc['tool']}({tc['input']})")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "agents_used": result.get("agents_used", []),
        "tool_calls": result.get("tool_calls", [])
    })
