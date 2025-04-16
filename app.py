import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import socket

# Set network timeout
socket.setdefaulttimeout(20)

# Tool initialization with proper input handling
def get_tools():
    def create_tool(factory, name):
        try:
            tool = factory()
            tool.name = name  # Explicit name setting
            tool.args_schema = None  # Disable Pydantic validation
            return tool
        except Exception:
            return None

    return [
        DuckDuckGoSearchRun(name="WebSearch"),
        create_tool(lambda: ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        ), "Arxiv"),
        create_tool(lambda: WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        ), "Wikipedia")
    ]

# Streamlit UI
st.title("🔍 AI Research Assistant")
st.caption("Powered by Groq's Llama3-8B with real-time web search capabilities")

# Session management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can search the web, arXiv, and Wikipedia. Ask me anything!"}
    ]

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get API key from secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# Process input
if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not groq_api_key:
        st.error("Missing GROQ_API_KEY in secrets!")
        st.stop()

    try:
        # Initialize components
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192",
            temperature=0.4,
            max_tokens=1024
        )
        
        tools = [tool for tool in get_tools() if tool is not None]

        # Create agent with proper config
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=3,
            verbose=True
        )

        # Execute agent with correct arguments
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                # Corrected execution call
                response = agent.run(
                    input=prompt,  # Pass as keyword argument
                    callbacks=[st_cb]
                )
                output = response or "No relevant results found"
            except Exception as e:
                output = f"⚠️ Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.write(output)

    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚨 Service temporarily unavailable. Please try again."
        })
