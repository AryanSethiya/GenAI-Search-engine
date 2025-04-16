import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import socket

# Set timeout for network requests
socket.setdefaulttimeout(15)

# Tool initialization with input schema fixes
def get_tools():
    def create_wiki_tool():
        try:
            tool = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(
                    top_k_results=1,
                    doc_content_chars_max=200
                )
            )
            tool.args_schema = None  # Disable structured input
            return tool
        except Exception:
            return None

    def create_arxiv_tool():
        try:
            tool = ArxivQueryRun(
                api_wrapper=ArxivAPIWrapper(
                    top_k_results=1,
                    doc_content_chars_max=200
                )
            )
            tool.args_schema = None  # Disable structured input
            return tool
        except Exception:
            return None

    return [
        DuckDuckGoSearchRun(name="WebSearch"),
        create_arxiv_tool(),
        create_wiki_tool()
    ]

# Streamlit UI setup
st.title("🔍 AI Research Assistant")
st.markdown("""
Powered by **Groq/Llama3-8B** with access to:
- Web Search (DuckDuckGo)
- Academic Papers (arXiv)
- Wikipedia
""")

# Session state management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything and I'll search relevant sources."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get API key from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# Chat input handling
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
            temperature=0.4
        )
        
        # Get tools with error handling
        tools = [tool for tool in get_tools() if tool is not None]

        # Create agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=3,
            verbose=True
        )

        # Generate response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(
                    {"input": prompt},
                    callbacks=[st_cb],
                    include_run_info=True
                )
                output = response if response else "No results found"
            except Exception as e:
                output = f"⚠️ Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.write(output)

    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚨 Service unavailable. Please try later."
        })
