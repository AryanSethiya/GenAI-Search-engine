import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import socket

# Set timeout for all sockets
socket.setdefaulttimeout(15)

# Safe tool initializers
def safe_arxiv():
    try:
        return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=200,
            load_max_docs=1
        ))
    except Exception as e:
        return lambda _: f"arXiv API Error: {str(e)}"

def safe_wiki():
    try:
        return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=200
        ))
    except Exception as e:
        return lambda _: f"Wikipedia API Error: {str(e)}"

# Streamlit UI
st.title("🔍 Smart Research Assistant")
st.caption("Powered by Groq (Llama3-8B) with Web Search capabilities")

# Session state management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web, arXiv, and Wikipedia. Ask me anything!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get Groq API key from secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# Chat input handling
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not groq_api_key:
        st.error("Please configure GROQ_API_KEY in Streamlit secrets!")
        st.stop()

    try:
        # Initialize LLM and tools
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192",
            temperature=0.3
        )
        
        tools = [
            DuckDuckGoSearchRun(name="Web Search"),
            safe_arxiv(),
            safe_wiki()
        ]

        # Create agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=4,
            verbose=True
        )

        # Generate response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = agent({"input": prompt}, callbacks=[st_cb])
                output = response.get("output", "No results found")
            except Exception as e:
                output = f"⚠️ Error processing request: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.write(output)

    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚨 Failed to process request. Please try again."
        })
