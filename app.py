import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os

# Fixed initialization
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session state with correct spelling
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}  # Fixed typo
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar!")
        st.stop()

    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [
            DuckDuckGoSearchRun(name="Search"),
            ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
        ]

        # Initialize agent with better error handling
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,  # Updated parameter name
            verbose=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # Pass only the latest prompt instead of entire message history
            response = search_agent.run(
                {"input": prompt},
                callbacks=[st_cb]
            )
            
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.messages.append({'role': 'assistant', "content": "⚠️ Sorry, something went wrong. Please try again."})