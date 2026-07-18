import streamlit as st
import requests
from src.logger import logger

API_URL = "http://127.0.0.1:8000/research"


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Agentic Researcher AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# Sidebar Configuration
# ==============================

with st.sidebar:

    st.title("🤖 Agent Control Panel")

    st.markdown("---")

    # Application status
    st.subheader("🟢 System Status")

    st.success("API Online")
    st.info("LLM: Groq Llama 3.3 70B")


    st.markdown("---")


    # Agent settings
    st.subheader("🧠 Agent Settings")

    max_agents = st.slider(
        "Number of Research Agents",
        min_value=1,
        max_value=10,
        value=3
    )


    research_depth = st.selectbox(
        "Research Depth",
        [
            "Quick Summary",
            "Standard Analysis",
            "Deep Investigation"
        ]
    )


    enable_sources = st.checkbox(
        "Include Sources",
        value=True
    )


    st.markdown("---")


    # Model settings
    st.subheader("⚙️ Model Configuration")


    model = st.selectbox(
        "Select LLM Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )


    temperature = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )


    st.markdown("---")


    # About section
    st.subheader("📌 About")

    st.caption(
        """
        Agentic Research Assistant

        Built with:
        - FastAPI
        - Streamlit
        - Groq LLM
        - Async Agents
        - LangGraph (Future)
        """
    )


# ==============================
# Main Application
# ==============================


st.title("🚀 Agentic AI Research Assistant")

st.markdown(
    """
    Enter a topic below and your AI research agents
    will collaborate to analyze the subject.
    """
)


# ==============================
# Session State
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================
# Chat History
# ==============================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# ==============================
# User Input
# ==============================

if prompt := st.chat_input(
    "What would you like to research?"
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    with st.chat_message("user"):
        st.markdown(prompt)



    # Future API call
        # API call to FastAPI backend
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        payload = {
            "topic": prompt,
            "max_analysts": max_agents
        }


        try:

            response = requests.post(
                API_URL,
                json=payload
            )


            if response.status_code == 200:

                data = response.json()

                full_response = data["answer"]

            else:

                full_response = (
                    f"Backend Error: {response.status_code}"
                )


        except Exception as e:

            full_response = (
                f"Connection Error: {str(e)}"
            )


        response_placeholder.markdown(
            full_response
        )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )


    logger.info(
        f"UI received input: {prompt}"
    )