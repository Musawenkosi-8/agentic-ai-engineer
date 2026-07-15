import streamlit as st
from src.logger import logger

st.set_page_config(page_title="Agentic Researcher UI", page_icon="🤖")

# =========================
# Sidebar
# =========================
st.sidebar.header("Research Settings")

max_analysts = st.sidebar.slider(
    "Maximum Analysts",
    min_value=1,
    max_value=5,
    value=3,
    help="Select how many analyst agents will participate in the research."
)

st.sidebar.success("Backend: Online")

# =========================
# Main Page
# =========================
st.title("🚀 Agentic AI Research Assistant")
st.markdown("Enter a topic below to launch your concurrent research agents.")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to research?"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        full_response = (
            f"I have received your request to research: "
            f"**{prompt}** using **{max_analysts} analyst(s)**. "
            f"(Backend integration coming Friday!)"
        )

        response_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    logger.info(
        f"UI received input: {prompt} | max_analysts={max_analysts}"
    )