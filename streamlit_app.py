import streamlit as st
from streamlit import config as _config
from streamlit import config_util, development, env_util, file_util, util
from streamlit import cli_util, url_util
import requests
from src.logger import logger

st.set_page_config(page_title="Agentic Researcher AI", page_icon="🤖")
st.title("Resilient Research Agent")

# =========================
# Sidebar
# =========================
st.sidebar.header("Research Settings")

max_analysts = st.sidebar.slider("Specialist Agents", 1, 5, 3)
st.sidebar.info("Logs: logs/agent_execution.log")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Research topic..."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post("http://127.0.0.1:8000/stream-research",
                                     json={"topic": prompt,
                                           "max_analysts": max_analysts},
                                           stream=True,
                                           timeout=30)
            response.raise_for_status()
            def stream_handler():
                for chunk in response.iter_content(chunk_size=None,
                                                   decode_unicode=True):
                    yield chunk

                full_response = st.write_stream(stream_handler())

                st.session_state.messaegs.append({"role":"assistant", "content": full_response})
                logger.info(f"UI Successfully rendered response for: {prompt}")
        except requests.exceptions.RequestException as e:
            error_msg = f" Connection Failed: {str(e)}"
            st.error(error-msg)
            logger.error(f"UI Connection Error: {str(e)}")        