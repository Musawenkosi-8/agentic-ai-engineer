import os
import uuid

import streamlit as st

from src.logger import logger
from src.ingestor import ingest_source
from src.vector_store import add_to_memory
from src.rag_graph import graph


# =====================================================
# Configuration
# =====================================================

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# Session Initialization
# =====================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []



# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title(
        "🧠 Agent Control Panel"
    )


    st.divider()


    st.subheader(
        "Conversation Memory"
    )


    st.info(
        f"Thread ID:\n{st.session_state.thread_id}"
    )


    if st.button(
        "New Conversation"
    ):

        st.session_state.thread_id = str(
            uuid.uuid4()
        )

        st.session_state.messages = []

        st.success(
            "New memory thread created."
        )


    st.divider()


    st.subheader(
        "Knowledge Ingestion"
    )


    uploaded_file = st.file_uploader(
        "Upload PDF or CSV",
        type=[
            "pdf",
            "csv"
        ]
    )


    url = st.text_input(
        "Website URL"
    )


    if st.button(
        "Index Knowledge"
    ):

        source = None


        if uploaded_file:

            file_path = os.path.join(
                "data",
                uploaded_file.name
            )


            os.makedirs(
                "data",
                exist_ok=True
            )


            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )


            source = file_path


        elif url:

            source = url


        else:

            st.warning(
                "Provide a file or URL."
            )


        if source:

            with st.spinner(
                "Processing knowledge source..."
            ):


                chunks = ingest_source(
                    source
                )


                if chunks:


                    for index, chunk in enumerate(chunks):

                        add_to_memory(
                            text=chunk.page_content,
                            doc_id=f"{source}_{index}",
                            metadata=chunk.metadata,
                            priority="Medium"
                        )


                    st.success(
                        f"Indexed {len(chunks)} chunks."
                    )


                    logger.info(
                        f"Indexed {len(chunks)} chunks from {source}"
                    )


                else:

                    st.error(
                        "No chunks created."
                    )



# =====================================================
# Main UI
# =====================================================

st.title(
    "🚀 Agentic Multi-Format RAG Assistant"
)


st.markdown(
    """
Ask questions against your private knowledge base.

Supported sources:

- PDF
- CSV
- Websites

Powered by:

- LangGraph
- ChromaDB
- Groq Llama 3.3
- SqliteSaver
"""
)



# =====================================================
# Display Conversation
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# =====================================================
# Chat Input
# =====================================================

if question := st.chat_input(
    "Ask your knowledge base..."
):


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )



    with st.chat_message(
        "assistant"
    ):


        with st.spinner(
            "Thinking..."
        ):


            try:

                response = graph.invoke(

                    {
                        "question": question
                    },

                    config={

                        "configurable": {

                            "thread_id":
                            st.session_state.thread_id

                        }

                    }

                )


                answer = response.get(
                    "answer",
                    "No answer generated."
                )


            except Exception as e:


                logger.exception(
                    f"Graph execution failed: {e}"
                )


                answer = (
                    "An error occurred while "
                    "processing your request."
                )


            st.markdown(
                answer
            )


    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )