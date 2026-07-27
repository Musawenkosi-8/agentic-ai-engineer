from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from src.logger import logger
import os

from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a resilient AI Agent. You remember every detail of the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | model

# 2. Senior Mindset: Survival via SQL Persistence
# This ensures memory lives in memory.db, not just RAM [5, 9]
def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///memory.db",
    )

# 3. Wrapping with History Logic
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def chat_with_agent(session_id: str, question: str):
    try:
        logger.info(f"💬 [Session {session_id}] Requesting Groq...")
        response = runnable_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content
    except Exception as e:
        logger.error(f"🚨 API Failure: {e}. State is preserved in memory.db.")
        return "I encountered a temporary connection issue, but I still remember our conversation. Please try again in a moment."

if __name__ == "__main__":
    session_id = input("Session ID: ")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            break

        answer = chat_with_agent(session_id, question)
        print(f"Agent: {answer}")   