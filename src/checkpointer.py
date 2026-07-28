"""
checkpointer.py

Provides a shared LangGraph SQLite checkpointer.

The checkpointer automatically persists the graph state after
each node execution, enabling durable conversations that can
survive application restarts.
"""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from src.logger import logger


DB_PATH = "checkpoints.db"


# Create persistent SQLite connection
conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)


# Create actual LangGraph checkpoint saver
checkpointer = SqliteSaver(
    conn
)


logger.info(
    "✅ LangGraph SQLite checkpointer initialized: checkpoints.db"
)