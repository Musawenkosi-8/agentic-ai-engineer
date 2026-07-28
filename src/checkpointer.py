"""
checkpointer.py

Provides a shared LangGraph SQLite checkpointer.

The checkpointer automatically persists the graph state after
each node execution, enabling durable conversations that can
survive application restarts.
"""

from langgraph.checkpoint.sqlite import SqliteSaver

from src.logger import logger


CHECKPOINT_DB = "checkpoints.db"


try:

    checkpointer = SqliteSaver.from_conn_string(
        CHECKPOINT_DB
    )

    logger.info(
        f"✅ LangGraph checkpointer initialized: {CHECKPOINT_DB}"
    )

except Exception as e:

    logger.exception(
        f"❌ Failed to initialize LangGraph checkpointer: {e}"
    )

    raise