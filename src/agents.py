from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

from src.logger import logger
from src.config import GROQ_API_KEY
load_dotenv()


# 1. Setup Database
# SQLite database for local development
# For production, use a read-only database user.
db = SQLDatabase.from_uri(
    "sqlite:///data/company.db"
)


# 2. Setup LLM Brain
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
    max_retries=2
)

# 3. Create SQL Toolkit
# Generates tools:
# - sql_db_list_tables
# - sql_db_schema
# - sql_db_query_checker
# - sql_db_query

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# 4. Agent Instructions
system_message = """
You are a SQL database analyst.

Workflow:
1. Use sql_db_list_tables to discover tables.
2. Use sql_db_schema only for relevant tables.
3. Write SELECT queries only.
4. Execute the query.
5. After receiving the database result, provide the final answer.

IMPORTANT:
Do not repeatedly inspect schemas.
Do not call tools after obtaining the answer.
Do not modify the database.
"""


# 5. Create ReAct SQL Agent
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_message
)


# 6. Public Function
def ask_database(question: str):
    """
    Send a natural language question to the SQL agent.
    """

    logger.info(f"📊 Database Question: {question}")

    try:
        response = agent_executor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },
    config={
        "recursion_limit": 5
    }
)
        


    except Exception as e:
        logger.exception(
            f"🚨 SQL Agent Failure: {e}"
        )

        return (
            "I was unable to complete the database request. "
            "Please check the query or database configuration."
        )