from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.logger import logger

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    "You are a Senior Researcher. Analyze the following topic step-by-step: {topic}"
)

parser = StrOutputParser()

research_chain = prompt | model | parser

def run_research(topic: str):
    logger.info(f"🚀 Starting research chain for: {topic}")
    return research_chain.invoke({"topic": topic})

if __name__ == "__main__":
    print(run_research("The future of Agentic AI in 2026"))