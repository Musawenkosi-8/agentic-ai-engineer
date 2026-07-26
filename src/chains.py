from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.logger import logger


# 1. Define the model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# 2. Define the prompt
prompt = ChatPromptTemplate.from_template(
    "You are a Senior Researcher. Analyze the following topic step-by-step: {topic}"
)


# 3. Define output parser
parser = StrOutputParser()


# 4. Create LCEL Chain -----
research_chain = prompt | model | parser


# 5. Streaming function
def stream_research(topic: str):
    logger.info(f"🚀 Starting streaming research chain for: {topic}")

    for chunk in research_chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)

    print()  # New line after streaming finishes


if __name__ == "__main__":
    stream_research("The future of Agentic AI in 2026")