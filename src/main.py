from fastapi import FastAPI
from src.llm_handler import ask_llm

app = FastAPI(title="Agentic AI Research Assistant")


@app.get("/")
def home():
    return {
        "message": "Agentic AI API is running"
    }


@app.post("/ask")
def ask_question(question: str):
    answer = ask_llm(question)

    return {
        "question": question,
        "answer": answer
    }