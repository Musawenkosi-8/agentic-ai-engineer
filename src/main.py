from .llm_handler import ask_llm

question = input("Enter your question: ")
answer = ask_llm(question)

print(f"Answer: {answer}")