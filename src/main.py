from src.llm_handler import ask_llm
from src.prompts import get_intent_prompt

user_input = input("Enter your question: ")
prompt = get_intent_prompt(user_input)
intent = ask_llm(prompt)

print(f"Intent: {intent}")