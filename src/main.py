from .llm_handler import ask_llm
from .prompts import get_intent_prompt

question = input("Enter your question: ")
prompt = get_intent_prompt(user_input)
answer = ask_llm(prompt)

print(f"Intent: {intent}")