from src.agents import ask_database


questions = [
    "What tables exist in the database?",

    "show me the schema of the employees table?",

    "Which employees earn more than 50000?"
]


for q in questions:
    print("\nQUESTION:")
    print(q)

    print("\nANSWER:")
    print(ask_database(q))