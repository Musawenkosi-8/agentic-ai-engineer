# Standard persona and instruction
SYSTEM_PROMPT = """You are a highly precise Intent Classifier. 
Your job is to categorize user input into one of three labels: RESEARCH, CALCULATION, or GREETING.
Return ONLY the label."""

# Few-shot examples to "teach" the model the pattern
FEW_SHOT_EXAMPLES = """
User: Hello there, how are you?
Label: GREETING

User: What is the square root of 144?
Label: CALCULATION

User: Can you find the latest papers on Quantum Computing?
Label: RESEARCH

User: Hi!
Label: GREETING
"""

def get_intent_prompt(user_input: str):
    """Combines the system instruction, few-shot examples, and the new query."""
    return f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}\nUser: {user_input}\nLabel:"