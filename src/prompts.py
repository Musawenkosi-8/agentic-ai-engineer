EXPERT_SYSTEM_PROMPT = """You are a Senior Industrial Compliance Auditor. 
You analyze site reports and must provide a reasoning trace followed by a structured JSON object."""

FEW_SHOT_AUDIT = """
User: Report: 'Pressure valve on Boiler 4 is leaking steam. Minor rust on support legs.'
Thought: 
1. A leaking valve on a boiler is a direct safety hazard and efficiency loss.
2. Minor rust on support legs is a long-term maintenance issue but not immediate failure.
3. Priority is the valve. Cost of valve replacement and labor is approx $1,200.
Conclusion:
{
  "reasoning": "Primary concern is the valve leak which poses immediate safety risks. Rust is secondary.",
  "severity_level": "High",
  "detected_risks": ["Steam explosion", "Scalding", "Structural corrosion"],
  "estimated_cost_usd": 1200.00
}
"""

def get_intent_prompt(user_input: str):
    return f"""
{EXPERT_SYSTEM_PROMPT}

Here is an example:

{FEW_SHOT_AUDIT}

Now analyze this report:

User:
{user_input}
"""

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