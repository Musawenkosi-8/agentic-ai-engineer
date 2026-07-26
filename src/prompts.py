from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# INDUSTRIAL COMPLIANCE AUDITOR PROMPT
# ==========================================================

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
  "detected_risks": [
    "Steam explosion",
    "Scalding",
    "Structural corrosion"
  ],
  "estimated_cost_usd": 1200.00
}
"""

audit_prompt = ChatPromptTemplate.from_template(
    """
You are a {persona}.

You analyze site reports and must provide a reasoning trace followed by a structured JSON object.

Here is an example:

{example}

Now analyze this report:

User:
{topic}
"""
)

# ==========================================================
# INTENT CLASSIFIER PROMPT
# ==========================================================

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

intent_prompt = ChatPromptTemplate.from_template(
    """
You are a {persona}.

Your job is to categorize user input into one of three labels:

- RESEARCH
- CALCULATION
- GREETING

Return ONLY the label.

Examples:

{examples}

User: {topic}

Label:
"""
)

# ==========================================================
# EXAMPLES
# ==========================================================

# Industrial Audit Prompt
audit_messages = audit_prompt.invoke(
    {
        "persona": "Senior Industrial Compliance Auditor",
        "example": FEW_SHOT_AUDIT,
        "topic": "Pressure valve on Boiler 7 is leaking and excessive corrosion has formed around the support frame."
    }
)

print("=== AUDIT PROMPT ===")
print(audit_messages.to_string())


print("\n" + "=" * 70 + "\n")

# Intent Classifier Prompt
intent_messages = intent_prompt.invoke(
    {
        "persona": "Highly Precise Intent Classifier",
        "examples": FEW_SHOT_EXAMPLES,
        "topic": "Find the latest research papers on Agentic AI."
    }
)

print("=== INTENT PROMPT ===")
print(intent_messages.to_string())