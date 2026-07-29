from src.agent_runner import run_agent


result = run_agent(
    """
    Research the current Bitcoin price.
    Write a one sentence summary.
    Save it into crypto.txt.
    """
)


print("\n===== FINAL ANSWER =====")
print(result)