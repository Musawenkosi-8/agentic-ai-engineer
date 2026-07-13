import asyncio
import time
from groq import AsyncGroq
from .config import Config

client = AsyncGroq(api_key=Config.GROQ_API_KEY)

async def ask_agent(expert_type, topic):
    """Specialized Agent Node"""
    print(f"{expert_type} agent is analyzing {topic}...")
    prompt = f"As a {expert_type}, what is the main impact of {topic}?"  
    response = await client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100)
    return f"{expert_type.upper()}: {response.choices[0].message.content}"

async def run_concurrent_research(topic):
    start_time = time.perf_counter()
    experts = ["Economist", "Technologist", "Ethicist"]

    print(f"Launching {len(experts)} agents concurrently...")

    results = await asyncio.gather(*[ask_agent(expert, topic) for expert in experts]
                                    )
    end_time = time.perf_counter()

    print("\n--- Consolidated Research Report---")

    for r in results:
        print(r)
    print(f"\n Total time with Asyncio: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(run_concurrent_research("Nuclear Fusion"))
    
              
