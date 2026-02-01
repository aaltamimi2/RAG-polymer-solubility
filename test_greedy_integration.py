"""Test the integrated greedy separation algorithm."""
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def test():
    from agent_sql_final_1212_patched import plan_sequential_separation

    # Original 10 polymers from user query (without PMMA, with PA66/6)
    polymers = 'PS,PVC,LDPE,HDPE,PP,EVOH,PA6,PA66,PET'

    print("="*70)
    print("Testing Greedy Separation for 9 Polymers")
    print("="*70)
    print(f"Polymers: {polymers}")
    print("="*70 + "\n")

    result = await plan_sequential_separation.ainvoke({
        'polymers': polymers,
        'temperature': 80.0,
        'top_k_solvents': 3
    })

    print(result)

if __name__ == "__main__":
    asyncio.run(test())
