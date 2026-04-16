import sys
import os
from pathlib import Path

# Set working directory to repo root
target_dir = Path(__file__).resolve().parent
os.chdir(target_dir)

# Add src to path
sys.path.insert(0, str(target_dir / "src"))

from strap.agent import create_dissolve_agent, _extract_text
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    
    print("Loading DISSOLVE agent...")
    agent = create_dissolve_agent()
    
    test_prompt = """I have 8000 tonnes of mixed plastic waste.
The mix is:
- 60 % PE
- 20 % PET
- 10 % Nylon‑6
- 10 % EVOH
Please run the DISSOLVE optimizer (scenario A) and tell me:
1. The circularity score (a number between 0 and 1).
2. Which processing steps the model chose (separation, conversion, end‑of‑life).
3. A short, plain‑English reason why it picked those steps.
Give the answer as a short paragraph or a tiny table – nothing technical.
"""
    print(f"\n[USER PROMPT]: {test_prompt}\n")
    print("Agent is thinking... (running tool calls!)\n")
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": test_prompt}]},
        config={"recursion_limit": 150}
    )
    
    answer = None
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            answer = _extract_text(msg.content)
            break
            
    print("\n" + "="*50)
    print("--- DISSOLVE AGENT FINAL RESPONSE ---")
    print("="*50 + "\n")
    print(answer)
