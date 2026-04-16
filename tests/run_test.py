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
    
    test_prompt = """I have 8000 tonnes of plastic waste consisting of 40% PE, 40% PET, 1% Nylon‑6, and 19% EVOH. 
Run the optimization using SCENARIO_NAME and tell me the best processing pathway, the total profit, and the emissions.
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
