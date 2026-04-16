"""
Full pipeline test: BioSTEAM -> Excel update -> Pyomo optimization.
Run from the RAG-polymer-solubility/ directory.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from strap.tools.waste_optimization import run_waste_management_optimization

print("=" * 60)
print(" FULL PIPELINE TEST: BioSTEAM → Excel → Pyomo")
print("=" * 60)
print()
print("Inputs:")
print("  Feed           : 8000 tonnes/year")
print("  PE fraction    : 0.60")
print("  PET fraction   : 0.20")
print("  N6 fraction    : 0.10")
print("  EVOH fraction  : 0.10")
print("  Scenario       : A")
print("  Objective      : max_profit")
print()
print("Running... (BioSTEAM sims take ~30-60s per solvent)")
print("-" * 60)

result = run_waste_management_optimization(
    feed=8000,
    pe_fraction=0.60,
    pet_fraction=0.20,
    n6_fraction=0.10,
    evoh_fraction=0.10,
    scenario="A",
    objective="max_profit",
)

print()
print("=" * 60)
print(" RAW TOOL RESPONSE")
print("=" * 60)
# result may be a dict or a string depending on how safe_tool_wrapper formats it
if isinstance(result, dict):
    # Try to pretty-print the display text and the structured data separately
    display = result.get("content", result.get("display", str(result)))
    data    = result.get("data", {})
    print(display)
    print()
    print("--- Structured Data ---")
    print(json.dumps(data, indent=2))
else:
    print(result)
