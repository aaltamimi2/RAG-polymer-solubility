import sys
import os
from pathlib import Path

# Add src to python path so we can import strap package
root_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(root_dir))

from strap.vendor.biosteam_runner import run_single_simulation
from strap.services.biosteam_service import build_single_config
import json

def test_single_biosteam_simulation():
    print("--- Running BioSTEAM Simulation Test ---")
    
    # 1. Build a configuration for PE with Toluene 
    config = build_single_config(
        solvent="Toluene",
        target_plastic="PE",
        target_plastic_percent=60.0,
        processing_capacity=4800,  # e.g., 60% of 8000
        energy_case="C1",
    )
    
    print(f"Executing config: {config.get('label')}")
    
    # 2. Run simulation
    try:
        result = run_single_simulation(config)
        
        # 3. Print the result
        print("\n--- Final Output ---")
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            print("\nSUCCESS! The biosteam module is fully functional.")
        else:
            print("\nWARNING: The simulation completed but marked success=False internally.")
            
    except Exception as e:
        print("\nFATAL EXCEPTION:")
        print(f"BioSTEAM crashed: {e}")

if __name__ == "__main__":
    test_single_biosteam_simulation()
