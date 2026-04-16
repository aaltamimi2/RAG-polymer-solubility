from strap.vendor.biosteam_runner import run_single_simulation, build_single_config
import json

config = build_single_config("Toluene", "PE", target_plastic_percent=60, processing_capacity=20000)
res = run_single_simulation(config)
print(json.dumps(res, indent=2))
