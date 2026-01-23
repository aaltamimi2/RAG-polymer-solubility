def analyze_data(polymer, capacity):
    # ... perform calculations to generate raw data ...
    # Example: data could be a list of dicts or a pandas DataFrame
    simulated_data = [
        {"scale": 1, "operating_cost": 100, "capital_investment": 500},
        {"scale": 5, "operating_cost": 60, "capital_investment": 1200},
        {"scale": 10, "operating_cost": 40, "capital_investment": 1900}
    ] # Placeholder for actual analysis output
    insights = [
        "Unit Operating Cost decreases with scale",
        "Capital Investment increases sub-linearly"
    ]
    return simulated_data, insights # Return data and insights, not a plot path