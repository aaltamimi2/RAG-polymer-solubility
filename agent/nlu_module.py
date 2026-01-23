def parse_prompt(user_prompt, context):
    # Improved: Detects comparison intent and extracts multiple capacities
    # This is a conceptual example; actual implementation would be more robust.
    comparison_keywords = ["compare", "comparison", "across", "different"] # etc.
    if any(kw in user_prompt.lower() for kw in comparison_keywords) or context.get("comparison_requested"):
        # Attempt to extract multiple capacities (e.g., '1000 kg/hr, 5000 kg/hr, 10000 kg/hr')
        capacities = extract_capacities_from_text(user_prompt)
        if capacities:
            return "compare_economics_visual", {"polymers": [context.get("polymer", "PE")], "capacities": capacities}

    # Fallback to single visual if no comparison intent detected
    if "visual" in user_prompt:
        polymer = context.get("polymer", "PE")
        capacity_range = context.get("capacity_range", "1 - 10 mt/yr") # Or a single default capacity
        return "show_economics_visual", {"polymer": polymer, "capacity_range": capacity_range}
    # ... other intents

def extract_capacities_from_text(text):
    # Placeholder for actual regex/NLP to parse '1000 kg/hr, 5000 kg/hr, 10000 kg/hr'
    # Returns a list of parsed capacity values (e.g., [1000, 5000, 10000])
    import re
    matches = re.findall(r'(\d+)\s*kg/hr', text, re.IGNORECASE)
    return [int(m) for m in matches] if matches else []