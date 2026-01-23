import tools.economics_analyzer
import utils.plot_generator
import concurrent.futures # For parallel execution

def process_request(user_prompt, context):
    intent, params = nlu_module.parse_prompt(user_prompt, context)

    if intent == "compare_economics_visual":
        polymers = params.get("polymers", ["PE"])
        capacities = params.get("capacities")

        if not capacities:
            return {"error": "Comparison requested but no specific capacities found."}

        all_results = []
        # Execute analysis for each capacity, potentially in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_params = {
                executor.submit(tools.economics_analyzer.analyze_data, p, c):
                {"polymer": p, "capacity": c}
                for p in polymers for c in capacities
            }
            for future in concurrent.futures.as_completed(future_to_params):
                current_params = future_to_params[future]
                try:
                    analysis_data, insights = future.result()
                    all_results.append({
                        "polymer": current_params["polymer"],
                        "capacity": current_params["capacity"],
                        "data": analysis_data,
                        "insights": insights
                    })
                except Exception as exc:
                    print(f"Analysis for {current_params} generated an exception: {exc}")
                    all_results.append({"error": str(exc), **current_params})
        
        # Generate a single comparative plot from all results
        comparative_plot_path, combined_insights = utils.plot_generator.create_comparative_plot(all_results)
        
        return {
            "visualization": comparative_plot_path,
            "key_insights": combined_insights,
            "individual_results": all_results # Optional: for detailed breakdown
        }

    elif intent == "show_economics_visual":
        polymer = params.get("polymer")
        capacity = params.get("capacity_range") # Assuming this is parsed to a single value or range
        
        # Call tool for single analysis
        analysis_data, insights = tools.economics_analyzer.analyze_data(polymer, capacity)
        plot_path = utils.plot_generator.create_single_plot(analysis_data, f"STRAP Scale Economics ({polymer})")
        
        return {"visualization": plot_path, "key_insights": insights}
    # ... other intents