import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def create_single_plot(data, title):
    # ... existing implementation ...
    plot_path = f"./plots/{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    # Example: plt.figure(); plt.plot(data['scale'], data['operating_cost']); plt.title(title); plt.savefig(plot_path)
    return plot_path

def create_comparative_plot(list_of_analysis_results):
    # list_of_analysis_results is expected to be like: 
    # [{'polymer': 'PE', 'capacity': 1000, 'data': [...], 'insights': [...]}, ...]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    all_insights = set()
    
    for result in list_of_analysis_results:
        if 'data' in result:
            df = pd.DataFrame(result['data'])
            label = f"{result['polymer']} @ {result['capacity']} kg/hr"
            ax.plot(df['scale'], df['operating_cost'], label=label)
            if 'insights' in result: all_insights.update(result['insights'])

    ax.set_xlabel("Scale")
    ax.set_ylabel("Unit Operating Cost")
    ax.set_title("Comparative STRAP Scale Economics Analysis")
    ax.legend()
    ax.grid(True)
    
    comparative_plot_path = f"./plots/comparative_strap_scale_economics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(comparative_plot_path)
    plt.close(fig)
    
    combined_insights = list(all_insights)
    return comparative_plot_path, combined_insights