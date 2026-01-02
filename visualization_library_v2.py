"""
Clean Visualization Library v2
================================

Simple, clear visualizations - one concept per visual.
Optimized for chatbot integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def create_hsp_radar_plot(polymer_hsp, solvent_hsp, polymer_name, solvent_name,
                          prediction, probability, output_path):
    """
    Create clean radar plot showing HSP parameter overlap.
    One visual - just the radar plot with parameters.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')

    # Parameters
    categories = ['Dispersion\n(δD)', 'Polar\n(δP)', 'Hydrogen\n(δH)']
    N = len(categories)

    # Values
    polymer_values = [
        polymer_hsp['Dispersion'],
        polymer_hsp['Polar'],
        polymer_hsp['Hydrogen']
    ]
    solvent_values = [
        solvent_hsp['Dispersion'],
        solvent_hsp['Polar'],
        solvent_hsp['Hydrogen']
    ]

    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    polymer_values += polymer_values[:1]
    solvent_values += solvent_values[:1]
    angles += angles[:1]

    # Plot
    ax.plot(angles, polymer_values, 'o-', linewidth=2.5, label=polymer_name,
            color='#1f77b4', markersize=8)
    ax.fill(angles, polymer_values, alpha=0.25, color='#1f77b4')

    ax.plot(angles, solvent_values, 'o-', linewidth=2.5, label=solvent_name,
            color='#ff7f0e', markersize=8)
    ax.fill(angles, solvent_values, alpha=0.25, color='#ff7f0e')

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, max(max(polymer_values), max(solvent_values)) * 1.2)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    # Title
    pred_text = "SOLUBLE" if prediction else "NON-SOLUBLE"
    color = '#2ca02c' if prediction else '#d62728'
    plt.title(f'{polymer_name} + {solvent_name}\n'
              f'Prediction: {pred_text} ({probability*100:.1f}%)',
              fontsize=13, fontweight='bold', pad=20, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved radar plot: {output_path}")


def create_red_gauge(polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
                     prediction, probability, output_path):
    """
    Create simple RED gauge visualization.
    Shows RED value on a gauge from 0 to 2.
    """
    # Calculate RED
    delta_d = polymer_hsp['Dispersion'] - solvent_hsp['Dispersion']
    delta_p = polymer_hsp['Polar'] - solvent_hsp['Polar']
    delta_h = polymer_hsp['Hydrogen'] - solvent_hsp['Hydrogen']

    ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)
    red = ra / r0 if r0 > 0 else float('inf')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw gauge background
    gauge_y = 0.5
    gauge_height = 0.2

    # Background (gray)
    ax.add_patch(patches.Rectangle((0.1, gauge_y), 2.0, gauge_height,
                                    facecolor='#f0f0f0', edgecolor='black', linewidth=2))

    # Soluble region (green)
    ax.add_patch(patches.Rectangle((0.1, gauge_y), 0.9, gauge_height,
                                    facecolor='#2ca02c', alpha=0.3, edgecolor='none'))

    # Non-soluble region (red)
    ax.add_patch(patches.Rectangle((1.0, gauge_y), 1.1, gauge_height,
                                    facecolor='#d62728', alpha=0.3, edgecolor='none'))

    # Threshold line at RED = 1.0
    ax.plot([1.0, 1.0], [gauge_y - 0.05, gauge_y + gauge_height + 0.05],
            'b--', linewidth=3, label='RED = 1.0 Threshold')

    # RED value indicator
    red_display = min(red, 2.0)
    indicator_x = 0.1 + red_display
    ax.plot([indicator_x, indicator_x], [gauge_y - 0.1, gauge_y + gauge_height + 0.1],
            'black', linewidth=4, marker='v', markersize=15, markerfacecolor='black')

    # Labels
    ax.text(0.55, gauge_y + gauge_height + 0.2, 'SOLUBLE', ha='center', fontsize=12,
            fontweight='bold', color='#2ca02c')
    ax.text(1.55, gauge_y + gauge_height + 0.2, 'NON-SOLUBLE', ha='center', fontsize=12,
            fontweight='bold', color='#d62728')

    # Scale labels
    ax.text(0.1, gauge_y - 0.15, '0.0', ha='center', fontsize=10)
    ax.text(1.0, gauge_y - 0.15, '1.0', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.1, gauge_y - 0.15, '2.0', ha='center', fontsize=10)

    # RED value
    ax.text(1.1, 0.85, f'RED = {red:.3f}', ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))

    # Title
    ax.text(1.1, 0.15, f'{polymer_name} + {solvent_name}',
            ha='center', fontsize=13, fontweight='bold')

    # Prediction
    pred_text = f"ML Prediction: {'SOLUBLE' if prediction else 'NON-SOLUBLE'} ({probability*100:.1f}%)"
    color = '#2ca02c' if prediction else '#d62728'
    ax.text(1.1, 0.05, pred_text, ha='center', fontsize=11, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved RED gauge: {output_path}")


def create_red_sphere_3d(polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
                         prediction, probability, output_dir):
    """
    Create interactive 3D RED sphere visualization (HTML).
    Shows polymer interaction sphere and solvent point.
    """
    # Extract values
    p_d = polymer_hsp['Dispersion']
    p_p = polymer_hsp['Polar']
    p_h = polymer_hsp['Hydrogen']

    s_d = solvent_hsp['Dispersion']
    s_p = solvent_hsp['Polar']
    s_h = solvent_hsp['Hydrogen']

    # Calculate distance
    delta_d = p_d - s_d
    delta_p = p_p - s_p
    delta_h = p_h - s_h
    ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)
    red = ra / r0 if r0 > 0 else float('inf')

    # Create sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = p_d + r0 * np.outer(np.cos(u), np.sin(v))
    y_sphere = p_p + r0 * np.outer(np.sin(u), np.sin(v))
    z_sphere = p_h + r0 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Create figure
    fig = go.Figure()

    # Add polymer sphere
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.3,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        name='Polymer Interaction Sphere',
        hovertemplate='Polymer Sphere<br>R0=%{text}<extra></extra>',
        text=[[f'{r0:.2f}'] * y_sphere.shape[1]] * y_sphere.shape[0]
    ))

    # Add polymer center point
    fig.add_trace(go.Scatter3d(
        x=[p_d], y=[p_p], z=[p_h],
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='diamond'),
        text=[polymer_name],
        textposition='top center',
        name='Polymer Center',
        hovertemplate=f'{polymer_name}<br>D=%{{x:.2f}}<br>P=%{{y:.2f}}<br>H=%{{z:.2f}}<extra></extra>'
    ))

    # Add solvent point
    solvent_color = 'green' if red < 1.0 else 'red'
    solvent_status = 'INSIDE (Soluble)' if red < 1.0 else 'OUTSIDE (Non-Soluble)'

    fig.add_trace(go.Scatter3d(
        x=[s_d], y=[s_p], z=[s_h],
        mode='markers+text',
        marker=dict(size=12, color=solvent_color, symbol='circle'),
        text=[solvent_name],
        textposition='bottom center',
        name=f'Solvent ({solvent_status})',
        hovertemplate=f'{solvent_name}<br>D=%{{x:.2f}}<br>P=%{{y:.2f}}<br>H=%{{z:.2f}}<br>Ra={ra:.2f}<br>RED={red:.3f}<extra></extra>'
    ))

    # Add distance line
    fig.add_trace(go.Scatter3d(
        x=[p_d, s_d], y=[p_p, s_p], z=[p_h, s_h],
        mode='lines',
        line=dict(color='gray', width=3, dash='dash'),
        name=f'Distance (Ra={ra:.2f})',
        hovertemplate=f'Ra=%{{text:.2f}}<extra></extra>',
        text=[ra, ra]
    ))

    # Layout
    pred_text = f"SOLUBLE ({probability*100:.1f}%)" if prediction else f"NON-SOLUBLE ({probability*100:.1f}%)"
    title_color = 'green' if prediction else 'red'

    fig.update_layout(
        title=dict(
            text=f'<b>{polymer_name} + {solvent_name}</b><br>'
                 f'<span style="color:{title_color}">ML Prediction: {pred_text}</span><br>'
                 f'<span style="font-size:12px">RED = {red:.3f} (Ra={ra:.2f}, R0={r0:.2f})</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis_title='Dispersion (δD)',
            yaxis_title='Polar (δP)',
            zaxis_title='Hydrogen (δH)',
            aspectmode='cube'
        ),
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
        width=900,
        height=700
    )

    # Save
    output_path = Path(output_dir) / 'red_sphere_3d.html'
    fig.write_html(str(output_path))

    print(f"  ✓ Saved 3D sphere: {output_path}")


def create_hsp_comparison_bars(polymer_hsp, solvent_hsp, polymer_name, solvent_name,
                               prediction, probability, output_path):
    """
    Create side-by-side bar comparison of HSP parameters.
    Simple and clear.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    parameters = ['Dispersion\n(δD)', 'Polar\n(δP)', 'Hydrogen\n(δH)']
    polymer_vals = [polymer_hsp['Dispersion'], polymer_hsp['Polar'], polymer_hsp['Hydrogen']]
    solvent_vals = [solvent_hsp['Dispersion'], solvent_hsp['Polar'], solvent_hsp['Hydrogen']]

    x = np.arange(len(parameters))
    width = 0.35

    bars1 = ax.bar(x - width/2, polymer_vals, width, label=polymer_name,
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, solvent_vals, width, label=solvent_name,
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize
    ax.set_ylabel('Hansen Parameter Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{polymer_name} + {solvent_name}\nHansen Solubility Parameters',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(parameters, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # Add prediction
    pred_text = f"ML Prediction: {'SOLUBLE' if prediction else 'NON-SOLUBLE'} ({probability*100:.1f}%)"
    color = '#2ca02c' if prediction else '#d62728'
    ax.text(0.5, 0.95, pred_text, transform=ax.transAxes,
           ha='center', va='top', fontsize=11, fontweight='bold',
           color=color, bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 edgecolor=color, linewidth=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved HSP comparison: {output_path}")


def create_prediction_summary(polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
                              prediction, probability, output_path):
    """
    Create text summary card with all key information.
    """
    # Calculate metrics
    delta_d = polymer_hsp['Dispersion'] - solvent_hsp['Dispersion']
    delta_p = polymer_hsp['Polar'] - solvent_hsp['Polar']
    delta_h = polymer_hsp['Hydrogen'] - solvent_hsp['Hydrogen']

    ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)
    red = ra / r0 if r0 > 0 else float('inf')

    # Create text content
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║ POLYMER-SOLVENT SOLUBILITY PREDICTION                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Polymer:  {polymer_name:50} ║
║ Solvent:  {solvent_name:50} ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ PREDICTION                                                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Result:       {'✓ SOLUBLE' if prediction else '✗ NON-SOLUBLE':50} ║
║ Probability:  {probability*100:5.1f}%{' '*44} ║
║ Confidence:   {'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low':50} ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ HANSEN SOLUBILITY PARAMETERS                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Polymer HSPs:                                                ║
║   Dispersion (δD):  {polymer_hsp['Dispersion']:5.1f}{' '*38} ║
║   Polar (δP):       {polymer_hsp['Polar']:5.1f}{' '*38} ║
║   Hydrogen (δH):    {polymer_hsp['Hydrogen']:5.1f}{' '*38} ║
║                                                              ║
║ Solvent HSPs:                                                ║
║   Dispersion (δD):  {solvent_hsp['Dispersion']:5.1f}{' '*38} ║
║   Polar (δP):       {solvent_hsp['Polar']:5.1f}{' '*38} ║
║   Hydrogen (δH):    {solvent_hsp['Hydrogen']:5.1f}{' '*38} ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ RED CALCULATION                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Parameter Differences:                                       ║
║   ΔD = {delta_d:+6.2f}{' '*43} ║
║   ΔP = {delta_p:+6.2f}{' '*43} ║
║   ΔH = {delta_h:+6.2f}{' '*43} ║
║                                                              ║
║ Hansen Distance:                                             ║
║   Ra = sqrt(4×ΔD² + ΔP² + ΔH²) = {ra:6.2f}{' '*21} ║
║                                                              ║
║ Interaction Radius:                                          ║
║   R0 = {r0:6.2f}{' '*43} ║
║                                                              ║
║ RED (Relative Energy Difference):                            ║
║   RED = Ra / R0 = {red:6.3f}{' '*32} ║
║   Threshold: RED < 1.0 for solubility                        ║
║   Theory prediction: {'SOLUBLE' if red < 1.0 else 'NON-SOLUBLE':40} ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ RECOMMENDATION                                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
"""

    if prediction and red < 1.0:
        rec = "Both ML model and Hansen theory predict solubility."
        rec2 = "High confidence - proceed with application."
    elif prediction and red >= 1.0:
        rec = "ML predicts soluble but theory suggests non-soluble."
        rec2 = "Medium confidence - recommend experimental validation."
    elif not prediction and red < 1.0:
        rec = "Theory predicts soluble but ML suggests non-soluble."
        rec2 = "Medium confidence - recommend experimental validation."
    else:
        rec = "Both ML model and Hansen theory predict non-solubility."
        rec2 = "High confidence - not recommended for application."

    summary += f"║ {rec:60} ║\n"
    summary += f"║ {rec2:60} ║\n"
    summary += "║                                                              ║\n"
    summary += "╚══════════════════════════════════════════════════════════════╝\n"

    # Save to file
    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"  ✓ Saved summary: {output_path}")


# Convenience function to generate all visualizations
def generate_all_visualizations(polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
                                prediction, probability, output_dir):
    """Generate all visualization types for a prediction."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Radar plot (HSP overlap)
    create_hsp_radar_plot(
        polymer_hsp, solvent_hsp, polymer_name, solvent_name,
        prediction, probability,
        output_dir / 'radar_plot.png'
    )

    # 2. RED gauge
    create_red_gauge(
        polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
        prediction, probability,
        output_dir / 'red_gauge.png'
    )

    # 3. 3D sphere (HTML)
    create_red_sphere_3d(
        polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
        prediction, probability,
        output_dir
    )

    # 4. HSP comparison bars
    create_hsp_comparison_bars(
        polymer_hsp, solvent_hsp, polymer_name, solvent_name,
        prediction, probability,
        output_dir / 'hsp_comparison.png'
    )

    # 5. Text summary
    create_prediction_summary(
        polymer_hsp, solvent_hsp, r0, polymer_name, solvent_name,
        prediction, probability,
        output_dir / 'summary.txt'
    )

    print(f"\n✓ Generated all visualizations in: {output_dir}")

    # Return paths to all generated files
    return {
        'Radar Plot': str(output_dir / 'radar_plot.png'),
        'RED Gauge': str(output_dir / 'red_gauge.png'),
        '3D Sphere (Interactive HTML)': str(output_dir / 'red_sphere_3d.html'),
        'HSP Comparison': str(output_dir / 'hsp_comparison.png'),
        'Text Summary': str(output_dir / 'summary.txt')
    }
