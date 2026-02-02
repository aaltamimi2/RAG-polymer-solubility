#!/usr/bin/env python3
"""
Run STRAP-CORE multi-agent query with full telemetry capture.
Demonstrates multi-agent architecture for publication.
Outputs visualizations to plots/tests-1.2/
"""

import os
import sys
import asyncio
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("plots/tests-1.2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Complex STRAP-CORE integrated query (Level 4) - Uses reliable polymers PP/PS
STRAP_QUERY = """Design a two-stage STRAP process for PP/PS mixed plastic separation at 100°C.

Stage 1: Find solvents that dissolve PP at 100°C with high selectivity over PS. List the top 3 candidates.

Stage 2: Run TEA analysis at 5000 kg/hr for cyclohexane recovery. Include cost per kg and payback period.

Generate LCA comparison visualizations for the recommended solvent."""


def generate_architecture_visualizations(telemetry_data: dict, output_dir: Path):
    """Generate publication-quality visualizations from telemetry data."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import numpy as np
    except ImportError:
        print("Matplotlib not available for visualizations")
        return

    # Extract data
    routing = telemetry_data.get("routing", {})
    execution_trace = telemetry_data.get("execution_trace", {})
    agent_timings = execution_trace.get("agent_timings", {})
    elapsed = telemetry_data.get("elapsed_seconds", 0)

    # Color palette (Tableau 10)
    COLORS = {
        "router": "#4E79A7",
        "separation": "#F28E2B",
        "tea_lca": "#E15759",
        "literature": "#76B7B2",
        "aggregator": "#59A14F",
        "user": "#EDC948",
    }

    # =========================================
    # 1. Workflow Graph Visualization
    # =========================================
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Node positions
    nodes = {
        "User Query": (0, 1),
        "Router": (1, 1),
        "Separation\nAgent": (2.5, 1.7),
        "TEA/LCA\nAgent": (4, 1.7),
        "Smart\nAggregator": (5, 1),
        "Response": (6, 1),
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        # Get color
        key = name.split()[0].lower().replace("\n", "_")
        if "user" in key or "query" in key:
            color = COLORS["user"]
        elif "router" in key:
            color = COLORS["router"]
        elif "separation" in key:
            color = COLORS["separation"]
        elif "tea" in key:
            color = COLORS["tea_lca"]
        elif "aggregator" in key:
            color = COLORS["aggregator"]
        else:
            color = "#BAB0AC"

        # Draw node box
        box = FancyBboxPatch(
            (x - 0.4, y - 0.25), 0.8, 0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color, edgecolor="black", linewidth=2, alpha=0.9
        )
        ax.add_patch(box)

        # Add label
        ax.text(x, y, name, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white' if color not in ["#EDC948"] else "black")

    # Draw arrows
    arrows = [
        ("User Query", "Router", ""),
        ("Router", "Separation\nAgent", f"complexity={routing.get('complexity', 5)}"),
        ("Separation\nAgent", "TEA/LCA\nAgent", f"{agent_timings.get('separation', 0)*1000:.0f}ms"),
        ("TEA/LCA\nAgent", "Smart\nAggregator", f"{agent_timings.get('tea_lca', 0)*1000:.0f}ms"),
        ("Smart\nAggregator", "Response", ""),
    ]

    for from_node, to_node, label in arrows:
        x1, y1 = nodes[from_node]
        x2, y2 = nodes[to_node]

        # Adjust for node size
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        x1 += 0.4 * dx / length
        y1 += 0.25 * dy / length if dy != 0 else 0
        x2 -= 0.4 * dx / length
        y2 -= 0.25 * dy / length if dy != 0 else 0

        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax.text(mx, my, label, ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Title and metadata
    ax.set_title(f"Multi-Agent Workflow: {routing.get('path', 'integrated').upper()} Path\n"
                f"Total: {elapsed:.2f}s | Complexity: {routing.get('complexity', 5)}/5",
                fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["router"], label='Router'),
        mpatches.Patch(facecolor=COLORS["separation"], label='Separation Agent'),
        mpatches.Patch(facecolor=COLORS["tea_lca"], label='TEA/LCA Agent'),
        mpatches.Patch(facecolor=COLORS["aggregator"], label='Aggregator'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "workflow_graph.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "workflow_graph.svg", format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: workflow_graph.png/svg")

    # =========================================
    # 2. Agent Timing Bar Chart
    # =========================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    agents = list(agent_timings.keys())
    times = [agent_timings[a] * 1000 for a in agents]  # Convert to ms
    colors = [COLORS.get(a.lower().replace(" ", "_"), "#BAB0AC") for a in agents]

    bars = ax.barh(agents, times, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
               f'{t:.0f}ms', va='center', fontsize=10)

    ax.set_xlabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Agent Execution Timing', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(times) * 1.2 if times else 100)

    # Add total time annotation
    total_time = sum(times)
    ax.annotate(f'Total Agent Time: {total_time:.0f}ms',
               xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / "agent_timing.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: agent_timing.png")

    # =========================================
    # 3. Routing Decision Visualization
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # Panel 1: Complexity Gauge
    ax1 = axes[0]
    complexity = routing.get("complexity", 3)

    # Draw gauge
    theta = np.linspace(0, np.pi, 100)
    for i in range(1, 6):
        color = plt.cm.RdYlGn_r((i-1)/4)
        ax1.fill_between(theta[(i-1)*20:i*20], 0.6, 1.0,
                        alpha=0.3 if i != complexity else 0.9,
                        color=color)

    # Needle
    needle_theta = np.pi * (1 - (complexity - 0.5) / 5)
    ax1.plot([0, 0.9 * np.cos(needle_theta)], [0, 0.9 * np.sin(needle_theta)],
            'k-', linewidth=3)
    ax1.plot(0, 0, 'ko', markersize=10)

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'Complexity: {complexity}/5', fontsize=12, fontweight='bold')

    # Labels
    for i, label in enumerate(['1', '2', '3', '4', '5']):
        angle = np.pi * (1 - (i + 0.5) / 5)
        ax1.text(1.1 * np.cos(angle), 1.1 * np.sin(angle), label,
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Panel 2: Path Selection
    ax2 = axes[1]
    paths = ['fast', 'standard', 'specialist', 'integrated']
    selected = routing.get("path", "integrated")

    for i, path in enumerate(paths):
        color = '#59A14F' if path == selected else '#E0E0E0'
        rect = mpatches.FancyBboxPatch((0.1, 0.8 - i*0.25), 0.8, 0.2,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(0.5, 0.9 - i*0.25, path.upper(), ha='center', va='center',
                fontsize=11, fontweight='bold' if path == selected else 'normal',
                color='white' if path == selected else 'gray')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Path Selection', fontsize=12, fontweight='bold')

    # Panel 3: Specialist Assignment
    ax3 = axes[2]
    specialist = routing.get("specialist", "separation")
    specialists = ['separation', 'tea_lca', 'literature']

    for i, spec in enumerate(specialists):
        is_active = spec == specialist or (selected == 'integrated')
        color = COLORS.get(spec, '#BAB0AC') if is_active else '#E0E0E0'
        alpha = 1.0 if is_active else 0.4

        rect = mpatches.FancyBboxPatch((0.1, 0.75 - i*0.3), 0.8, 0.25,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=alpha)
        ax3.add_patch(rect)
        ax3.text(0.5, 0.875 - i*0.3, spec.replace('_', '/').upper(),
                ha='center', va='center', fontsize=10,
                fontweight='bold' if is_active else 'normal',
                color='white' if is_active else 'gray')

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Active Specialists', fontsize=12, fontweight='bold')

    plt.suptitle(f"Routing Decision: {routing.get('reason', 'Multi-domain query')[:60]}...",
                fontsize=11, style='italic', y=0.02)
    plt.tight_layout()
    plt.savefig(output_dir / "routing_decision.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: routing_decision.png")

    # =========================================
    # 4. Timeline Visualization
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    # Build timeline from agent timings
    current_time = 0
    timeline_data = []

    # Router phase (estimate ~100ms)
    timeline_data.append(("Router", 0, 0.1, COLORS["router"]))
    current_time = 0.1

    for agent, duration in agent_timings.items():
        color = COLORS.get(agent.lower().replace(" ", "_"), "#BAB0AC")
        timeline_data.append((agent.replace("_", " ").title(), current_time, duration, color))
        current_time += duration

    # Aggregator phase (estimate ~50ms)
    timeline_data.append(("Aggregator", current_time, 0.05, COLORS["aggregator"]))

    # Draw timeline bars
    for i, (name, start, duration, color) in enumerate(timeline_data):
        ax.barh(0, duration * 1000, left=start * 1000, height=0.6,
               color=color, edgecolor='black', linewidth=1, label=name)

        # Add label
        if duration > 0.1:  # Only label if wide enough
            ax.text(start * 1000 + duration * 500, 0, name,
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax.set_xlim(0, current_time * 1000 * 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Execution Timeline', fontsize=14, fontweight='bold')

    # Add markers
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Start')
    ax.axvline(x=current_time * 1000, color='red', linestyle='--', linewidth=2, label='End')

    plt.tight_layout()
    plt.savefig(output_dir / "timeline.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: timeline.png")

    print(f"\n  All visualizations saved to: {output_dir}")


async def run_query():
    """Run the STRAP query through the multi-agent system."""

    # Add current directory to path
    sys.path.insert(0, os.getcwd())

    print("\n" + "="*80)
    print("STRAP-CORE Multi-Agent Query - Publication Demo")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Start time: {datetime.now().isoformat()}")
    print("\n" + "-"*80)
    print("QUERY:")
    print("-"*80)
    print(STRAP_QUERY)
    print("-"*80 + "\n")

    # Import agent module
    import importlib.util
    import unittest.mock as mock

    agent_file = "agent_sql_final_1212_patched.py"
    if not os.path.exists(agent_file):
        print(f"ERROR: {agent_file} not found!")
        return

    print(f"Loading agent from: {agent_file}")

    # Mock gradio
    mock_gradio = mock.MagicMock()
    mock_gradio.Blocks = mock.MagicMock(return_value=mock.MagicMock())
    sys.modules['gradio'] = mock_gradio

    spec = importlib.util.spec_from_file_location("agent_module", agent_file)
    agent_module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(agent_module)
    except SystemExit:
        pass

    # Get agent components
    multi_agent_graph = getattr(agent_module, 'multi_agent_graph', None)
    agent_graph = getattr(agent_module, 'agent_graph', None)
    get_routing_info = getattr(agent_module, 'get_routing_info', None)
    create_thread_id = getattr(agent_module, 'create_thread_id', None)
    MULTI_AGENT_AVAILABLE = getattr(agent_module, 'MULTI_AGENT_AVAILABLE', False)
    MAX_ITERATIONS = getattr(agent_module, 'MAX_ITERATIONS', 15)

    from langchain_core.messages import HumanMessage

    print(f"\nAgent loaded successfully!")
    print(f"  Multi-agent available: {MULTI_AGENT_AVAILABLE}")
    print(f"  Max iterations: {MAX_ITERATIONS}")

    # Get routing info for telemetry (even if not using multi-agent)
    routing = {}
    if get_routing_info:
        routing = get_routing_info(STRAP_QUERY)
        print(f"\nRouting Analysis (for telemetry):")
        print(f"  Complexity: {routing.get('complexity', 'N/A')}")
        print(f"  Path: {routing.get('path', 'N/A')}")
        print(f"  Specialist: {routing.get('specialist', 'N/A')}")
        print(f"  Reason: {routing.get('reason', 'N/A')}")
    else:
        # Default routing for single agent
        routing = {
            "complexity": 4,
            "path": "standard",
            "specialist": "separation",
            "reason": "Single agent with full tool access"
        }

    # Choose graph - use multi_agent_graph for proper multi-agent workflow
    use_multi_agent = MULTI_AGENT_AVAILABLE and multi_agent_graph is not None
    active_graph = multi_agent_graph if use_multi_agent else agent_graph

    if not active_graph:
        print("ERROR: No agent graph available!")
        return

    print(f"  Using graph: {'multi_agent_graph' if use_multi_agent else 'agent_graph (single agent)'}")

    # Create thread config
    thread_id = create_thread_id() if create_thread_id else f"strap-{int(time.time())}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "model": "gemini-2.5-flash"
        },
        "recursion_limit": 100
    }

    print(f"\nExecuting query with thread_id: {thread_id}")
    print("Please wait, this may take 30-90 seconds...\n")

    start_time = time.time()

    # Run the query
    try:
        result = await active_graph.ainvoke(
            {
                "messages": [HumanMessage(content=STRAP_QUERY)],
                "iteration_count": 0,
                "max_iterations": MAX_ITERATIONS,
                "user_id": "publication-test",
                "memory_context": "",
                "memory_enabled": False
            },
            config
        )

        elapsed = time.time() - start_time

        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Iterations: {result.get('iteration_count', 'N/A')}")

        # Extract response - check all messages for content
        messages = result.get("messages", [])
        print(f"Total messages: {len(messages)}")

        # Debug: show all message types and content previews
        print("\nMessage breakdown:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            msg_content = getattr(msg, 'content', None)
            if isinstance(msg_content, list):
                preview = str(msg_content)[:100]
            elif msg_content:
                preview = str(msg_content)[:100]
            else:
                preview = "(empty)"
            print(f"  [{i}] {msg_type}: {preview}...")

        # Try to find the best response content
        content = ""

        # Strategy 1: Check final message
        if messages:
            final = messages[-1]
            final_content = getattr(final, 'content', None)

            # Handle list-type content
            if isinstance(final_content, list):
                text_parts = []
                for part in final_content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif isinstance(part, str):
                        text_parts.append(part)
                final_content = '\n'.join(text_parts) if text_parts else str(final_content)

            if final_content and str(final_content).strip():
                content = str(final_content)

        # Strategy 2: If final is empty, look for AIMessage with content
        if not content or not content.strip():
            from langchain_core.messages import AIMessage
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    msg_content = getattr(msg, 'content', None)
                    if isinstance(msg_content, list):
                        text_parts = []
                        for part in msg_content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        msg_content = '\n'.join(text_parts) if text_parts else None
                    if msg_content and str(msg_content).strip():
                        content = str(msg_content)
                        break

        # Strategy 3: Check for ToolMessage with substantial content
        if not content or len(content.strip()) < 50:
            for msg in reversed(messages):
                msg_type = type(msg).__name__
                if msg_type == 'ToolMessage':
                    tool_content = getattr(msg, 'content', None)
                    if tool_content and len(str(tool_content)) > 100:
                        content = str(tool_content)
                        break

        # Strategy 4: Check result state for separation_results, tea_results
        if not content or len(content.strip()) < 50:
            state_response_parts = []

            sep_results = result.get("separation_results", {})
            if sep_results:
                state_response_parts.append("## Separation Results")
                state_response_parts.append(f"- Polymers: {sep_results.get('polymers', [])}")
                state_response_parts.append(f"- Solvents: {sep_results.get('solvents', [])}")
                state_response_parts.append(f"- Best sequence: {sep_results.get('best_sequence', [])}")

            tea_results = result.get("tea_results", {})
            if tea_results:
                state_response_parts.append("\n## TEA Results")
                state_response_parts.append(f"- Best solvent: {tea_results.get('best_solvent')}")
                state_response_parts.append(f"- Cost per kg: ${tea_results.get('cost_per_kg', 0):.2f}")
                state_response_parts.append(f"- Payback years: {tea_results.get('payback_years', 0):.1f}")

            lit_results = result.get("literature_results", {})
            if lit_results:
                state_response_parts.append("\n## Literature Results")
                state_response_parts.append(f"- Papers found: {lit_results.get('papers_found', 0)}")
                state_response_parts.append(f"- Key findings: {lit_results.get('key_findings', [])[:3]}")

            if state_response_parts:
                content = "\n".join(state_response_parts)
                print("\n[Response extracted from state results]")

        if not content:
            content = "(No response content found)"

        print("\n" + "-"*80)
        print("RESPONSE (first 2000 chars):")
        print("-"*80)
        print(content[:2000] + ("..." if len(content) > 2000 else ""))

        # Get execution trace for telemetry
        execution_trace = result.get("execution_trace", {})

        # Also extract handoff_metrics from state if available
        handoff_metrics = result.get("handoff_metrics", [])

        # Save telemetry data
        telemetry_file = OUTPUT_DIR / "telemetry_strap.json"
        telemetry_data = {
            "query": STRAP_QUERY,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "iterations": result.get("iteration_count", 0),
            "routing": routing if routing else {},
            "execution_trace": execution_trace,
            "handoff_metrics": handoff_metrics,
            "message_count": len(messages),
            "thread_id": str(thread_id)
        }

        with open(telemetry_file, 'w') as f:
            json.dump(telemetry_data, f, indent=2, default=str)
        print(f"\nTelemetry saved to: {telemetry_file}")

        # Save full response
        response_file = OUTPUT_DIR / "response_strap.txt"
        with open(response_file, 'w') as f:
            f.write(f"Query: {STRAP_QUERY}\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")
            f.write(f"Iterations: {result.get('iteration_count', 0)}\n\n")
            f.write("="*80 + "\n")
            f.write("RESPONSE:\n")
            f.write("="*80 + "\n\n")
            f.write(content if content else "No response")
        print(f"Response saved to: {response_file}")

        # Copy any generated plots
        import glob
        import shutil
        plots = glob.glob("plots/*.png")
        if plots:
            print(f"\nCopying {len(plots)} plots to output directory...")
            for plot in plots:
                if "tests-1.2" not in plot:  # Don't copy our own output
                    dest = OUTPUT_DIR / os.path.basename(plot)
                    shutil.copy2(plot, dest)
                    print(f"  Copied: {os.path.basename(plot)}")

        # Generate architecture visualizations
        print("\nGenerating architecture visualizations...")
        generate_architecture_visualizations(telemetry_data, OUTPUT_DIR)

        print("\n" + "="*80)
        print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
        print("="*80)

        return result

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    asyncio.run(run_query())
