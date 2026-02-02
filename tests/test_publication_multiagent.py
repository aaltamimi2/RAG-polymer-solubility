"""
Publication-quality multi-agent test with detailed telemetry visualization.

Goal: Demonstrate multi-agent architecture for academic publication.
Output: plots/test-1.1/
"""
import os
import sys
import uuid
import json
import time

# Add project to path FIRST
sys.path.insert(0, '/home/aaltamimi2/polymer-solubility-app')

# Load environment BEFORE any imports that need API keys
from dotenv import load_dotenv
load_dotenv('/home/aaltamimi2/polymer-solubility-app/.env')

if not os.getenv('GOOGLE_API_KEY'):
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

print("✓ GOOGLE_API_KEY loaded")

import asyncio
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for noisy in ['httpx', 'httpcore', 'urllib3', 'google', 'langsmith', 'google_genai']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = "/home/aaltamimi2/polymer-solubility-app/plots/test-1.1"

# Complex 10-polymer query for publication
PUBLICATION_QUERY = """
I need to design a selective dissolution process for recycling multilayer packaging waste containing 10 polymers: LDPE, HDPE, PP, PS, PET, PVC, PVDC, EVOH, PA6, and PMMA.

Requirements:
1. **Solvent Selection**: Prioritize greener solvents (low toxicity, bio-based preferred). Avoid chlorinated solvents if possible.
2. **Boiling Point Constraint**: Solvents must have boiling points between 60-150°C for energy-efficient recovery.
3. **Separation Sequence**: Design an optimal dissolution sequence that maximizes polymer purity (>95%) while minimizing solvent variety.
4. **Economic Analysis**: Provide TEA for a 500 kg/hr plant including:
   - Capital costs (equipment sizing)
   - Operating costs (solvent, energy, labor)
   - Solvent recovery economics (distillation costs)
5. **Environmental Assessment**: Compare the carbon footprint of selective dissolution vs mechanical recycling for this waste stream.
6. **Critical Question**: Which 3 polymers are most economically viable to recover, and which should be sent to energy recovery?

Please provide specific Hansen Solubility Parameter data to justify solvent choices.
"""


async def run_publication_test():
    """Run multi-agent test with full telemetry for publication."""
    print("=" * 80)
    print("PUBLICATION MULTI-AGENT TEST")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    from agent_sql_final_1212_patched import multi_agent_graph, MULTI_AGENT_AVAILABLE
    from langchain_core.messages import HumanMessage
    from workflow_engine import (
        WorkflowTrace, StageTrace, AgentTrace,
        trace_to_timeline_svg, format_trace_report,
        trace_to_architecture_svg,
    )

    if not MULTI_AGENT_AVAILABLE:
        print("ERROR: Multi-agent system not available")
        return None

    print("✓ Multi-agent graph imported")
    print(f"\nQuery ({len(PUBLICATION_QUERY)} chars):")
    print("-" * 40)
    print(PUBLICATION_QUERY[:500] + "..." if len(PUBLICATION_QUERY) > 500 else PUBLICATION_QUERY)
    print("-" * 40)

    # Initial state with all 10 polymers
    initial_state = {
        "messages": [HumanMessage(content=PUBLICATION_QUERY)],
        "shared_context": {
            "polymers": ["LDPE", "HDPE", "PP", "PS", "PET", "PVC", "PVDC", "EVOH", "PA6", "PMMA"],
            "constraints": [
                "greener solvents",
                "low toxicity",
                "boiling point 60-150°C",
                "purity >95%",
            ],
            "throughput_kg_hr": 500,
        },
        "collaboration_specialists": ["separation", "tea_lca", "literature"],
        "path": "integrated",
    }

    config = {"configurable": {"thread_id": f"pub-test-{uuid.uuid4().hex[:8]}"}}

    print(f"\nStarting workflow execution...")
    print(f"Polymers: {initial_state['shared_context']['polymers']}")
    print(f"Specialists: {initial_state['collaboration_specialists']}")

    start_time = time.time()

    try:
        result = await multi_agent_graph.ainvoke(initial_state, config)

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)
        print(f"Total execution time: {total_time:.1f}s")

        # Extract telemetry
        orchestration = result.get("orchestration", {})
        workflow_trace_data = result.get("workflow_trace", {})
        workflow_trace_detailed = result.get("workflow_trace_detailed", {})

        print(f"\n--- Orchestration Metadata ---")
        print(f"Workflow: {orchestration.get('workflow_name', 'N/A')}")
        print(f"Used LLM Planner: {orchestration.get('used_planner', 'N/A')}")
        print(f"Planning Time: {orchestration.get('planning_time_ms', 0):.0f}ms")
        print(f"Total Time: {orchestration.get('total_time_seconds', 0):.1f}s")

        if orchestration.get('planning_reasoning'):
            print(f"\nPlanning Reasoning:")
            print(f"  {orchestration['planning_reasoning'][:300]}...")

        print(f"\n--- Telemetry Summary ---")
        print(f"Stages: {workflow_trace_data.get('stages_count', 0)}")
        print(f"Agents Run: {workflow_trace_data.get('agents_run', 0)}")
        print(f"Tool Calls: {workflow_trace_data.get('tool_calls', 0)}")
        print(f"Success: {workflow_trace_data.get('success', 'N/A')}")

        # Save raw telemetry data
        telemetry_file = os.path.join(OUTPUT_DIR, "telemetry_raw.json")
        with open(telemetry_file, 'w') as f:
            json.dump({
                "query": PUBLICATION_QUERY,
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "orchestration": orchestration,
                "workflow_trace": workflow_trace_data,
                "workflow_trace_detailed": workflow_trace_detailed,
            }, f, indent=2, default=str)
        print(f"\n✓ Raw telemetry saved: {telemetry_file}")

        # Generate visualizations
        if workflow_trace_detailed:
            print("\n" + "=" * 80)
            print("GENERATING PUBLICATION VISUALIZATIONS")
            print("=" * 80)

            # Reconstruct WorkflowTrace from detailed data
            stages_data = workflow_trace_detailed.get("stages", [])
            stage_traces = []

            from workflow_engine import ToolCallTrace

            for stage_data in stages_data:
                agent_traces = []
                for at_data in stage_data.get("agent_traces", []):
                    # Reconstruct tool calls from count (creates placeholders for visualization)
                    tool_call_count = at_data.get("tool_calls", 0)
                    tool_calls_list = [
                        ToolCallTrace(
                            tool_name=f"tool_{i+1}",
                            arguments={},
                            result_summary="(from telemetry)",
                            duration_ms=0,
                            success=True,
                        )
                        for i in range(tool_call_count)
                    ]

                    agent_traces.append(AgentTrace(
                        agent_name=at_data.get("agent", "unknown"),
                        iterations=at_data.get("iterations", 1),
                        tool_calls=tool_calls_list,
                        duration_seconds=at_data.get("duration_seconds", 0),
                        success=at_data.get("success", True),
                        output_keys=at_data.get("output_keys", []),
                    ))

                stage_traces.append(StageTrace(
                    stage_index=stage_data.get("index", 0),
                    stage_type=stage_data.get("type", "sequential"),
                    agents=stage_data.get("agents", []),
                    agent_traces=agent_traces,
                    duration_seconds=stage_data.get("duration_seconds", 0),
                    filter_applied=stage_data.get("filter"),
                    polymers_before_filter=stage_data.get("polymers_before"),
                    polymers_after_filter=stage_data.get("polymers_after"),
                ))

            trace = WorkflowTrace(
                workflow_name=orchestration.get("workflow_name", "integrated_workflow"),
                query=PUBLICATION_QUERY,
                context={
                    "polymers": ["LDPE", "HDPE", "PP", "PS", "PET", "PVC", "PVDC", "EVOH", "PA6", "PMMA"],
                    "specialists": ["separation", "tea_lca", "literature"],
                    "throughput": "500 kg/hr",
                },
                stages=stage_traces,
                total_duration_seconds=orchestration.get("total_time_seconds", total_time),
                used_planner=orchestration.get("used_planner", False),
                planning_time_ms=orchestration.get("planning_time_ms", 0),
                planning_reasoning=orchestration.get("planning_reasoning"),
                success=workflow_trace_data.get("success", True),
            )

            # 1. Timeline visualization
            timeline_path = os.path.join(OUTPUT_DIR, "execution_timeline.svg")
            try:
                trace_to_timeline_svg(
                    trace,
                    output_path=timeline_path,
                    title="Multi-Agent Workflow: 10-Polymer Dissolution Planning"
                )
                print(f"✓ Timeline SVG: {timeline_path}")
            except Exception as e:
                print(f"✗ Timeline generation failed: {e}")

            # 2. Text report
            report_path = os.path.join(OUTPUT_DIR, "trace_report.txt")
            try:
                report = format_trace_report(trace)
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"✓ Trace report: {report_path}")
                print("\n--- Trace Report Preview ---")
                print(report[:1000] + "..." if len(report) > 1000 else report)
            except Exception as e:
                print(f"✗ Report generation failed: {e}")

        # Save response
        print("\n" + "=" * 80)
        print("RESPONSE")
        print("=" * 80)

        messages = result.get("messages", [])
        response_text = ""
        if messages:
            last_msg = messages[-1]
            response_text = getattr(last_msg, 'content', str(last_msg))

        response_file = os.path.join(OUTPUT_DIR, "response.md")
        with open(response_file, 'w') as f:
            f.write(f"# Multi-Agent Response\n\n")
            f.write(f"**Query:** {PUBLICATION_QUERY[:200]}...\n\n")
            f.write(f"**Execution Time:** {total_time:.1f}s\n\n")
            f.write(f"**Workflow:** {orchestration.get('workflow_name', 'N/A')}\n\n")
            f.write("---\n\n")
            f.write(response_text)
        print(f"✓ Response saved: {response_file}")

        # Print response preview
        if response_text:
            print(f"\nResponse preview ({len(response_text)} chars):")
            print("-" * 40)
            print(response_text[:2000] + "..." if len(response_text) > 2000 else response_text)

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print(f"Output files in: {OUTPUT_DIR}")
        print(f"  - telemetry_raw.json")
        print(f"  - execution_timeline.svg")
        print(f"  - trace_report.txt")
        print(f"  - response.md")

        return result

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

        # Save error info
        error_file = os.path.join(OUTPUT_DIR, "error.txt")
        with open(error_file, 'w') as f:
            f.write(f"Test failed at {datetime.now().isoformat()}\n\n")
            f.write(f"Error: {type(e).__name__}: {e}\n\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"Error details saved to: {error_file}")
        raise


if __name__ == "__main__":
    asyncio.run(run_publication_test())
