#!/usr/bin/env python3
"""
Multilayer Film Separation - Comprehensive Case Study
======================================================

A detailed case study analyzing the dissolution and separation of a
three-layer film composed of LDPE (80%), EVOH (8%), and PET (12%).

This script runs a systematic analysis and generates both visualizations
and a markdown report suitable for documentation.

Usage:
    python benchmark/run_case_study.py [--server-url URL] [--output-dir DIR]

Requirements:
    pip install requests

The server must be running before executing this script.
"""

import os
import sys
import json
import time
import shutil
import argparse
import requests
from datetime import datetime
from pathlib import Path

# ============================================================
# Case Study Configuration
# ============================================================

FILM_COMPOSITION = {
    "LDPE": 80,
    "EVOH": 8,
    "PET": 12
}

TEMPERATURE = 80  # °C

CASE_STUDY_PROMPTS = [
    {
        "id": "01_polymers_overview",
        "section": "1. Dataset Overview",
        "title": "Available Polymers",
        "prompt": "List all available polymers in the database with their data counts",
        "description": "Overview of all polymers available for analysis"
    },
    {
        "id": "02_selectivity_heatmap",
        "section": "2. Solvent Selection at 80°C",
        "title": "Selectivity Heatmap",
        "prompt": f"Create a selectivity heatmap for LDPE, EVOH, and PET at {TEMPERATURE}°C. Include the top 10 common solvents that show good solubility for at least one polymer.",
        "description": "Visual overview of solvent selectivity patterns"
    },
    {
        "id": "03_film_separation_analysis",
        "section": "3. Multilayer Film Analysis",
        "title": "Sequential Separation Strategy",
        "prompt": f"I have a multilayer film with {FILM_COMPOSITION['LDPE']}% LDPE, {FILM_COMPOSITION['EVOH']}% EVOH, and {FILM_COMPOSITION['PET']}% PET. Plan a sequential separation strategy at {TEMPERATURE}°C. Show the best sequence and create a decision tree visualization.",
        "description": "Optimal dissolution sequence for the three-layer film"
    },
    {
        "id": "04_alternative_sequence",
        "section": "3. Multilayer Film Analysis",
        "title": "Alternative Separation Strategy",
        "prompt": "Show me the second-best separation sequence for LDPE, EVOH, and PET as an alternative approach",
        "description": "Backup separation strategy"
    },
    {
        "id": "05_ldpe_solubility",
        "section": "4. Detailed Solubility Analysis",
        "title": "LDPE Solubility at 80°C",
        "prompt": f"What are the top 10 solvents for LDPE at {TEMPERATURE}°C? Show solubility percentages and create a bar chart.",
        "description": "LDPE dissolution options"
    },
    {
        "id": "06_evoh_solubility",
        "section": "4. Detailed Solubility Analysis",
        "title": "EVOH Solubility at 80°C",
        "prompt": f"What are the top 10 solvents for EVOH at {TEMPERATURE}°C? Show solubility percentages and create a bar chart.",
        "description": "EVOH dissolution options"
    },
    {
        "id": "07_pet_solubility",
        "section": "4. Detailed Solubility Analysis",
        "title": "PET Solubility at 80°C",
        "prompt": f"What are the top 10 solvents for PET at {TEMPERATURE}°C? Show solubility percentages and create a bar chart.",
        "description": "PET dissolution options"
    },
    {
        "id": "08_ml_ldpe_solvent1",
        "section": "5. ML Validation - LDPE",
        "title": "ML Prediction: LDPE in Recommended Solvent",
        "prompt": "Predict solubility of LDPE in cyclohexane using machine learning. Include all visualizations.",
        "description": "ML validation for primary LDPE solvent (will be updated based on recommendation)"
    },
    {
        "id": "09_ml_evoh_solvent1",
        "section": "6. ML Validation - EVOH",
        "title": "ML Prediction: EVOH in Recommended Solvent",
        "prompt": "Predict solubility of EVOH in DMF using machine learning. Include all visualizations.",
        "description": "ML validation for primary EVOH solvent (will be updated)"
    },
    {
        "id": "10_ml_pet_solvent1",
        "section": "7. ML Validation - PET",
        "title": "ML Prediction: PET in Recommended Solvent",
        "prompt": "Predict solubility of PET in chloroform using machine learning. Include all visualizations.",
        "description": "ML validation for primary PET solvent (will be updated)"
    },
    {
        "id": "11_solvent_properties",
        "section": "8. Solvent Properties Analysis",
        "title": "Physical and Chemical Properties",
        "prompt": "Show properties (boiling point, LogP, energy cost, heat capacity) for cyclohexane, DMF, chloroform, toluene, xylene, and benzene",
        "description": "Key properties of recommended solvents"
    },
    {
        "id": "12_boiling_points",
        "section": "8. Solvent Properties Analysis",
        "title": "Boiling Point Comparison",
        "prompt": "Create a bar chart comparing boiling points for cyclohexane, DMF, chloroform, toluene, xylene, and benzene",
        "description": "Visualization of boiling point differences"
    },
    {
        "id": "13_energy_cost",
        "section": "8. Solvent Properties Analysis",
        "title": "Energy Cost Comparison",
        "prompt": "Create a bar chart comparing energy costs (J/g) for cyclohexane, DMF, chloroform, toluene, xylene, and benzene",
        "description": "Economic comparison of solvents"
    },
    {
        "id": "14_safety_gscores",
        "section": "9. Safety Analysis",
        "title": "GSK G-Scores",
        "prompt": "Get GSK safety G-scores for cyclohexane, DMF, chloroform, toluene, xylene, and benzene. Visualize the scores.",
        "description": "Safety assessment using GSK methodology"
    },
    {
        "id": "15_safety_logp",
        "section": "9. Safety Analysis",
        "title": "LogP Toxicity Indicator",
        "prompt": "Show LogP values for cyclohexane, DMF, chloroform, toluene, xylene, and benzene. Create a visualization showing relative toxicity.",
        "description": "Alternative toxicity assessment using LogP"
    },
    {
        "id": "16_integrated_analysis",
        "section": "10. Integrated Decision Making",
        "title": "Multi-Factor Analysis",
        "prompt": f"Perform integrated analysis for separating LDPE and EVOH at {TEMPERATURE}°C considering selectivity, cost, toxicity (LogP), and boiling point. Rank the top 5 solvents.",
        "description": "Holistic evaluation combining all factors"
    },
    {
        "id": "17_temperature_curves",
        "section": "11. Temperature Sensitivity",
        "title": "LDPE Temperature Dependence",
        "prompt": "Create an interactive temperature plot for LDPE in cyclohexane, toluene, and xylene from 25°C to 120°C",
        "description": "Understanding solubility variation with temperature"
    },
]

# ============================================================
# Case Study Runner
# ============================================================

class CaseStudyRunner:
    def __init__(self, server_url: str, output_dir: str, delay_between_calls: float = 3.0):
        self.server_url = server_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.delay = delay_between_calls
        self.session_id = None
        self.results = []
        self.images_dir = self.output_dir / "images"

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def check_server(self) -> bool:
        """Check if server is running and ready."""
        try:
            response = requests.get(f"{self.server_url}/api/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"Server is ready")
                print(f"   Tables: {status.get('tables_loaded', 0)}")
                print(f"   Tools: {status.get('tools_available', 0)}")
                return status.get('status') == 'ready'
            return False
        except Exception as e:
            print(f"Server check failed: {e}")
            return False

    def send_prompt(self, prompt: str) -> dict:
        """Send a prompt to the API and return the response."""
        try:
            # Add explicit request for output to force tool results to be included
            prompt_with_output_request = f"{prompt}\n\nIMPORTANT: Return the complete output from the tools you use."

            payload = {
                "message": prompt_with_output_request,
                "session_id": self.session_id,
                "model": "gemini-2.5-flash"  # Use flash for case study
            }

            response = requests.post(
                f"{self.server_url}/api/chat",
                json=payload,
                timeout=300  # 5 minute timeout
            )

            if response.status_code == 200:
                data = response.json()
                if not self.session_id:
                    self.session_id = data.get('session_id')
                return {
                    "success": True,
                    "response": data.get('response', ''),
                    "images": data.get('images', []),
                    "elapsed_time": data.get('elapsed_time', 0),
                    "iterations": data.get('iterations', 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:500]}"
                }
        except requests.Timeout:
            return {"success": False, "error": "Request timed out (5 minutes)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_image(self, image_name: str, prompt_id: str) -> str:
        """Download an image from the server and save locally."""
        try:
            response = requests.get(f"{self.server_url}/plots/{image_name}", timeout=30)
            if response.status_code == 200:
                # Save with prompt ID prefix
                local_name = f"{prompt_id}_{image_name}"
                local_path = self.images_dir / local_name
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return str(local_path)
        except Exception as e:
            print(f"   Warning: Failed to download {image_name}: {e}")
        return None

    def run_analysis(self, prompt_data: dict) -> dict:
        """Run a single analysis prompt."""
        result = {
            "id": prompt_data["id"],
            "section": prompt_data["section"],
            "title": prompt_data["title"],
            "prompt": prompt_data["prompt"],
            "description": prompt_data["description"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "response": "",
            "images": [],
            "local_images": [],
            "elapsed_time": 0,
            "iterations": 0,
            "error": None
        }

        print(f"\n{'='*70}")
        print(f"[{prompt_data['id']}] {prompt_data['title']}")
        print(f"Section: {prompt_data['section']}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt_data['prompt']}")

        api_result = self.send_prompt(prompt_data["prompt"])

        if api_result["success"]:
            result["success"] = True
            result["response"] = api_result["response"]
            result["elapsed_time"] = api_result["elapsed_time"]
            result["iterations"] = api_result["iterations"]
            result["images"] = api_result.get("images", [])

            # Download images
            for img_name in api_result.get("images", []):
                local_path = self.download_image(img_name, prompt_data["id"])
                if local_path:
                    result["local_images"].append(local_path)

            print(f"Success ({result['elapsed_time']:.1f}s, {result['iterations']} iterations)")
            print(f"   Response length: {len(result['response'])} chars")
            print(f"   Images: {len(result['local_images'])}")
        else:
            result["error"] = api_result.get("error", "Unknown error")
            print(f"Failed: {result['error'][:100]}")

        return result

    def run_all(self):
        """Run all case study prompts."""
        print("\n" + "="*70)
        print("  MULTILAYER FILM SEPARATION - CASE STUDY")
        print("="*70)
        print(f"Film Composition: LDPE {FILM_COMPOSITION['LDPE']}%, EVOH {FILM_COMPOSITION['EVOH']}%, PET {FILM_COMPOSITION['PET']}%")
        print(f"Temperature: {TEMPERATURE}°C")
        print(f"Server: {self.server_url}")
        print(f"Output: {self.output_dir}")
        print(f"Analysis steps: {len(CASE_STUDY_PROMPTS)}")
        print("="*70)

        # Check server
        if not self.check_server():
            print("\nServer is not ready. Please start the server first.")
            return False

        start_time = time.time()

        for i, prompt_data in enumerate(CASE_STUDY_PROMPTS, 1):
            print(f"\n[{i}/{len(CASE_STUDY_PROMPTS)}]", end="")

            result = self.run_analysis(prompt_data)
            self.results.append(result)

            # Save intermediate results
            self.save_json_results()

            # Delay between calls
            if i < len(CASE_STUDY_PROMPTS):
                print(f"   Waiting {self.delay}s before next call...")
                time.sleep(self.delay)

        total_time = time.time() - start_time

        # Summary
        print("\n" + "="*70)
        print("  CASE STUDY COMPLETE")
        print("="*70)

        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        total_images = sum(len(r["local_images"]) for r in self.results)

        print(f"Total time: {total_time:.1f}s")
        print(f"Successful: {successful}/{len(self.results)}")
        print(f"Failed: {failed}")
        print(f"Images generated: {total_images}")

        return True

    def save_json_results(self):
        """Save results to JSON file."""
        json_path = self.output_dir / "case_study_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "server_url": self.server_url,
                "film_composition": FILM_COMPOSITION,
                "temperature": TEMPERATURE,
                "results": self.results
            }, f, indent=2)

    def generate_markdown_report(self):
        """Generate comprehensive markdown report."""
        md_path = self.output_dir / "CASE_STUDY.md"
        print(f"\nGenerating markdown report: {md_path}")

        with open(md_path, 'w') as f:
            # Header
            f.write("# Multilayer Film Separation - Comprehensive Case Study\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"This case study analyzes the dissolution and separation of a three-layer polymer film ")
            f.write(f"composed of **LDPE ({FILM_COMPOSITION['LDPE']}%)**, **EVOH ({FILM_COMPOSITION['EVOH']}%)**, ")
            f.write(f"and **PET ({FILM_COMPOSITION['PET']}%)** at **{TEMPERATURE}°C**.\n\n")
            f.write("The analysis integrates:\n")
            f.write("- Database exploration and polymer overview\n")
            f.write("- Solvent selectivity analysis and heatmap visualization\n")
            f.write("- Sequential separation strategy optimization\n")
            f.write("- Detailed solubility data for each polymer\n")
            f.write("- Machine learning validation using Hansen Solubility Parameters\n")
            f.write("- Solvent property analysis (boiling point, energy cost)\n")
            f.write("- Safety assessment (G-scores and LogP)\n")
            f.write("- Multi-factor integrated decision making\n\n")

            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Table of Contents
            f.write("## Table of Contents\n\n")
            current_section = None
            for i, result in enumerate(self.results, 1):
                if result["section"] != current_section:
                    current_section = result["section"]
                    f.write(f"{i}. [{current_section}](#{self._section_to_anchor(current_section)})\n")
            f.write("\n---\n\n")

            # Results by section
            current_section = None
            section_number = 0

            for result in self.results:
                # Section header
                if result["section"] != current_section:
                    current_section = result["section"]
                    section_number += 1
                    f.write(f"## {current_section}\n\n")

                # Subsection
                f.write(f"### {result['title']}\n\n")
                f.write(f"**Description:** {result['description']}\n\n")

                if result["success"]:
                    # Response
                    f.write("**Results:**\n\n")
                    f.write(f"{result['response']}\n\n")

                    # Images
                    if result["local_images"]:
                        f.write("**Visualizations:**\n\n")
                        for img_path in result["local_images"]:
                            img_name = os.path.basename(img_path)
                            # Relative path for markdown
                            rel_path = f"images/{img_name}"
                            f.write(f"![{result['title']}]({rel_path})\n\n")

                    # Metadata
                    f.write(f"*Analysis completed in {result['elapsed_time']:.1f}s*\n\n")
                else:
                    f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")

                f.write("---\n\n")

            # Conclusions
            f.write("## Conclusions and Recommendations\n\n")
            f.write("Based on the comprehensive analysis above:\n\n")
            f.write("### Optimal Separation Strategy\n\n")
            f.write("The recommended approach for separating this three-layer film is:\n\n")
            f.write("1. **Step 1:** [To be filled based on sequential separation results]\n")
            f.write("2. **Step 2:** [To be filled based on sequential separation results]\n")
            f.write("3. **Step 3:** [To be filled based on sequential separation results]\n\n")
            f.write("### Key Considerations\n\n")
            f.write("- **Selectivity:** Choose solvents with high selectivity to ensure clean separation\n")
            f.write("- **Economics:** Consider energy cost and boiling point for process efficiency\n")
            f.write("- **Safety:** Prioritize solvents with low G-scores and favorable LogP values\n")
            f.write("- **Temperature:** Monitor temperature carefully to maintain selectivity\n\n")
            f.write("### ML Validation\n\n")
            f.write("Machine learning predictions using Hansen Solubility Parameters provide:\n")
            f.write("- Independent validation of experimental solubility data\n")
            f.write("- Confidence metrics for each polymer-solvent pair\n")
            f.write("- Predictive capability for untested combinations\n\n")
            f.write("---\n\n")
            f.write("*Report generated by DISSOLVE Agent - Data-Integrated Solubility Solver via LLM Evaluation*\n")

        print(f"Markdown report saved: {md_path}")
        return str(md_path)

    def _section_to_anchor(self, section: str) -> str:
        """Convert section name to markdown anchor."""
        return section.lower().replace(' ', '-').replace('.', '')

# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run multilayer film separation case study"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: benchmark/case_study_TIMESTAMP)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API calls in seconds (default: 3.0)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"benchmark/case_study_{timestamp}"

    # Run case study
    runner = CaseStudyRunner(
        server_url=args.server_url,
        output_dir=output_dir,
        delay_between_calls=args.delay
    )

    success = runner.run_all()

    if success:
        # Save final results
        runner.save_json_results()

        # Generate markdown report
        md_path = runner.generate_markdown_report()

        print(f"\nResults saved to: {output_dir}/")
        print(f"   - case_study_results.json")
        print(f"   - CASE_STUDY.md")
        print(f"   - images/")

        print(f"\nNext steps:")
        print(f"1. Review the markdown report: {md_path}")
        print(f"2. Add images to git: git add {output_dir}/images/")
        print(f"3. Copy CASE_STUDY.md to documentation/: cp {output_dir}/CASE_STUDY.md documentation/")
        print(f"4. Commit changes: git commit -m 'Add multilayer film case study'")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
