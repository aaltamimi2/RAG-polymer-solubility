#!/usr/bin/env python3
"""
STRAP Solubility Search - Benchmark Suite
==========================================

Runs a comprehensive set of test prompts against the API and generates
a detailed PDF report with all outputs and visualizations.

Usage:
    python benchmark/run_benchmark.py [--server-url URL] [--output-dir DIR]

Requirements:
    pip install requests reportlab Pillow

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

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        PageBreak, Preformatted, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. PDF report will not be generated.")
    print("Install with: pip install reportlab Pillow")

# ============================================================
# Benchmark Configuration
# ============================================================

BENCHMARK_PROMPTS = [
    # --- Data Exploration ---
    {
        "id": "01_tables",
        "category": "Data Exploration",
        "name": "List Available Tables",
        "prompt": "What tables are available and what columns do they contain?",
        "description": "Verify database schema and table structure"
    },
    {
        "id": "02_polymers",
        "category": "Data Exploration",
        "name": "List Polymers",
        "prompt": "List all unique polymers in the database with their data counts",
        "description": "Check available polymers for analysis"
    },

    # --- Basic Solubility Queries ---
    {
        "id": "03_ldpe_solvents",
        "category": "Basic Queries",
        "name": "LDPE Solvents at 25°C",
        "prompt": "Find the top 10 solvents that dissolve LDPE at 25°C, ranked by solubility percentage",
        "description": "Basic solubility query for single polymer"
    },
    {
        "id": "04_pet_solvents",
        "category": "Basic Queries",
        "name": "PET Solvents at 80°C",
        "prompt": "What solvents dissolve PET at 80°C? Show top 10 with solubility values",
        "description": "Solubility query at elevated temperature"
    },

    # --- Solvent Properties ---
    {
        "id": "05_rank_cost",
        "category": "Solvent Properties",
        "name": "Rank by Energy Cost",
        "prompt": "Rank all solvents by energy cost (lowest first) and show a bar chart of the top 20",
        "description": "Test solvent property ranking and visualization"
    },
    {
        "id": "06_rank_toxicity",
        "category": "Solvent Properties",
        "name": "Rank by Toxicity (LogP)",
        "prompt": "Rank solvents by LogP (toxicity indicator) from lowest to highest. Show top 20 least toxic solvents.",
        "description": "Test toxicity ranking using LogP values"
    },
    {
        "id": "07_solvent_props",
        "category": "Solvent Properties",
        "name": "Specific Solvent Properties",
        "prompt": "Show properties for cyclohexane, toluene, hexane, and xylene including boiling point, LogP, and energy cost",
        "description": "Verify correct property lookup for specific solvents"
    },

    # --- Polymer Separation (2 polymers) ---
    {
        "id": "08_separate_ldpe_pet",
        "category": "Polymer Separation",
        "name": "Separate LDPE from PET",
        "prompt": "Find selective solvents to separate LDPE from PET at 25°C. Rank by selectivity and include cost and toxicity data.",
        "description": "Two-polymer separation with property integration"
    },
    {
        "id": "09_separate_ldpe_evoh",
        "category": "Polymer Separation",
        "name": "Separate LDPE from EVOH",
        "prompt": "Find solvents to separate LDPE from EVOH at 25°C with selectivity analysis",
        "description": "LDPE/EVOH separation for multilayer film"
    },

    # --- Multilayer Film Analysis ---
    {
        "id": "10_multilayer_sequence",
        "category": "Multilayer Film",
        "name": "Multilayer Film Separation Strategy",
        "prompt": "I have a multilayer film with 80% LDPE, 12% PET, and 8% EVOH. Plan a sequential separation strategy at 25°C to isolate each polymer. Create decision trees showing the separation pathways.",
        "description": "Complete multilayer film dissolution sequence"
    },
    {
        "id": "11_multilayer_elevated_temp",
        "category": "Multilayer Film",
        "name": "Multilayer at Elevated Temperature",
        "prompt": "Plan sequential separation of LDPE, PET, and EVOH at 80°C. Compare with room temperature options.",
        "description": "Temperature comparison for multilayer separation"
    },

    # --- Temperature Analysis ---
    {
        "id": "12_temp_curve_ldpe",
        "category": "Temperature Analysis",
        "name": "LDPE Temperature Curves",
        "prompt": "Plot solubility vs temperature curves for LDPE in cyclohexane, toluene, and xylene from 25°C to 120°C",
        "description": "Multi-solvent temperature dependency visualization"
    },
    {
        "id": "13_temp_comparison",
        "category": "Temperature Analysis",
        "name": "Temperature Window Analysis",
        "prompt": "Analyze the temperature window for separating LDPE from PET using toluene. Show where selective dissolution is possible.",
        "description": "Identify optimal temperature ranges"
    },

    # --- Statistical Analysis ---
    {
        "id": "14_statistics",
        "category": "Statistical Analysis",
        "name": "Statistical Summary",
        "prompt": "Provide statistical summary of solubility data for LDPE including mean, median, std dev, and distribution visualization",
        "description": "Basic statistical analysis"
    },
    {
        "id": "15_correlation",
        "category": "Statistical Analysis",
        "name": "Correlation Analysis",
        "prompt": "Analyze correlation between solubility and temperature for HDPE. Include regression analysis.",
        "description": "Statistical correlation and regression"
    },

    # --- Visualizations ---
    {
        "id": "16_heatmap",
        "category": "Visualization",
        "name": "Solubility Heatmap",
        "prompt": "Create a heatmap showing solubility of LDPE, PET, HDPE, and PS across different solvents at 25°C",
        "description": "Multi-polymer heatmap visualization"
    },
    {
        "id": "17_comparison_dashboard",
        "category": "Visualization",
        "name": "Polymer Comparison Dashboard",
        "prompt": "Create a comparison dashboard for LDPE, PET, and EVOH at 25°C showing solubility distributions and top solvents",
        "description": "Comprehensive multi-panel visualization"
    },

    # --- Advanced Queries ---
    {
        "id": "18_optimal_conditions",
        "category": "Advanced",
        "name": "Optimal Separation Conditions",
        "prompt": "Find the optimal temperature and solvent combination for separating LDPE from a mixture with PET and EVOH. Consider both selectivity and practical factors (cost, toxicity, boiling point).",
        "description": "Multi-factor optimization query"
    },
    {
        "id": "19_adaptive_search",
        "category": "Advanced",
        "name": "Adaptive Threshold Search",
        "prompt": "Use adaptive threshold search to find the best selectivity achievable for LDPE separation from PET at temperatures between 20-100°C",
        "description": "Test adaptive analysis functionality"
    },
    {
        "id": "20_comprehensive",
        "category": "Advanced",
        "name": "Comprehensive Analysis",
        "prompt": "Provide a comprehensive analysis for dissolving LDPE at 60°C: list top 5 solvents with their properties (cost, toxicity, boiling point), show a bar chart comparison, and provide recommendations for industrial use.",
        "description": "Full analysis with multiple outputs"
    },
]

# ============================================================
# Benchmark Runner
# ============================================================

class BenchmarkRunner:
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
                print(f"✅ Server is ready")
                print(f"   Tables: {status.get('tables_loaded', 0)}")
                print(f"   Tools: {status.get('tools_available', 0)}")
                return status.get('status') == 'ready'
            return False
        except Exception as e:
            print(f"❌ Server check failed: {e}")
            return False

    def send_prompt(self, prompt: str) -> dict:
        """Send a prompt to the API and return the response."""
        try:
            payload = {
                "message": prompt,
                "session_id": self.session_id
            }

            response = requests.post(
                f"{self.server_url}/api/chat",
                json=payload,
                timeout=300  # 5 minute timeout for complex queries
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

    def download_image(self, image_name: str, benchmark_id: str) -> str:
        """Download an image from the server and save locally."""
        try:
            response = requests.get(f"{self.server_url}/plots/{image_name}", timeout=30)
            if response.status_code == 200:
                # Save with benchmark ID prefix
                local_name = f"{benchmark_id}_{image_name}"
                local_path = self.images_dir / local_name
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return str(local_path)
        except Exception as e:
            print(f"   ⚠️ Failed to download {image_name}: {e}")
        return None

    def run_benchmark(self, prompt_data: dict) -> dict:
        """Run a single benchmark prompt."""
        result = {
            "id": prompt_data["id"],
            "category": prompt_data["category"],
            "name": prompt_data["name"],
            "prompt": prompt_data["prompt"],
            "description": prompt_data["description"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "response": "",
            "images": [],
            "elapsed_time": 0,
            "iterations": 0,
            "error": None
        }

        print(f"\n{'='*60}")
        print(f"[{prompt_data['id']}] {prompt_data['name']}")
        print(f"Category: {prompt_data['category']}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt_data['prompt'][:100]}...")

        api_result = self.send_prompt(prompt_data["prompt"])

        if api_result["success"]:
            result["success"] = True
            result["response"] = api_result["response"]
            result["elapsed_time"] = api_result["elapsed_time"]
            result["iterations"] = api_result["iterations"]

            # Download images
            for img_name in api_result.get("images", []):
                local_path = self.download_image(img_name, prompt_data["id"])
                if local_path:
                    result["images"].append(local_path)

            print(f"✅ Success ({result['elapsed_time']:.1f}s, {result['iterations']} iterations)")
            print(f"   Response length: {len(result['response'])} chars")
            print(f"   Images: {len(result['images'])}")
        else:
            result["error"] = api_result.get("error", "Unknown error")
            print(f"❌ Failed: {result['error'][:100]}")

        return result

    def run_all(self, prompts: list = None):
        """Run all benchmark prompts."""
        if prompts is None:
            prompts = BENCHMARK_PROMPTS

        print("\n" + "="*70)
        print("  STRAP SOLUBILITY SEARCH - BENCHMARK SUITE")
        print("="*70)
        print(f"Server: {self.server_url}")
        print(f"Output: {self.output_dir}")
        print(f"Prompts: {len(prompts)}")
        print(f"Delay between calls: {self.delay}s")
        print("="*70)

        # Check server
        if not self.check_server():
            print("\n❌ Server is not ready. Please start the server first.")
            return False

        start_time = time.time()

        for i, prompt_data in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]", end="")

            result = self.run_benchmark(prompt_data)
            self.results.append(result)

            # Save intermediate results
            self.save_json_results()

            # Delay between calls to avoid rate limiting
            if i < len(prompts):
                print(f"   Waiting {self.delay}s before next call...")
                time.sleep(self.delay)

        total_time = time.time() - start_time

        # Summary
        print("\n" + "="*70)
        print("  BENCHMARK COMPLETE")
        print("="*70)

        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        total_images = sum(len(r["images"]) for r in self.results)

        print(f"Total time: {total_time:.1f}s")
        print(f"Successful: {successful}/{len(self.results)}")
        print(f"Failed: {failed}")
        print(f"Images generated: {total_images}")

        return True

    def save_json_results(self):
        """Save results to JSON file."""
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "server_url": self.server_url,
                "results": self.results
            }, f, indent=2)

    def generate_pdf_report(self):
        """Generate comprehensive PDF report."""
        if not REPORTLAB_AVAILABLE:
            print("⚠️ Skipping PDF generation (reportlab not installed)")
            return None

        pdf_path = self.output_dir / "benchmark_report.pdf"
        print(f"\n📄 Generating PDF report: {pdf_path}")

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()

        # Custom styles
        styles.add(ParagraphStyle(
            name='Title2',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        ))
        styles.add(ParagraphStyle(
            name='Category',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a365d'),
            spaceBefore=20,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            name='BenchmarkTitle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2d3748'),
            spaceBefore=15,
            spaceAfter=5
        ))
        styles.add(ParagraphStyle(
            name='Prompt',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4a5568'),
            leftIndent=20,
            spaceBefore=5,
            spaceAfter=10,
            fontName='Helvetica-Oblique'
        ))
        styles.add(ParagraphStyle(
            name='Response',
            parent=styles['Normal'],
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=5,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            name='Code',
            parent=styles['Normal'],
            fontSize=8,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            backColor=colors.HexColor('#f7fafc'),
            spaceBefore=5,
            spaceAfter=10
        ))

        story = []

        # Title page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("STRAP Solubility Search", styles['Title2']))
        story.append(Paragraph("Benchmark Report", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.25*inch))

        # Summary statistics
        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        total_images = sum(len(r["images"]) for r in self.results)
        avg_time = sum(r["elapsed_time"] for r in self.results if r["success"]) / max(successful, 1)

        summary_data = [
            ["Metric", "Value"],
            ["Total Tests", str(len(self.results))],
            ["Successful", f"{successful} ({100*successful/len(self.results):.0f}%)"],
            ["Failed", str(failed)],
            ["Total Images", str(total_images)],
            ["Avg Response Time", f"{avg_time:.1f}s"],
        ]

        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        story.append(summary_table)
        story.append(PageBreak())

        # Results by category
        current_category = None

        for result in self.results:
            # Category header
            if result["category"] != current_category:
                current_category = result["category"]
                story.append(Paragraph(current_category, styles['Category']))

            # Benchmark section
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            status_color = '#27ae60' if result["success"] else '#e74c3c'

            story.append(Paragraph(
                f"<font color='{status_color}'>{status}</font> [{result['id']}] {result['name']}",
                styles['BenchmarkTitle']
            ))

            story.append(Paragraph(f"<i>{result['description']}</i>", styles['Normal']))
            story.append(Paragraph(f"<b>Prompt:</b> {result['prompt']}", styles['Prompt']))

            if result["success"]:
                # Metadata
                story.append(Paragraph(
                    f"<font size='8' color='#718096'>Time: {result['elapsed_time']:.1f}s | "
                    f"Iterations: {result['iterations']} | "
                    f"Images: {len(result['images'])}</font>",
                    styles['Normal']
                ))

                # Response (truncated if too long)
                response_text = result["response"]
                if len(response_text) > 2000:
                    response_text = response_text[:2000] + "\n\n[... truncated ...]"

                # Clean up markdown for PDF
                response_text = response_text.replace('**', '')
                response_text = response_text.replace('`', '')

                story.append(Paragraph("<b>Response:</b>", styles['Normal']))

                # Split into paragraphs
                for para in response_text.split('\n\n')[:10]:  # Limit paragraphs
                    if para.strip():
                        try:
                            story.append(Paragraph(para.replace('\n', '<br/>'), styles['Response']))
                        except:
                            # If paragraph has issues, use preformatted
                            story.append(Preformatted(para[:500], styles['Code']))

                # Images
                for img_path in result["images"]:
                    if os.path.exists(img_path):
                        try:
                            img = Image(img_path, width=5.5*inch, height=3.5*inch)
                            img.hAlign = 'CENTER'
                            story.append(Spacer(1, 0.2*inch))
                            story.append(img)
                            story.append(Spacer(1, 0.1*inch))
                        except Exception as e:
                            story.append(Paragraph(f"[Image: {os.path.basename(img_path)}]", styles['Normal']))
            else:
                story.append(Paragraph(
                    f"<font color='#e74c3c'><b>Error:</b> {result.get('error', 'Unknown error')}</font>",
                    styles['Response']
                ))

            story.append(Spacer(1, 0.3*inch))

        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF report saved: {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return None


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run STRAP Solubility Search benchmark suite"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: benchmark/results_TIMESTAMP)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API calls in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Run only prompts matching this category (e.g., 'Multilayer Film')"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"benchmark/results_{timestamp}"

    # Filter prompts if subset specified
    prompts = BENCHMARK_PROMPTS
    if args.subset:
        prompts = [p for p in prompts if args.subset.lower() in p["category"].lower()]
        if not prompts:
            print(f"No prompts match category '{args.subset}'")
            print("Available categories:", set(p["category"] for p in BENCHMARK_PROMPTS))
            return 1

    # Run benchmark
    runner = BenchmarkRunner(
        server_url=args.server_url,
        output_dir=output_dir,
        delay_between_calls=args.delay
    )

    success = runner.run_all(prompts)

    if success:
        # Save final results
        runner.save_json_results()

        # Generate PDF report
        runner.generate_pdf_report()

        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"   - benchmark_results.json")
        print(f"   - benchmark_report.pdf")
        print(f"   - images/")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
