"""ML-based solubility prediction using Hansen Solubility Parameters."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strap.database import get_connection
from strap.tools._helpers import safe_tool_wrapper, truncate_output, save_plot, get_plots_dir

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

try:
    from strap.vendor.solubility_predictor import get_predictor
except ImportError:
    get_predictor = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local fuzzy-match helper (adapted from monolith to use get_connection())
# ---------------------------------------------------------------------------

def _fuzzy_match_solvent_name(
    solvent_name: str,
    dataset: str = "all",
    threshold: int = 80,
) -> Optional[Dict[str, Any]]:
    """Find the best matching solvent name across datasets using fuzzy matching."""
    try:
        from thefuzz import fuzz, process
    except ImportError:
        return None

    try:
        conn = get_connection()
        best_match: Optional[str] = None
        best_score: int = 0
        best_dataset: Optional[str] = None
        solvent_name_clean = solvent_name.strip().lower()

        dataset_configs: list[tuple[str, str, str, str]] = [
            ("gsk", "SELECT DISTINCT solvent_common_name FROM gsk_dataset", "solvent_common_name", "gsk_dataset"),
            ("solvent_data", "SELECT DISTINCT cosmobase_name FROM solvent_data", "cosmobase_name", "solvent_data"),
            ("common_solvents", "SELECT DISTINCT solvent FROM common_solvents_database", "solvent", "common_solvents_database"),
        ]

        for ds_key, query, column, ds_name in dataset_configs:
            if dataset not in (ds_key, "all"):
                continue
            try:
                df = conn.execute(query).fetchdf()
                if len(df) == 0:
                    continue
                names = df[column].tolist()
                names_lower = [n.lower() for n in names]
                match = process.extractOne(solvent_name_clean, names_lower, scorer=fuzz.ratio)
                if match and match[1] > best_score:
                    idx = names_lower.index(match[0])
                    best_match = names[idx]
                    best_score = match[1]
                    best_dataset = ds_name
            except Exception as exc:
                logger.debug(f"{ds_name} search failed: {exc}")

        if best_score >= threshold and best_match is not None:
            return {
                "matched_name": best_match,
                "score": best_score,
                "dataset": best_dataset,
                "original_query": solvent_name,
            }
        return None

    except Exception as exc:
        logger.error(f"Fuzzy matching error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

@safe_tool_wrapper
async def predict_solubility_ml(
    polymer_name: str,
    solvent_name: str,
    temperature: float = 25.0,
    generate_visualizations: bool = True,
) -> str:
    """Predict polymer-solvent solubility using an ML model trained on Hansen Solubility Parameters.

    Args:
        polymer_name: Name of polymer (e.g., "HDPE", "PET", "PVDF")
        solvent_name: Name of solvent (e.g., "Toluene", "Water", "Acetone")
        temperature: Temperature in Celsius (default: 25.0)
        generate_visualizations: Whether to create visualization files (default: True)

    WHEN TO USE:
    - "Will PET dissolve in toluene?"
    - "Predict solubility of HDPE in acetone"
    - "Is PVDF soluble in DMF?"
    """
    try:
        if get_predictor is None:
            return "ML predictor unavailable — strap.vendor.solubility_predictor could not be imported."

        PLOTS_DIR = get_plots_dir()

        # Get predictor
        predictor = get_predictor()

        # First, try to get polymer HSP from CSV (since we don't know if DB tables exist)
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent.parent.parent / 'HSP-ML-integration' / 'RED_values_complete_CORRECTED.csv'
        if not csv_path.exists():
            return f"HSP data file not found at {csv_path}. Ensure the HSP-ML-integration directory is in the project root."

        try:
            hsp_data = pd.read_csv(csv_path)

            # Find polymer
            polymer_data = hsp_data[hsp_data['Polymer'].str.lower() == polymer_name.lower()]

            if len(polymer_data) == 0:
                # Try fuzzy matching with partial string match
                all_polymers = hsp_data['Polymer'].unique()
                matches = [p for p in all_polymers if polymer_name.upper() in p.upper()]

                if len(matches) > 0:
                    # Use the first match
                    polymer_data = hsp_data[hsp_data['Polymer'] == matches[0]]
                    polymer_name = matches[0]  # Update name to matched name
                    logger.info(f"Fuzzy matched '{polymer_name}' to '{matches[0]}'")
                else:
                    # Suggest similar polymers
                    suggestions = [p for p in all_polymers if any(term in p.upper() for term in ['PE', 'POLY', 'PET', 'PP', 'PVC', 'PS'])][:10]
                    suggestion_text = "\n- ".join(suggestions) if suggestions else "No suggestions available"
                    return f"Hansen parameters not found for polymer '{polymer_name}'.\n\n**Similar polymers you might try:**\n- {suggestion_text}"

            if len(polymer_data) == 0:
                return f"Hansen parameters not found for polymer '{polymer_name}'. Try listing available polymers."

            # Get polymer HSP values
            polymer_row = polymer_data.iloc[0]
            polymer_hsp = {
                'Dispersion': float(polymer_row['Polymer_Dispersion']),
                'Polar': float(polymer_row['Polymer_Polar']),
                'Hydrogen': float(polymer_row['Polymer_Hydrogen'])
            }
            r0 = float(polymer_row['R0'])

            # Common name to IUPAC name mapping for solvents
            common_to_iupac = {
                'acetone': 'Propan-2-one',
                'ethanol': 'Ethanol',
                'methanol': 'Methanol',
                'isopropanol': 'Propan-2-ol',
                'ipa': 'Propan-2-ol',
                'thf': 'Oxolane',
                'dmf': 'N,N-Dimethylformamide',
                'dmso': 'Dimethyl sulfoxide',
                'dma': 'N,N-Dimethylacetamide',
                'nmp': 'N-Methyl-2-pyrrolidone',
                'mek': 'Butan-2-one',
                'mibk': '4-Methylpentan-2-one',
                'dcm': 'Dichloromethane',
                'chloroform': 'Trichloromethane',
                'etoh': 'Ethanol',
                'meoh': 'Methanol',
                'acn': 'Acetonitrile',
                'dce': '1,2-Dichloroethane',
                'ea': 'Ethyl acetate',
                'ether': 'Diethyl ether',
                'hexane': 'Hexane',
                'heptane': 'Heptane',
                'octane': 'Octane',
                'decane': 'Decane',
                'benzene': 'Benzene',
                'toluene': 'Toluene',
                'xylene': 'Xylene',
                'water': 'Water',
                'dioxane': '1,4-Dioxane',
                'pyridine': 'Pyridine',
                'aniline': 'Aniline',
                'nitromethane': 'Nitromethane',
                'nitroethane': 'Nitroethane',
                'cyclohexane': 'Cyclohexane',
                'ccl4': 'Tetrachloromethane',
                'carbon tetrachloride': 'Tetrachloromethane',
                'carbon disulfide': 'Carbon disulfide',
                'cs2': 'Carbon disulfide',
                'butanol': 'Butan-1-ol',
                'propanol': 'Propan-1-ol',
                'pentane': 'Pentane',
                'butyl acetate': 'Butyl acetate',
                'methyl acetate': 'Methyl acetate',
                'propyl acetate': 'Propyl acetate'
            }

            # Find solvent - first try exact match
            solvent_data = hsp_data[hsp_data['Solvent'].str.lower() == solvent_name.lower()]

            # If not found, try common name mapping
            if len(solvent_data) == 0 and solvent_name.lower() in common_to_iupac:
                iupac_name = common_to_iupac[solvent_name.lower()]
                solvent_data = hsp_data[hsp_data['Solvent'].str.lower() == iupac_name.lower()]
                if len(solvent_data) > 0:
                    logger.info(f"Mapped common name '{solvent_name}' to IUPAC '{iupac_name}'")
                    solvent_name = iupac_name  # Update to IUPAC name for display

            # If still not found, try fuzzy matching
            if len(solvent_data) == 0:
                # Try partial string match first
                all_solvents = hsp_data['Solvent'].unique()
                matches = [s for s in all_solvents if solvent_name.upper() in s.upper()]

                if len(matches) > 0:
                    solvent_data = hsp_data[hsp_data['Solvent'] == matches[0]]
                    logger.info(f"Fuzzy matched '{solvent_name}' to '{matches[0]}'")
                    solvent_name = matches[0]
                else:
                    # Try database fuzzy matching as last resort
                    match_result = _fuzzy_match_solvent_name(solvent_name, dataset="all", threshold=80)
                    if match_result:
                        solvent_data = hsp_data[hsp_data['Solvent'].str.lower() == match_result["matched_name"].lower()]

            if len(solvent_data) == 0:
                return (
                    f"Hansen parameters not found for solvent '{solvent_name}'.\n\n"
                    "**Tip:** Common solvents in the database include:\n"
                    "- Water, Methanol, Ethanol, Isopropanol\n"
                    "- Acetone, MEK, MIBK\n"
                    "- Toluene, Benzene, Xylene\n"
                    "- THF, DMF, DMSO, NMP\n"
                    "- Hexane, Heptane, Cyclohexane\n"
                    "- Ethyl acetate, DCM, Chloroform\n\n"
                    "Try using `list_available_solvents()` for a complete list."
                )

            # Get solvent HSP values
            solvent_row = solvent_data.iloc[0]
            solvent_hsp = {
                'Dispersion': float(solvent_row['Solvent_Dispersion']),
                'Polar': float(solvent_row['Solvent_Polar']),
                'Hydrogen': float(solvent_row['Solvent_Hydrogen'])
            }
            molar_volume = float(solvent_row.get('Molar Volume', 100.0))

        except Exception as csv_error:
            logger.error(f"Error reading CSV: {csv_error}")
            return f"Error loading Hansen parameters: {str(csv_error)}"

        # Make prediction
        prediction = predictor.predict(polymer_hsp, solvent_hsp, r0, molar_volume)

        # Format output
        output = [f"**ML Solubility Prediction**\n"]
        output.append(f"**Polymer:** {polymer_name}")
        output.append(f"**Solvent:** {solvent_name}")
        output.append(f"**Temperature:** {temperature}\u00b0C\n")

        # Prediction result
        if prediction['soluble']:
            output.append(f"**Prediction:** SOLUBLE")
            output.append(f"**Probability:** {prediction['probability']*100:.1f}%")
        else:
            output.append(f"**Prediction:** NON-SOLUBLE")
            output.append(f"**Probability:** {(1-prediction['probability'])*100:.1f}%")

        output.append(f"**Confidence:** {prediction['confidence']*100:.1f}%")
        output.append(f"**RED Value:** {prediction['red']:.3f} (Hansen distance/R0)")
        output.append(f"**Ra (Hansen distance):** {prediction['ra']:.3f}")
        output.append(f"**R0 (Interaction radius):** {prediction['r0']:.3f}\n")

        # Interpretation
        output.append("**Interpretation:**")
        if prediction['red'] < 1.0:
            output.append(f"- RED < 1.0: Polymer and solvent are compatible (likely to dissolve)")
        else:
            output.append(f"- RED > 1.0: Polymer and solvent are incompatible (unlikely to dissolve)")

        # Generate visualizations
        if generate_visualizations:
            try:
                from visualization_library_v2 import generate_all_visualizations
                from datetime import datetime

                # Create subdirectory for full viz set
                safe_dirname = re.sub(r'[^\w\-]', '_', f"{polymer_name}_{solvent_name}")
                viz_dir = os.path.join(PLOTS_DIR, safe_dirname)
                os.makedirs(viz_dir, exist_ok=True)

                # Generate all visualizations in subdirectory
                viz_paths = generate_all_visualizations(
                    polymer_hsp=polymer_hsp,
                    solvent_hsp=solvent_hsp,
                    r0=r0,
                    polymer_name=polymer_name,
                    solvent_name=solvent_name,
                    prediction=prediction['soluble'],
                    probability=prediction['probability'],
                    output_dir=viz_dir
                )

                # Copy radar plot and RED gauge to root plots directory (so they auto-display)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r'[^\w\-]', '_', f"{polymer_name}_{solvent_name}")[:30]

                radar_src = viz_paths.get('Radar Plot')
                gauge_src = viz_paths.get('RED Gauge')

                if radar_src and os.path.exists(radar_src):
                    radar_dest = os.path.join(PLOTS_DIR, f"ml_radar_{safe_name}_{timestamp}.png")
                    shutil.copy(radar_src, radar_dest)

                if gauge_src and os.path.exists(gauge_src):
                    gauge_dest = os.path.join(PLOTS_DIR, f"ml_gauge_{safe_name}_{timestamp}.png")
                    shutil.copy(gauge_src, gauge_dest)

                # Copy 3D sphere HTML to root plots directory for easy access
                sphere_src = viz_paths.get('3D Sphere (Interactive HTML)')
                if sphere_src and os.path.exists(sphere_src):
                    sphere_dest = os.path.join(PLOTS_DIR, f"ml_sphere_{safe_name}_{timestamp}.html")
                    shutil.copy(sphere_src, sphere_dest)

                    # Add link to 3D sphere (opens in new tab)
                    import urllib.parse
                    sphere_filename = os.path.basename(sphere_dest)
                    sphere_url = f"/plots/{sphere_filename}"
                    # Use markdown link syntax (not HTML) for proper rendering
                    output.append(f"\n**Interactive 3D Visualization:** [Click to open Hansen Sphere]({sphere_url})")
                    output.append(f"\n**Tip:** The 3D sphere opens in a new tab - you can rotate, zoom, and explore the Hansen space!")

            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
                output.append(f"\nNote: Visualization generation encountered an issue: {str(viz_error)}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in predict_solubility_ml: {e}")
        return f"Error making ML prediction: {str(e)}"
