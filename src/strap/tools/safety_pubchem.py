"""PubChem safety and toxicity data retrieval tools."""
from __future__ import annotations
import gc
import json
import logging
import os
import asyncio
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional
from strap.services.tool_response_service import json_tool_error, json_tool_response
from strap.tools._helpers import safe_tool_wrapper, truncate_output, save_plot, get_plots_dir
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------
# PubChem rate-limiting and retry helpers
# ------------------------------------------------------------------
_last_pubchem_request = 0.0
def _rate_limit_pubchem():
    global _last_pubchem_request
    elapsed = time.time() - _last_pubchem_request
    if elapsed < 0.25:  # max ~4 req/s, under PubChem's 5/s limit
        time.sleep(0.25 - elapsed)
    _last_pubchem_request = time.time()
def _pubchem_request(url: str, timeout: int = 15, max_retries: int = 3):
    """Fetch a URL from PubChem with rate limiting and exponential backoff on errors."""
    for attempt in range(max_retries):
        _rate_limit_pubchem()
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                        "User-Agent": "PolymerSolubilityApp/1.0"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            code = getattr(e, "code", None)
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.debug(
                    "PubChem request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
                continue
            raise
    return None
# ------------------------------------------------------------------
# Helper functions (not tool-wrapped)
# ------------------------------------------------------------------
def fetch_pubchem_cid(compound_name: str) -> Optional[int]:
    """Fetch PubChem CID (Compound ID) for a given compound name."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(compound_name)}/cids/JSON"
        raw = _pubchem_request(url, timeout=10)
        if raw is None:
            return None
        data = json.loads(raw.decode())
        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            return data['IdentifierList']['CID'][0]
    except Exception as e:
        logger.warning(f"Could not fetch CID for {compound_name}: {e}")
    return None
def fetch_pubchem_properties(cid: int) -> Optional[Dict]:
    """Fetch compound properties from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight/JSON"
        raw = _pubchem_request(url, timeout=10)
        if raw is None:
            return None
        data = json.loads(raw.decode())
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            return data['PropertyTable']['Properties'][0]
    except Exception as e:
        logger.warning(f"Could not fetch properties for CID {cid}: {e}")
    return None
def fetch_pubchem_ghs_data(cid: int) -> Optional[Dict]:
    """Fetch GHS safety classification data from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        raw = _pubchem_request(url, timeout=15)
        if raw is None:
            return None
        data = json.loads(raw.decode())
        result = {
            'cid': cid,
            'pictograms': [],
            'signal_word': None,
            'hazard_statements': [],
            'precautionary_codes': []
        }
        # Parse the nested JSON structure for GHS data
        def extract_ghs_info(obj):
            if isinstance(obj, dict):
                name = obj.get('Name', '')
                if name == 'Signal':
                    value = obj.get('Value', {})
                    if 'StringWithMarkup' in value:
                        for item in value['StringWithMarkup']:
                            if 'String' in item:
                                result['signal_word'] = item['String']
                elif name == 'Pictogram(s)':
                    value = obj.get('Value', {})
                    if 'StringWithMarkup' in value:
                        for item in value['StringWithMarkup']:
                            if 'Markup' in item:
                                for markup in item['Markup']:
                                    if 'Extra' in markup:
                                        result['pictograms'].append(markup['Extra'])
                elif name == 'GHS Hazard Statements':
                    value = obj.get('Value', {})
                    if 'StringWithMarkup' in value:
                        for item in value['StringWithMarkup']:
                            if 'String' in item:
                                result['hazard_statements'].append(item['String'])
                elif name == 'Precautionary Statement Codes':
                    value = obj.get('Value', {})
                    if 'StringWithMarkup' in value:
                        for item in value['StringWithMarkup']:
                            if 'String' in item:
                                result['precautionary_codes'].append(item['String'])
                # Recurse into nested structures
                for key, val in obj.items():
                    extract_ghs_info(val)
            elif isinstance(obj, list):
                for item in obj:
                    extract_ghs_info(item)
        extract_ghs_info(data)
        # Remove duplicates
        result['pictograms'] = list(set(result['pictograms']))
        result['hazard_statements'] = list(set(result['hazard_statements']))
        return result
    except Exception as e:
        logger.warning(f"Could not fetch GHS data for CID {cid}: {e}")
    return None
def fetch_pubchem_toxicity_data(cid: int) -> Optional[Dict]:
    """Fetch toxicity and environmental data from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Toxicity"
        raw = _pubchem_request(url, timeout=20)
        if raw is None:
            return None
        data = json.loads(raw.decode())
        result = {
            'cid': cid,
            'ld50_values': [],
            'lc50_values': [],
            'biodegradation': [],
            'aquatic_toxicity': [],
            'ecological_info': []
        }
        def extract_toxicity_info(obj, current_heading=''):
            if isinstance(obj, dict):
                heading = obj.get('TOCHeading', current_heading)
                # Check for toxicity values in Information sections
                if 'Information' in obj:
                    for info in obj['Information']:
                        value = info.get('Value', {})
                        string_value = ''
                        if 'StringWithMarkup' in value:
                            for item in value['StringWithMarkup']:
                                if 'String' in item:
                                    string_value = item['String']
                                    break
                        elif 'Number' in value:
                            string_value = str(value['Number'])
                        if string_value:
                            # Categorize by heading
                            heading_lower = heading.lower()
                            if 'ld50' in heading_lower or 'lethal dose' in heading_lower:
                                if len(result['ld50_values']) < 5:
                                    result['ld50_values'].append(string_value[:200])
                            elif 'lc50' in heading_lower or 'lethal concentration' in heading_lower:
                                if len(result['lc50_values']) < 3:
                                    result['lc50_values'].append(string_value[:200])
                            elif 'biodegradation' in heading_lower or 'biodegradability' in heading_lower:
                                if len(result['biodegradation']) < 3:
                                    result['biodegradation'].append(string_value[:200])
                            elif 'aquatic' in heading_lower or 'fish' in heading_lower or 'daphnia' in heading_lower:
                                if len(result['aquatic_toxicity']) < 3:
                                    result['aquatic_toxicity'].append(string_value[:200])
                            elif 'ecological' in heading_lower or 'environmental' in heading_lower:
                                if len(result['ecological_info']) < 3:
                                    result['ecological_info'].append(string_value[:200])
                # Recurse into sections
                if 'Section' in obj:
                    for section in obj['Section']:
                        extract_toxicity_info(section, heading)
                for key, val in obj.items():
                    if key not in ['Section', 'Information']:
                        extract_toxicity_info(val, heading)
            elif isinstance(obj, list):
                for item in obj:
                    extract_toxicity_info(item, current_heading)
        extract_toxicity_info(data)
        return result
    except Exception as e:
        logger.warning(f"Could not fetch toxicity data for CID {cid}: {e}")
    return None
# ------------------------------------------------------------------
# Helper: plot URL display
# ------------------------------------------------------------------
def _get_plot_url(filepath: str) -> str:
    """Convert filepath to displayable format."""
    return f"Plot saved: `{filepath}`"
def _pubchem_response(tool_name: str, display: str, **data) -> str:
    return json_tool_response(display, data, tool_name=tool_name)
def _pubchem_error(tool_name: str, message: str, *, error_code: str = "invalid_input", **data) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)
# ------------------------------------------------------------------
# Tool functions (wrapped with @safe_tool_wrapper)
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
async def get_pubchem_safety_info(compound_name: str) -> str:
    """Fetch GHS hazard classification and molecular properties from PubChem for a compound.
    Args:
        compound_name: Name of the compound (e.g., "toluene", "ethanol")
    WHEN TO USE:
    - "What are the safety hazards for toluene?"
    - "Is dichloromethane dangerous?"
    - "Get PubChem safety data for acetone"
    """
    output = []
    # Normalize common solvent names
    name_mapping = {
        'dcm': 'dichloromethane',
        'dmf': 'dimethylformamide',
        'dmso': 'dimethyl sulfoxide',
        'thf': 'tetrahydrofuran',
        'mek': 'methyl ethyl ketone',
        'mibk': 'methyl isobutyl ketone',
        'ipa': 'isopropanol',
        'etoh': 'ethanol',
        'meoh': 'methanol',
        'acn': 'acetonitrile'
    }
    search_name = name_mapping.get(compound_name.lower().strip(), compound_name)
    # Step 1: Get CID
    cid = fetch_pubchem_cid(search_name)
    if not cid:
        msg = f"Compound '{compound_name}' not found in PubChem. Try using the full chemical name or check spelling."
        return _pubchem_error(
            "get_pubchem_safety_info",
            msg,
            error_code="compound_not_found",
            found=False,
            compound_name=compound_name,
            search_name=search_name,
        )
    output.append(f"# PubChem Safety Profile: {compound_name.title()}\n")
    output.append(f"**PubChem CID:** [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})\n")
    # Step 2: Get molecular properties (compact)
    props = fetch_pubchem_properties(cid)
    if props:
        parts = []
        if 'MolecularFormula' in props:
            parts.append(f"Formula: {props['MolecularFormula']}")
        if 'MolecularWeight' in props:
            try:
                parts.append(f"MW: {float(props['MolecularWeight']):.1f} g/mol")
            except (ValueError, TypeError):
                parts.append(f"MW: {props['MolecularWeight']} g/mol")
        if parts:
            output.append(f"**Identity:** {' | '.join(parts)}\n")
    # Step 3: Get GHS safety data
    ghs_data = fetch_pubchem_ghs_data(cid)
    if ghs_data:
        output.append("## GHS Hazard Classification\n")
        # Signal word
        if ghs_data.get('signal_word'):
            signal = ghs_data['signal_word']
            output.append(f"**Signal Word:** {signal}\n")
        # Pictograms
        if ghs_data.get('pictograms'):
            output.append("**Hazard Pictograms:**")
            for pic in ghs_data['pictograms']:
                output.append(f"- {pic}")
            output.append("")
        # Hazard statements
        if ghs_data.get('hazard_statements'):
            output.append("**Hazard Statements:**")
            for stmt in ghs_data['hazard_statements'][:5]:  # Limit to top 5
                output.append(f"- {stmt}")
            if len(ghs_data['hazard_statements']) > 5:
                output.append(f"- *...and {len(ghs_data['hazard_statements']) - 5} more*")
            output.append("")
        # Short GHS interpretation
        signal = ghs_data.get('signal_word', '')
        n_pics = len(ghs_data.get('pictograms', []))
        if signal == 'Danger' or n_pics >= 3:
            output.append("**GHS Assessment:** Significant hazards — handle with full PPE and engineering controls.\n")
        elif signal == 'Warning' or n_pics >= 1:
            output.append("**GHS Assessment:** Moderate hazards — standard precautions required.\n")
        else:
            output.append("**GHS Assessment:** Minimal GHS hazards identified.\n")
    else:
        output.append("## GHS Hazard Classification\n")
        output.append("*No GHS classification data available for this compound.*\n")
    # Add link to full PubChem page
    output.append(f"\n**Full Safety Data:** [View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Safety-and-Hazards)")
    display_str = "\n".join(output)
    data_dict = {
        "found": True,
        "compound_name": search_name,
        "cid": cid,
        "molecular_formula": props.get('MolecularFormula') if props else None,
        "molecular_weight": float(props['MolecularWeight']) if props and 'MolecularWeight' in props else None,
        "signal_word": ghs_data.get('signal_word') if ghs_data else None,
        "pictograms": ghs_data.get('pictograms', []) if ghs_data else [],
        "hazard_statements": ghs_data.get('hazard_statements', []) if ghs_data else [],
    }
    return _pubchem_response("get_pubchem_safety_info", display_str, **data_dict)
@safe_tool_wrapper(structured_output=True)
async def compare_pubchem_safety(compounds: List[str]) -> str:
    """Compare GHS hazard profiles of multiple compounds using PubChem data.
    Args:
        compounds: List of compound names to compare (2-5 max)
    WHEN TO USE:
    - "Compare the safety of toluene, benzene, and ethanol"
    - "Which is safer: DCM or chloroform?"
    """
    if isinstance(compounds, str):
        compounds = [c.strip() for c in compounds.split(",") if c.strip()]
    if len(compounds) < 2:
        msg = "Please provide at least 2 compounds to compare."
        return _pubchem_error(
            "compare_pubchem_safety",
            msg,
            error_code="too_few_compounds",
            compounds=compounds,
        )
    if len(compounds) > 5:
        compounds = compounds[:5]
    output = [f"# PubChem GHS Hazard Comparison\n"]
    # Collect data for all compounds
    compound_data = []
    for name in compounds:
        cid = fetch_pubchem_cid(name)
        if cid:
            ghs = fetch_pubchem_ghs_data(cid)
            compound_data.append({
                'name': name.title(),
                'cid': cid,
                'signal_word': ghs.get('signal_word') if ghs else None,
                'pictograms': ghs.get('pictograms', []) if ghs else [],
                'hazard_statements': ghs.get('hazard_statements', []) if ghs else [],
            })
        else:
            compound_data.append({
                'name': name.title(),
                'cid': None,
                'signal_word': None,
                'pictograms': [],
                'hazard_statements': [],
            })
    # Display each compound's hazards
    for comp in compound_data:
        if comp['cid'] is None:
            output.append(f"### {comp['name']}\n*Not found in PubChem*\n")
            continue
        signal = comp['signal_word'] or "None"
        output.append(f"### {comp['name']}")
        output.append(f"**Signal Word:** {signal}")
        if comp['pictograms']:
            hazard_list = [p for p in comp['pictograms']]
            output.append(f"**Hazards:** {', '.join(hazard_list)}")
        else:
            output.append("**Hazards:** None listed")
        if comp['hazard_statements']:
            output.append(f"**Key Statements:** {comp['hazard_statements'][0][:80]}...")
        output.append("")
    # Generate contextual recommendation
    output.append("## Recommendation\n")
    valid_data = [c for c in compound_data if c['cid'] is not None]
    safest_name = None
    most_hazardous_name = None
    if valid_data:
        # Rank by: no Danger signal > Warning > Danger, then fewer pictograms
        def hazard_rank(c):
            signal_rank = 0 if c['signal_word'] is None else (1 if c['signal_word'] == 'Warning' else 2)
            has_toxic = 1 if 'Acute Toxic' in c['pictograms'] or 'Health Hazard' in c['pictograms'] else 0
            has_flammable = 1 if 'Flammable' in c['pictograms'] else 0
            return (signal_rank, has_toxic, has_flammable, len(c['pictograms']))
        ranked = sorted(valid_data, key=hazard_rank)
        best = ranked[0]
        worst = ranked[-1]
        safest_name = best['name']
        most_hazardous_name = worst['name']
        # Build contextual summary
        if best['signal_word'] is None or best['signal_word'] == 'Warning':
            if worst['signal_word'] == 'Danger':
                output.append(f"**{best['name']}** appears to be the safer choice - it has a '{best['signal_word'] or 'no'}' signal word compared to **{worst['name']}**'s 'Danger' classification.")
            else:
                output.append(f"All compounds have similar hazard levels. **{best['name']}** has the fewest hazard categories ({len(best['pictograms'])}).")
        else:
            output.append(f"All compounds carry 'Danger' signal words. **{best['name']}** has fewer hazard categories ({len(best['pictograms'])} vs {len(worst['pictograms'])} for {worst['name']}).")
        # Specific warnings
        for comp in valid_data:
            if 'Acute Toxic' in comp['pictograms']:
                output.append(f"\n**{comp['name']}** is classified as acutely toxic - requires special handling.")
            if 'Health Hazard' in comp['pictograms']:
                output.append(f"\n**{comp['name']}** has serious health hazards (may be carcinogenic or cause organ damage).")
    output.append("\n*Data sourced from PubChem GHS Classification*")
    display_str = "\n".join(output)
    compounds_list = []
    for c in compound_data:
        compounds_list.append({
            "name": c['name'],
            "cid": c.get('cid'),
            "signal_word": c.get('signal_word'),
            "n_pictograms": len(c.get('pictograms', [])),
        })
    data_dict = {
        "compounds": compounds_list,
        "safest_compound": safest_name,
        "most_hazardous_compound": most_hazardous_name,
    }
    data_dict["n_compounds"] = len(compounds_list)
    return _pubchem_response("compare_pubchem_safety", display_str, **data_dict)
@safe_tool_wrapper(structured_output=True)
async def visualize_pubchem_safety(
    compounds: List[str],
    chart_type: str = "hazards"
) -> str:
    """Create a stacked bar chart comparing GHS hazard categories from PubChem.
    Args:
        compounds: List of compound names (2-5 max)
        chart_type: "hazards" for hazard count bar chart (default)
    WHEN TO USE:
    - "Create a safety comparison chart for toluene, benzene, and xylene"
    - "Visualize PubChem hazard data for common solvents"
    """
    if plt is None:
        msg = "matplotlib is not installed. Cannot generate safety chart."
        return _pubchem_error(
            "visualize_pubchem_safety",
            msg,
            error_code="matplotlib_unavailable",
        )
    if len(compounds) < 2:
        msg = "Please provide at least 2 compounds to visualize."
        return _pubchem_error(
            "visualize_pubchem_safety",
            msg,
            error_code="too_few_compounds",
            compounds=compounds,
        )
    if len(compounds) > 5:
        compounds = compounds[:5]
    # Collect data
    compound_data = []
    for name in compounds:
        cid = fetch_pubchem_cid(name)
        if cid:
            ghs = fetch_pubchem_ghs_data(cid)
            compound_data.append({
                'name': name.title(),
                'cid': cid,
                'signal_word': ghs.get('signal_word') if ghs else None,
                'pictograms': ghs.get('pictograms', []) if ghs else [],
                'n_pictograms': len(ghs.get('pictograms', [])) if ghs else 0,
            })
    if len(compound_data) < 2:
        msg = "Could not fetch safety data for enough compounds. Try different compound names."
        return _pubchem_error(
            "visualize_pubchem_safety",
            msg,
            error_code="insufficient_data",
            compounds=compounds,
        )
    # Sort by number of hazards (fewer = better)
    compound_data.sort(key=lambda x: x['n_pictograms'])
    # Create visualization - stacked bar showing hazard types
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [c['name'] for c in compound_data]
    # Define hazard categories and colors
    hazard_types = ['Flammable', 'Irritant', 'Health Hazard', 'Acute Toxic', 'Corrosive', 'Environmental Hazard', 'Oxidizer', 'Explosive']
    hazard_colors = ['#e74c3c', '#f39c12', '#9b59b6', '#2c3e50', '#1abc9c', '#27ae60', '#e67e22', '#c0392b']
    # Build data matrix
    y_pos = range(len(names))
    bar_data = {h: [] for h in hazard_types}
    for comp in compound_data:
        for hazard in hazard_types:
            bar_data[hazard].append(1 if hazard in comp['pictograms'] else 0)
    # Create stacked horizontal bars
    left = [0] * len(names)
    for hazard, color in zip(hazard_types, hazard_colors):
        values = bar_data[hazard]
        if sum(values) > 0:  # Only show hazards that exist
            ax.barh(y_pos, values, left=left, label=hazard, color=color, edgecolor='white', linewidth=0.5)
            left = [l + v for l, v in zip(left, values)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of GHS Hazard Categories', fontsize=14, fontweight='bold')
    ax.set_title('PubChem GHS Hazard Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    # Add signal word annotations
    for i, comp in enumerate(compound_data):
        signal = comp['signal_word'] or "None"
        color = '#e74c3c' if signal == 'Danger' else '#f39c12' if signal == 'Warning' else '#27ae60'
        ax.annotate(f"  {signal}", (comp['n_pictograms'], i), va='center', fontsize=10, color=color, fontweight='bold')
    ax.set_xlim(0, max(c['n_pictograms'] for c in compound_data) + 2)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    # Save
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pubchem_hazards_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # Build output
    output = ["**PubChem GHS Hazard Chart**\n"]
    # Summary for each compound
    output.append("**Hazard Summary:**")
    for comp in compound_data:
        hazards = ", ".join(comp['pictograms']) if comp['pictograms'] else "None"
        output.append(f"- **{comp['name']}**: {comp['signal_word'] or 'No signal'} | {hazards}")
    output.append(f"\n{_get_plot_url(filepath)}")
    display_str = "\n".join(output)
    compound_summary = []
    for comp in compound_data:
        compound_summary.append({
            "name": comp['name'],
            "cid": comp.get('cid'),
            "signal_word": comp.get('signal_word'),
            "n_pictograms": comp.get('n_pictograms', 0),
            "pictograms": comp.get('pictograms', []),
        })
    data_dict = {
        "success": True,
        "filepath": filepath,
        "compounds": compound_summary,
    }
    gc.collect()
    data_dict["n_compounds"] = len(compound_summary)
    return _pubchem_response("visualize_pubchem_safety", display_str, **data_dict)
@safe_tool_wrapper(structured_output=True)
async def get_pubchem_toxicity(compounds: List[str]) -> str:
    """Fetch LD50, LC50, biodegradation, and aquatic toxicity data from PubChem.
    Args:
        compounds: List of compound names (1-5)
    WHEN TO USE:
    - "What's the LD50 of toluene and benzene?"
    - "Is acetone biodegradable?"
    - "Compare the environmental toxicity of DCM vs chloroform"
    """
    if isinstance(compounds, str):
        compounds = [c.strip() for c in compounds.split(",") if c.strip()]
    if not compounds:
        msg = "No compounds provided. Pass a comma-separated list of chemical names."
        return _pubchem_error(
            "get_pubchem_toxicity",
            msg,
            error_code="missing_compounds",
            compounds=[],
        )
    if len(compounds) > 5:
        compounds = compounds[:5]
    output = [f"# PubChem Toxicity & Environmental Data\n"]
    compound_data = []
    for name in compounds:
        cid = fetch_pubchem_cid(name)
        if cid:
            tox = fetch_pubchem_toxicity_data(cid)
            compound_data.append({
                'name': name.title(),
                'cid': cid,
                'toxicity': tox
            })
        else:
            compound_data.append({
                'name': name.title(),
                'cid': None,
                'toxicity': None
            })
    # Display each compound's data
    for comp in compound_data:
        if comp['cid'] is None:
            output.append(f"### {comp['name']}\n*Not found in PubChem*\n")
            continue
        output.append(f"### {comp['name']}")
        output.append(f"[PubChem CID: {comp['cid']}](https://pubchem.ncbi.nlm.nih.gov/compound/{comp['cid']}#section=Toxicity)\n")
        tox = comp['toxicity']
        if not tox:
            output.append("*No toxicity data available*\n")
            continue
        # LD50 Values
        if tox.get('ld50_values'):
            output.append("**LD50 (Lethal Dose):**")
            for val in tox['ld50_values'][:3]:
                output.append(f"- {val}")
            output.append("")
        # LC50 Values
        if tox.get('lc50_values'):
            output.append("**LC50 (Lethal Concentration):**")
            for val in tox['lc50_values'][:2]:
                output.append(f"- {val}")
            output.append("")
        # Biodegradation
        if tox.get('biodegradation'):
            output.append("**Biodegradation:**")
            for val in tox['biodegradation'][:2]:
                output.append(f"- {val}")
            output.append("")
        # Aquatic Toxicity
        if tox.get('aquatic_toxicity'):
            output.append("**Aquatic Toxicity:**")
            for val in tox['aquatic_toxicity'][:2]:
                output.append(f"- {val}")
            output.append("")
        # Check if no data found
        has_data = any([tox.get('ld50_values'), tox.get('lc50_values'),
                       tox.get('biodegradation'), tox.get('aquatic_toxicity')])
        if not has_data:
            output.append("*Limited toxicity data available for this compound*\n")
    # Summary comparison if multiple compounds
    if len(compound_data) > 1:
        output.append("## Summary\n")
        # Find compounds with LD50 data for comparison
        with_ld50 = [c for c in compound_data if c['toxicity'] and c['toxicity'].get('ld50_values')]
        if with_ld50:
            output.append("**Toxicity Comparison:**")
            for comp in with_ld50:
                ld50_sample = comp['toxicity']['ld50_values'][0][:100] if comp['toxicity']['ld50_values'] else "N/A"
                output.append(f"- **{comp['name']}**: {ld50_sample}...")
        # Biodegradation summary
        with_biodeg = [c for c in compound_data if c['toxicity'] and c['toxicity'].get('biodegradation')]
        if with_biodeg:
            output.append("\n**Biodegradability:**")
            for comp in with_biodeg:
                biodeg = comp['toxicity']['biodegradation'][0][:80] if comp['toxicity']['biodegradation'] else "Unknown"
                output.append(f"- **{comp['name']}**: {biodeg}...")
    output.append("\n*Data sourced from PubChem Toxicity database*")
    display_str = "\n".join(output)
    compounds_list = []
    for comp in compound_data:
        tox = comp.get('toxicity') or {}
        compounds_list.append({
            "name": comp['name'],
            "cid": comp.get('cid'),
            "ld50_values": tox.get('ld50_values', []),
            "lc50_values": tox.get('lc50_values', []),
            "has_toxicity_data": bool(tox.get('ld50_values') or tox.get('lc50_values')),
        })
    data_dict = {
        "compounds": compounds_list,
        "n_compounds": len(compound_data),
        "n_with_ld50": sum(1 for c in compound_data if c.get('toxicity') and c['toxicity'].get('ld50_values')),
    }
    return _pubchem_response("get_pubchem_toxicity", display_str, **data_dict)
