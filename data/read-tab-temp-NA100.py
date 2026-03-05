# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 11:14:14 2025

@author: aaltamimi2
"""

# read_tab_TEMP_NAME.py
import re
import csv
import sys
from pathlib import Path

TABFILE = 'nylon6-solubility-GVL.tab'   # adjust path if needed
OUT     = 'nylon6-solubility-GVL.csv'

# Parse temperature (K) from lines like: "Settings  job X : T= 433.15 K ;"
RE_TEMP = re.compile(r'T=\s*([0-9]+(?:\.[0-9]+)?)\s*K')

rows = []                 # will collect rows IN INPUT ORDER
temps_seen = []

current_temp_C = None
in_table = False

# cp1252 handles degree symbol, etc.; replace undecodable chars
with open(TABFILE, 'r', encoding='cp1252', errors='replace') as f:
    for raw in f:
        line = raw.rstrip('\n')

        # 1) Find temperature blocks
        m = RE_TEMP.search(line)
        if m:
            T_K = float(m.group(1))
            current_temp_C = int(round(T_K - 273.15))  # keep your integer °C mapping
            if current_temp_C not in temps_seen:
                temps_seen.append(current_temp_C)
            in_table = False
            continue

        # 2) Detect the table header line
        if line.strip().startswith('Nr ') and 'Solvent' in line:
            in_table = True
            continue

        if not in_table:
            continue

        # 3) Data rows: start with an index number
        stripped = line.strip()
        if not stripped or not re.match(r'^\d+\s+', stripped):
            if stripped.startswith('Property') or stripped.startswith('Settings'):
                in_table = False
            continue

        # Split on 2+ spaces to preserve multi-word solvent names
        cols = re.split(r'\s{2,}', stripped)
        # Expected after split:
        # 0: "<idx> <solvent name>"
        # 1: log10(x_solub)   2: mu(self)   3: mu(solv)
        # 4: w_solub          5: log10(S)   6: density   7: MW
        if len(cols) < 5:
            continue

        try:
            _, solvent = cols[0].split(' ', 1)
        except ValueError:
            continue

        solvent = solvent.strip()           # KEEP ORIGINAL CASING
        if not solvent or current_temp_C is None:
            continue

        w_raw = cols[4].strip().strip('[]')  # e.g., "NA" or "1.23e-4"

        # ---- RULE: Any NA -> 100% ----
        if not w_raw or w_raw.upper() in {'NA', 'N/A', 'NULL', 'NONE'}:
            sol_percent = 100.0
        else:
            try:
                sol_percent = float(w_raw) * 100.0
            except ValueError:
                sol_percent = 100.0

        # APPEND in the order encountered (no dict, no sorting)
        rows.append((solvent, current_temp_C, sol_percent))

# 4) Sanity check
if not rows:
    print("❌ No data parsed! Check your file path & format.", file=sys.stderr)
    sys.exit(1)

# 5) Write CSV in the SAME ORDER as parsed
with open(OUT, 'w', newline='', encoding='utf-8-sig') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['Solvent', 'Temperature (°C)', 'Solubility (%)'])
    for name, temp, solub in rows:
        writer.writerow([name, temp, f"{solub:.8f}"])

print(f"✅ Parsed {len({r[0] for r in rows})} solvents across {len(set(r[1] for r in rows))} temperatures")
print(f"📄 Wrote: {Path(OUT).resolve()}")

