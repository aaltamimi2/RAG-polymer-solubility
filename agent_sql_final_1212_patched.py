# -*- coding: utf-8 -*-
"""Agent-SQL-FINAL-1212-PATCHED.py

Patched version with:
- Memory-efficient state management
- Graceful error handling and recovery
- Proper resource cleanup
- Timeout protection
- Session memory limits
- Tool output truncation
- Garbage collection
"""

# ============================================================
# INSTALLATION (run this cell first in Colab)
# ============================================================
# !pip install -U "google-generativeai>=0.8.3" "langchain-google-genai>=2.0.9" duckdb gradio langchain langgraph langchain-core

import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# API Keys setup - load from environment variables
# Set these in your environment or .env file before running
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable is required. Set it before running.")

# Optional: LangSmith for tracing (not required)
# if "LANGSMITH_API_KEY" not in os.environ:
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith AI API key:")

# Default LLM (can be overridden via config)
DEFAULT_MODEL = "gemini-2.5-flash-lite"

def create_llm(model_name: str = None):
    """Create LLM with specified model name."""
    model = model_name or DEFAULT_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
    )

# Default LLM instance
llm = create_llm()

"""
Enhanced SQL Agent for Polymer Solubility Analysis
==================================================
Features:
- Adaptive threshold searching (stringent to lenient)
- Temperature exploration for optimal separation
- Extensive data verification and validation
- Flexible polymer comparisons (only compare requested polymers)
- Statistical analysis tools
- Enhanced visualizations
- Hallucination prevention through cross-validation
- PATCHED: Memory efficiency & error handling
"""

import os
import glob
import json
import uuid
import re
import gc
import traceback
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any, Tuple, Union
import logging
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from enum import Enum
import time
import pandas as pd
import duckdb
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fuzzy matching for solvent name normalization
from rapidfuzz import fuzz, process

# Async utilities for concurrent execution
import asyncio
from async_utils import run_in_thread
from async_db import AsyncDuckDBWrapper

from langchain_core.tools import tool

# TEA/LCA Module - standalone file for easy editing by TEA/LCA specialists
import tea_lca_module as tea_lca

# RAG Module - Literature search with vector retrieval
import rag_module as rag

# Advanced Separation Tools - Modular algorithms for separation optimization
try:
    from tools.langchain_tools import ADVANCED_SEPARATION_TOOLS
except ImportError:
    ADVANCED_SEPARATION_TOOLS = []  # Fallback if tools module not available

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "./data"
SQL_DB_PATH = ":memory:"
PLOTS_DIR = "./plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Memory management constants
MAX_ITERATIONS = 15  # Reduced from 50 to prevent endless loops
MAX_MESSAGE_HISTORY = 50
MAX_TOOL_OUTPUT_LENGTH = 50000
MAX_PLOTS_TO_KEEP = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================
# Memory Management Utilities (NEW)
# ============================================================

def truncate_output(text: str, max_length: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
    """Truncate tool output to prevent memory issues."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 50
    return text[:half] + f"\n\n... [TRUNCATED {len(text) - max_length} chars] ...\n\n" + text[-half:]


def _format_tool_result(result) -> str:
    """Format tool result with truncation if needed."""
    if result is None:
        return "Operation completed (no output)."
    result_str = str(result)
    if len(result_str) > MAX_TOOL_OUTPUT_LENGTH:
        return truncate_output(result_str)
    return result_str


def _format_tool_error(func_name: str, error: Exception) -> str:
    """Format tool error with suggestions."""
    return (
        f"ERROR in {func_name}:\n"
        f"{str(error)[:500]}\n\n"
        f"Suggestions:\n"
        f"- Verify input parameters with describe_table()\n"
        f"- Check values with check_column_values()\n"
        f"- Use verify_data_accuracy() to confirm data exists"
    )


def safe_tool_wrapper(func):
    """Decorator for safe tool execution with error handling and memory cleanup (async-compatible)."""
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return _format_tool_result(result)
            except Exception as e:
                logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
                return _format_tool_error(func.__name__, e)
            finally:
                gc.collect()
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return _format_tool_result(result)
            except Exception as e:
                logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
                return _format_tool_error(func.__name__, e)
            finally:
                gc.collect()
        return sync_wrapper


# ============================================================
# Fuzzy Matching Utilities for Solvent Name Normalization
# ============================================================

def _search_fuzzy_match_in_dataset(
    sql_db,
    query: str,
    column_name: str,
    dataset_name: str,
    solvent_name_clean: str,
    current_best_score: int
) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Search a single dataset for fuzzy match.

    Args:
        sql_db: Database connection
        query: SQL query to get solvent names
        column_name: Column containing solvent names
        dataset_name: Name of dataset for logging
        solvent_name_clean: Normalized solvent name to match
        current_best_score: Current best score to beat

    Returns:
        Tuple of (matched_name, score, dataset_name) or (None, current_best_score, None)
    """
    try:
        result = sql_db.execute_query(query)
        if result["success"] and len(result["dataframe"]) > 0:
            names = result["dataframe"][column_name].tolist()
            names_lower = [n.lower() for n in names]
            match = process.extractOne(solvent_name_clean, names_lower, scorer=fuzz.ratio)
            if match and match[1] > current_best_score:
                idx = names_lower.index(match[0])
                return names[idx], match[1], dataset_name
    except Exception as e:
        logger.debug(f"{dataset_name} search failed: {e}")
    return None, current_best_score, None


def fuzzy_match_solvent_name(solvent_name: str, dataset: str = "all", threshold: int = 80) -> Optional[Dict[str, Any]]:
    """
    Find the best matching solvent name across datasets using fuzzy matching.

    Args:
        solvent_name: The solvent name to match
        dataset: Which dataset to search ("gsk", "solvent_data", "common_solvents", or "all")
        threshold: Minimum similarity score (0-100) to accept a match

    Returns:
        Dict with matched name, score, and dataset, or None if no good match found
    """
    try:
        global sql_db
        best_match = None
        best_score = 0
        best_dataset = None
        solvent_name_clean = solvent_name.strip().lower()

        # Dataset configurations: (dataset_key, query, column_name, dataset_name)
        dataset_configs = [
            ("gsk", "SELECT DISTINCT solvent_common_name FROM gsk_dataset", "solvent_common_name", "gsk_dataset"),
            ("solvent_data", "SELECT DISTINCT cosmobase_name FROM solvent_data", "cosmobase_name", "solvent_data"),
            ("common_solvents", "SELECT DISTINCT solvent FROM common_solvents_database", "solvent", "common_solvents_database"),
        ]

        for ds_key, query, column, ds_name in dataset_configs:
            if dataset in [ds_key, "all"]:
                match, score, matched_ds = _search_fuzzy_match_in_dataset(
                    sql_db, query, column, ds_name, solvent_name_clean, best_score
                )
                if match:
                    best_match, best_score, best_dataset = match, score, matched_ds

        if best_score >= threshold:
            return {
                "matched_name": best_match,
                "score": best_score,
                "dataset": best_dataset,
                "original_query": solvent_name
            }

        return None

    except Exception as e:
        logger.error(f"Fuzzy matching error: {e}")
        return None


# ============================================================
# Static Solvent Name Mapping (Cross-Database Normalization)
# ============================================================

# Maps solubility database names -> (property database name, GSK database name)
# None means the solvent is not in that database
SOLVENT_NAME_MAP = {
    '1,2-dimethylbenzene': ('o-Xylene', 'o-Xylene'),
    '1,4-dimethylbenzene': ('p-Xylene', 'p-Xylene'),
    '2,3-dihydropyran': (None, None),
    '2-propanol': ('2-Propanol', '2-Propanol'),
    'acetylacetone': ('2,4-Pentanedione', None),
    'benzene': ('Benzene', 'Benzene'),
    'butanone': ('Methyl ethyl ketone', '2-Butanone'),
    'ch2cl2': ('Dichloromethane', 'Dichloromethane'),
    'chcl3': ('Chloroform', 'Chloroform'),
    'cyclohexane': ('Cyclohexane', 'Cyclohexane'),
    'cyclohexanol': ('Cyclohexanol', 'Cyclohexanol'),
    'dimethylformamide': ('N,N-Dimethylformamide', 'DMF'),
    'dimethylsulfoxide': ('Dimethyl sulfoxide', 'Dimethyl sulfoxide'),
    'diphenylether': ('Diphenyl ether', None),
    'dodecane': ('Dodecane', 'Dodecane'),
    'ethanol': ('Ethanol', 'Ethanol'),
    'ethylacetate': ('Ethyl acetate', 'Ethyl acetate'),
    'glycol': ('Ethylene glycol', 'Ethylene glycol'),
    'h2o': ('Water', 'Water'),
    'hexane': ('Hexane', 'n-Hexane'),
    'isopropylamine': ('Isopropylamine', None),
    'methanol': ('Methanol', 'Methanol'),
    'methylacetate': ('Methyl acetate', 'Methyl acetate'),
    'n-heptane': ('Heptane', 'n-Heptane'),
    'propanol': ('1-Propanol', '1-Propanol'),
    'propanone': ('Acetone', 'Acetone'),
    'propyleneglycol': ('Propylene glycol', '1,2-Propanediol'),
    'tert-butanol': ('tert-Butanol', 'tert-Butanol'),
    'thf': ('Tetrahydrofuran (THF)', 'THF'),
    'thp': ('Tetrahydropyran', None),
    'toluene': ('Toluene', 'Toluene'),
    'triethylamine': ('Triethylamine', 'Triethylamine'),
}


def normalize_solvent_name(solvent_name: str, target_database: str = "property") -> Optional[str]:
    """
    Normalize a solvent name from the solubility database to match property or GSK databases.

    Args:
        solvent_name: The solvent name from common_solvents_database
        target_database: "property" for solvent_data, "gsk" for gsk_dataset

    Returns:
        The normalized name for the target database, or None if no mapping exists
    """
    name_lower = solvent_name.strip().lower()

    if name_lower in SOLVENT_NAME_MAP:
        prop_name, gsk_name = SOLVENT_NAME_MAP[name_lower]
        if target_database == "property":
            return prop_name
        elif target_database == "gsk":
            return gsk_name

    # If not in static map, try to return as-is (might work for some names)
    return solvent_name


def get_cross_database_properties(solvent_name: str, conn) -> Dict[str, Any]:
    """
    Get properties for a solvent from solubility DB by looking up in property and GSK databases.

    Returns dict with: bp, logp, energy, cp, g_score, gsk_class (or None for missing)
    """
    props = {
        'bp': None, 'logp': None, 'energy': None, 'cp': None,
        'g_score': None, 'gsk_class': None
    }

    # Get property database name
    prop_name = normalize_solvent_name(solvent_name, "property")
    if prop_name:
        try:
            query = f"""
            SELECT bp__oc_, logp, energy__j_g_, cp__j_g_k_
            FROM solvent_data
            WHERE LOWER(solvent_name) = LOWER('{prop_name}')
            OR LOWER(solvent_name) LIKE '%{prop_name.lower()}%'
            LIMIT 1
            """
            result = conn.execute(query).fetchdf()
            if len(result) > 0:
                row = result.iloc[0]
                props['bp'] = row.get('bp__oc_')
                props['logp'] = row.get('logp')
                props['energy'] = row.get('energy__j_g_')
                props['cp'] = row.get('cp__j_g_k_')
        except Exception as e:
            logger.debug(f"Property lookup failed for {solvent_name}: {e}")

    # Get GSK database name
    gsk_name = normalize_solvent_name(solvent_name, "gsk")
    if gsk_name:
        try:
            query = f"""
            SELECT g_score, classification
            FROM gsk_dataset
            WHERE LOWER(solvent_common_name) = LOWER('{gsk_name}')
            OR LOWER(solvent_common_name) LIKE '%{gsk_name.lower()}%'
            LIMIT 1
            """
            result = conn.execute(query).fetchdf()
            if len(result) > 0:
                row = result.iloc[0]
                props['g_score'] = row.get('g_score')
                props['gsk_class'] = row.get('classification')
        except Exception as e:
            logger.debug(f"GSK lookup failed for {solvent_name}: {e}")

    return props


# ============================================================
# Data Classes for Structured Results
# ============================================================

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    verified_row_count: int = 0

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_valid = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)
        self.data_quality_score *= 0.9


@dataclass
class SeparationResult:
    """Result of polymer separation analysis"""
    is_feasible: bool
    conditions: Dict[str, Any] = field(default_factory=dict)
    selectivity: float = 0.0
    confidence: float = 0.0
    alternative_conditions: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ThresholdSearchResult:
    """Result of adaptive threshold search"""
    found: bool
    threshold_used: float
    results: List[Dict] = field(default_factory=list)
    thresholds_tried: List[float] = field(default_factory=list)
    search_path: str = ""


# ============================================================
# Data Validator Class (with caching)
# ============================================================

class DataValidator:
    """Extensive data validation and verification with caching."""

    def __init__(self, db_connection):
        self.conn = db_connection
        self._schema_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 60  # seconds

    def _get_cached_schema(self, table_name: str):
        """Get cached schema or fetch if expired."""
        now = time.time()
        if (table_name in self._schema_cache and 
            now - self._cache_timestamps.get(table_name, 0) < self._cache_ttl):
            return self._schema_cache[table_name]
        
        try:
            schema_df = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            self._schema_cache[table_name] = schema_df
            self._cache_timestamps[table_name] = now
            return schema_df
        except Exception:
            return None

    def clear_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()
        self._cache_timestamps.clear()

    def verify_table_exists(self, table_name: str) -> ValidationResult:
        """Verify table exists and has data"""
        result = ValidationResult(is_valid=True)
        try:
            tables = self.conn.execute("SHOW TABLES").fetchdf()
            if table_name not in tables['name'].values:
                result.add_issue(f"Table '{table_name}' does not exist")
                result.add_warning(f"Available tables: {list(tables['name'].values)}")
                return result

            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            result.verified_row_count = count
            if count == 0:
                result.add_issue(f"Table '{table_name}' is empty")
        except Exception as e:
            result.add_issue(f"Error verifying table: {e}")
        return result

    def verify_column_exists(self, table_name: str, column_name: str) -> ValidationResult:
        """Verify column exists in table (uses cache)."""
        result = ValidationResult(is_valid=True)
        try:
            schema = self._get_cached_schema(table_name)
            if schema is None:
                result.add_issue(f"Could not get schema for '{table_name}'")
                return result
            
            if column_name not in schema['column_name'].values:
                result.add_issue(f"Column '{column_name}' not found in '{table_name}'")
                similar = [c for c in schema['column_name'] if column_name.lower() in c.lower()]
                if similar:
                    result.add_warning(f"Similar columns found: {similar}")
                else:
                    result.add_warning(f"Available columns: {list(schema['column_name'].values)[:10]}...")
        except Exception as e:
            result.add_issue(f"Error verifying column: {e}")
        return result

    def verify_value_exists(self, table_name: str, column_name: str, value: str) -> ValidationResult:
        """Verify a specific value exists in a column"""
        result = ValidationResult(is_valid=True)
        try:
            # Escape single quotes to prevent SQL injection
            safe_value = str(value).replace("'", "''")
            query = f"SELECT COUNT(*) FROM {table_name} WHERE LOWER(CAST({column_name} AS VARCHAR)) = LOWER('{safe_value}')"
            count = self.conn.execute(query).fetchone()[0]
            if count == 0:
                result.add_issue(f"Value '{value}' not found in {table_name}.{column_name}")
                available = self.conn.execute(
                    f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT 20"
                ).fetchdf()[column_name].tolist()
                result.add_warning(f"Available values (sample): {available}")
            result.verified_row_count = count
        except Exception as e:
            result.add_issue(f"Error verifying value: {e}")
        return result

    def cross_validate_query_result(self, query: str, expected_columns: List[str],
                                     min_rows: int = 1) -> ValidationResult:
        """Cross-validate query results"""
        result = ValidationResult(is_valid=True)
        try:
            df = self.conn.execute(query).fetchdf()
            result.verified_row_count = len(df)

            if len(df) < min_rows:
                result.add_issue(f"Query returned {len(df)} rows, expected at least {min_rows}")

            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                result.add_issue(f"Missing expected columns: {missing_cols}")

            null_counts = df.isnull().sum()
            high_null_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()
            if high_null_cols:
                result.add_warning(f"High null rate in columns: {high_null_cols}")

            if len(df) > 0 and df.duplicated().sum() > len(df) * 0.1:
                result.add_warning("High duplicate rate in results")

            # Clean up
            del df
            gc.collect()

        except Exception as e:
            result.add_issue(f"Query validation failed: {e}")
        return result

    def verify_numeric_range(self, table_name: str, column_name: str,
                            min_val: Optional[float] = None,
                            max_val: Optional[float] = None) -> ValidationResult:
        """Verify numeric values are in expected range"""
        result = ValidationResult(is_valid=True)
        try:
            stats_query = f"""
            SELECT MIN({column_name}) as min_val,
                   MAX({column_name}) as max_val,
                   AVG({column_name}) as avg_val,
                   STDDEV({column_name}) as std_val
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            """
            stats_df = self.conn.execute(stats_query).fetchdf()

            actual_min = stats_df['min_val'].iloc[0]
            actual_max = stats_df['max_val'].iloc[0]

            if min_val is not None and actual_min < min_val:
                result.add_warning(f"Values below expected minimum: {actual_min} < {min_val}")
            if max_val is not None and actual_max > max_val:
                result.add_warning(f"Values above expected maximum: {actual_max} > {max_val}")

        except Exception as e:
            result.add_issue(f"Range verification failed: {e}")
        return result


# ============================================================
# Adaptive Analysis Engine
# ============================================================

class AdaptiveAnalyzer:
    """Intelligent adaptive analysis with threshold searching and temperature exploration"""

    # Thresholds in PERCENTAGE form (0-100 scale) to match database solubility values
    SELECTIVITY_THRESHOLDS = [50, 30, 20, 15, 10, 5, 2, 1, 0.5, 0.1]
    SOLUBILITY_THRESHOLDS = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
    TEMPERATURE_STEPS = [25, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150]

    def __init__(self, db_connection, validator: DataValidator):
        self.conn = db_connection
        self.validator = validator

    def find_threshold_with_results(self,
                                    query_func,
                                    thresholds: List[float],
                                    min_results: int = 1,
                                    prefer_stringent: bool = True) -> ThresholdSearchResult:
        """Iteratively search thresholds from stringent to lenient until results found."""
        result = ThresholdSearchResult(found=False, threshold_used=0, thresholds_tried=[])

        search_order = thresholds if prefer_stringent else thresholds[::-1]

        for threshold in search_order:
            result.thresholds_tried.append(threshold)
            try:
                results = query_func(threshold)
                if len(results) >= min_results:
                    result.found = True
                    result.threshold_used = threshold
                    result.results = results
                    result.search_path = f"Tried {len(result.thresholds_tried)} thresholds, found results at {threshold}"
                    return result
            except Exception as e:
                logger.warning(f"Threshold {threshold} failed: {e}")
                continue

        result.search_path = f"Exhausted all {len(thresholds)} thresholds without finding {min_results}+ results"
        return result

    def explore_temperature_range(self,
                                  table_name: str,
                                  polymer_column: str,
                                  solvent_column: str,
                                  temperature_column: str,
                                  solubility_column: str,
                                  target_polymer: str,
                                  comparison_polymers: List[str],
                                  start_temp: float = 25,
                                  min_selectivity: float = 10.0) -> Dict[str, Any]:
        """Explore temperatures to find optimal separation conditions.

        Note: min_selectivity is in percentage points (0-100 scale).
        """
        results = {
            'optimal_conditions': None,
            'all_conditions': [],
            'temperatures_explored': [],
            'recommendation': ''
        }
        
        # Ensure comparison_polymers is a list
        if isinstance(comparison_polymers, str):
            comparison_polymers = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
        elif not isinstance(comparison_polymers, list):
            comparison_polymers = list(comparison_polymers) if comparison_polymers else []
        
        if not comparison_polymers:
            results['recommendation'] = "No comparison polymers provided"
            return results

        temp_query = f"""
        SELECT DISTINCT {temperature_column}
        FROM {table_name}
        WHERE {temperature_column} >= {start_temp}
        ORDER BY {temperature_column}
        """
        try:
            available_temps = self.conn.execute(temp_query).fetchdf()[temperature_column].tolist()
        except:
            available_temps = self.TEMPERATURE_STEPS

        for temp in available_temps:
            if temp < start_temp:
                continue

            results['temperatures_explored'].append(temp)

            selectivity_data = self._calculate_selectivity_at_temp(
                table_name, polymer_column, solvent_column,
                temperature_column, solubility_column,
                target_polymer, comparison_polymers, temp
            )

            if selectivity_data['best_selectivity'] >= min_selectivity:
                condition = {
                    'temperature': temp,
                    'selectivity': selectivity_data['best_selectivity'],
                    'best_solvent': selectivity_data['best_solvent'],
                    'target_solubility': selectivity_data['target_solubility'],
                    'max_other_solubility': selectivity_data['max_other_solubility']
                }
                results['all_conditions'].append(condition)

                if results['optimal_conditions'] is None:
                    results['optimal_conditions'] = condition
                elif condition['selectivity'] > results['optimal_conditions']['selectivity']:
                    results['optimal_conditions'] = condition

        if results['optimal_conditions']:
            opt = results['optimal_conditions']
            results['recommendation'] = (
                f"Optimal separation at {opt['temperature']}°C using {opt['best_solvent']} "
                f"(selectivity: {opt['selectivity']:.3f})"
            )
        else:
            results['recommendation'] = (
                f"No separation found with selectivity >= {min_selectivity}. "
                f"Consider lower selectivity threshold or different solvents."
            )

        return results

    def _calculate_selectivity_at_temp(self,
                                       table_name: str,
                                       polymer_column: str,
                                       solvent_column: str,
                                       temperature_column: str,
                                       solubility_column: str,
                                       target_polymer: str,
                                       comparison_polymers: List[str],
                                       temperature: float,
                                       temp_tolerance: float = 5.0) -> Dict[str, Any]:
        """Calculate selectivity at a specific temperature"""
        
        # Ensure comparison_polymers is a list
        if isinstance(comparison_polymers, str):
            comparison_polymers = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
        elif not isinstance(comparison_polymers, list):
            comparison_polymers = list(comparison_polymers) if comparison_polymers else []
        
        if not comparison_polymers:
            return {'best_selectivity': 0, 'best_solvent': None,
                    'target_solubility': 0, 'max_other_solubility': 0}

        all_polymers = [target_polymer] + comparison_polymers
        polymer_filter = "', '".join(all_polymers)

        query = f"""
        SELECT {solvent_column}, {polymer_column}, AVG({solubility_column}) as avg_sol
        FROM {table_name}
        WHERE {polymer_column} IN ('{polymer_filter}')
        AND {temperature_column} BETWEEN {temperature - temp_tolerance} AND {temperature + temp_tolerance}
        GROUP BY {solvent_column}, {polymer_column}
        """

        try:
            df = self.conn.execute(query).fetchdf()
        except Exception as e:
            logger.error(f"Selectivity query failed: {e}")
            return {'best_selectivity': 0, 'best_solvent': None,
                    'target_solubility': 0, 'max_other_solubility': 0}

        if len(df) == 0:
            return {'best_selectivity': 0, 'best_solvent': None,
                    'target_solubility': 0, 'max_other_solubility': 0}

        best_result = {'best_selectivity': 0, 'best_solvent': None,
                      'target_solubility': 0, 'max_other_solubility': 0}

        for solvent in df[solvent_column].unique():
            solvent_data = df[df[solvent_column] == solvent]

            target_data = solvent_data[solvent_data[polymer_column] == target_polymer]
            if len(target_data) == 0:
                continue
            target_sol = target_data['avg_sol'].values[0]

            other_data = solvent_data[solvent_data[polymer_column].isin(comparison_polymers)]
            if len(other_data) == 0:
                selectivity = target_sol
                max_other = 0
            else:
                max_other = other_data['avg_sol'].max()
                selectivity = target_sol - max_other

            if selectivity > best_result['best_selectivity']:
                best_result = {
                    'best_selectivity': selectivity,
                    'best_solvent': solvent,
                    'target_solubility': target_sol,
                    'max_other_solubility': max_other
                }

        # Clean up
        del df
        gc.collect()

        return best_result

    def adaptive_separation_analysis(self,
                                     table_name: str,
                                     polymer_column: str,
                                     solvent_column: str,
                                     temperature_column: str,
                                     solubility_column: str,
                                     target_polymer: str,
                                     comparison_polymers: Optional[List[str]] = None,
                                     initial_temp: float = 25,
                                     initial_selectivity: float = 30.0) -> SeparationResult:
        """Comprehensive adaptive separation analysis.

        Note: Selectivity is in percentage points (0-100 scale).
        """
        result = SeparationResult(is_feasible=False)

        # Ensure comparison_polymers is a list
        if comparison_polymers is None:
            try:
                all_polymers_query = f"SELECT DISTINCT {polymer_column} FROM {table_name}"
                all_polymers_df = self.conn.execute(all_polymers_query).fetchdf()
                if len(all_polymers_df) > 0 and polymer_column in all_polymers_df.columns:
                    comparison_polymers = [p for p in all_polymers_df[polymer_column].tolist() if p != target_polymer]
                else:
                    comparison_polymers = []
            except Exception as e:
                logger.error(f"Could not get polymers: {e}")
                comparison_polymers = []
        elif isinstance(comparison_polymers, str):
            comparison_polymers = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
        
        if not comparison_polymers:
            result.recommendations.append(f"No comparison polymers found for analysis")
            return result

        logger.info(f"Analyzing separation of {target_polymer} from {comparison_polymers}")

        for selectivity in self.SELECTIVITY_THRESHOLDS:
            if selectivity > initial_selectivity:
                continue

            temp_result = self.explore_temperature_range(
                table_name, polymer_column, solvent_column,
                temperature_column, solubility_column,
                target_polymer, comparison_polymers,
                start_temp=initial_temp, min_selectivity=selectivity
            )

            if temp_result['optimal_conditions']:
                opt = temp_result['optimal_conditions']
                result.is_feasible = True
                result.conditions = opt
                result.selectivity = opt['selectivity']
                result.confidence = self._calculate_confidence(
                    selectivity, opt['temperature'], initial_temp
                )
                result.alternative_conditions = temp_result['all_conditions'][1:5]
                result.recommendations.append(temp_result['recommendation'])
                return result

        result.recommendations.append(
            f"No selective separation found for {target_polymer} vs {comparison_polymers}"
        )
        result.recommendations.append(
            "Consider: (1) Different solvents, (2) Higher temperatures, "
            "(3) Lower selectivity requirements, (4) Sequential extraction"
        )

        return result

    def _calculate_confidence(self, selectivity_threshold: float,
                             actual_temp: float, requested_temp: float) -> float:
        """Calculate confidence score based on how close to ideal conditions.

        Note: selectivity_threshold is in percentage points (0-100 scale).
        """
        confidence = 1.0

        # Threshold penalty based on percentage-scale thresholds
        threshold_penalty = {50: 0, 30: 0.05, 20: 0.1, 15: 0.15,
                           10: 0.2, 5: 0.3, 2: 0.4, 1: 0.5, 0.5: 0.55, 0.1: 0.6}
        confidence -= threshold_penalty.get(selectivity_threshold, 0.3)

        temp_deviation = abs(actual_temp - requested_temp)
        confidence -= min(temp_deviation / 100, 0.3)

        return max(0.1, confidence)


# ============================================================
# Enhanced SQLDatabase
# ============================================================

class SQLDatabase:
    """Memory-efficient SQL database wrapper."""
    
    def __init__(self, db_path: str = SQL_DB_PATH):
        self.db_path = db_path
        self._conn = None
        self.table_schemas: Dict[str, Dict] = {}
        self._table_info_cache: Optional[str] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 300
        
        self.validator = None
        self.analyzer = None
        
        self._initialize_connection()
        logger.info(f"DuckDB initialized at {db_path}")

    def _initialize_connection(self):
        """Initialize database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(database=self.db_path)
            self.validator = DataValidator(self._conn)
            self.analyzer = AdaptiveAnalyzer(self._conn, self.validator)

    @property
    def conn(self):
        """Get database connection."""
        if self._conn is None:
            self._initialize_connection()
        return self._conn

    def invalidate_cache(self):
        """Invalidate the table info cache."""
        self._table_info_cache = None
        self._cache_timestamp = None
        if self.validator:
            self.validator.clear_cache()

    def load_csv_files(self, csv_dir: str = DATA_DIR):
        """Load CSV files with memory-efficient processing."""
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        if not csv_files:
            logger.warning("No CSV files found")
            return

        logger.info(f"\nLoading {len(csv_files)} CSV file(s)...")
        for csv_path in csv_files:
            try:
                table_name = Path(csv_path).stem.lower()
                table_name = re.sub(r'[^a-z0-9_]', '_', table_name)

                # Read CSV
                df = pd.read_csv(csv_path, encoding='utf-8')
                df.columns = [re.sub(r'[^a-z0-9_]', '_', col.lower().strip()) for col in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]

                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.conn.register(f'{table_name}_temp', df)
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp")
                self.conn.unregister(f'{table_name}_temp')

                schema_query = f"DESCRIBE {table_name}"
                schema_df = self.conn.execute(schema_query).fetchdf()
                row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                self.table_schemas[table_name] = {
                    "file_path": csv_path,
                    "columns": list(schema_df["column_name"]),
                    "types": dict(zip(schema_df["column_name"], schema_df["column_type"])),
                    "row_count": row_count,
                }

                logger.info(f"  ✅ Loaded '{table_name}': {row_count} rows, {len(schema_df)} columns")

                # Create indexes for performance optimization
                try:
                    if table_name == "common_solvents_database":
                        logger.info(f"  Creating indexes for {table_name}...")

                        # Single-column indexes for frequent filters
                        if "polymer" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_polymer ON {table_name}("polymer")')
                        if "solvent" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_solvent ON {table_name}("solvent")')
                        if "temperature" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_temperature ON {table_name}("temperature")')
                        if "solubility" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_solubility ON {table_name}("solubility")')

                        # Composite indexes for common query patterns
                        if "polymer" in self.table_schemas[table_name]["columns"] and "solvent" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_poly_solv ON {table_name}("polymer", "solvent")')
                        if "polymer" in self.table_schemas[table_name]["columns"] and "temperature" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_poly_temp ON {table_name}("polymer", "temperature")')

                        # Collect statistics for query optimizer
                        self.conn.execute(f'ANALYZE {table_name}')

                        logger.info(f"  ✅ Created 6 indexes for {table_name}")

                    elif table_name == "solvent_data":
                        logger.info(f"  Creating indexes for {table_name}...")

                        if "cosmobase_name" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}("cosmobase_name")')
                        if "logp" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_logp ON {table_name}("logp")')
                        if "bp" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_bp ON {table_name}("bp")')

                        # Collect statistics for query optimizer
                        self.conn.execute(f'ANALYZE {table_name}')

                        logger.info(f"  ✅ Created 3 indexes for {table_name}")

                    elif table_name == "gsk_dataset":
                        logger.info(f"  Creating indexes for {table_name}...")

                        # Index on solvent name for lookups
                        if "solvent_common_name" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_solvent_name ON {table_name}("solvent_common_name")')

                        # Index on G-score for filtering by safety
                        if "g_score" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_gscore ON {table_name}("g_score")')

                        # Index on classification (solvent family)
                        if "classification" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_family ON {table_name}("classification")')

                        # Composite index for family + G-score queries
                        if "classification" in self.table_schemas[table_name]["columns"] and "g_score" in self.table_schemas[table_name]["columns"]:
                            self.conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_family_gscore ON {table_name}("classification", "g_score")')

                        # Collect statistics for query optimizer
                        self.conn.execute(f'ANALYZE {table_name}')

                        logger.info(f"  ✅ Created 4 indexes for {table_name}")

                except Exception as idx_error:
                    logger.warning(f"  ⚠️ Failed to create indexes for {table_name}: {idx_error}")

                # Clean up
                del df
                gc.collect()

            except Exception as e:
                logger.error(f"  ❌ Error loading {csv_path}: {e}")

        self.invalidate_cache()
        logger.info("✅ CSV loading complete\n")

    def get_table_info(self) -> str:
        """Get table info with caching."""
        now = time.time()
        if (self._table_info_cache and self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_ttl):
            return self._table_info_cache

        if not self.table_schemas:
            return "No tables available."

        info_parts = ["Available Tables:\n"]
        for table_name, schema in self.table_schemas.items():
            info_parts.append(f"\n**Table: {table_name}** ({schema['row_count']} rows)")
            info_parts.append("Columns:")
            for col, dtype in schema['types'].items():
                try:
                    if 'INT' in str(dtype).upper() or 'DOUBLE' in str(dtype).upper() or 'FLOAT' in str(dtype).upper():
                        stats = self.conn.execute(
                            f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}"
                        ).fetchone()
                        info_parts.append(f"  - {col}: {dtype} [min={stats[0]:.4f}, max={stats[1]:.4f}, avg={stats[2]:.4f}]")
                    else:
                        unique_count = self.conn.execute(
                            f"SELECT COUNT(DISTINCT {col}) FROM {table_name}"
                        ).fetchone()[0]
                        info_parts.append(f"  - {col}: {dtype} [{unique_count} unique values]")
                except:
                    info_parts.append(f"  - {col}: {dtype}")

        self._table_info_cache = "\n".join(info_parts)
        self._cache_timestamp = now
        return self._table_info_cache

    def execute_query(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Execute query with memory-efficient result handling."""
        try:
            query_lower = query.lower().strip()
            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create', 'truncate']
            if any(keyword in query_lower.split() for keyword in dangerous_keywords):
                return {"success": False, "error": "Unsafe operation detected", "query": query}

            if 'limit' not in query_lower and not query_lower.strip().endswith(';'):
                query = f"{query.rstrip(';')} LIMIT {limit}"

            result_df = self.conn.execute(query).fetchdf()

            # Create preview efficiently
            preview = result_df.head(10).to_markdown(index=False) if len(result_df) > 0 else "No data"

            return {
                "success": True,
                "query": query,
                "rows": len(result_df),
                "columns": list(result_df.columns),
                "data": result_df.to_dict('records'),
                "dataframe": result_df,
                "preview": preview,
                "dtypes": {str(k): str(v) for k, v in result_df.dtypes.to_dict().items()}
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def get_sample_data(self, table_name: str, n: int = 3) -> str:
        try:
            query = f"SELECT * FROM {table_name} LIMIT {n}"
            df = self.conn.execute(query).fetchdf()
            result = df.to_markdown(index=False)
            del df
            return result
        except Exception as e:
            return f"Error: {e}"

    def close(self):
        """Close database connection and cleanup."""
        if self._conn is not None:
            try:
                self._conn.close()
            except:
                pass
            finally:
                self._conn = None
                self.table_schemas.clear()
                self.invalidate_cache()
                gc.collect()


# Initialize SQL database
sql_db = SQLDatabase()

# Initialize async DB wrapper (lazy initialization)
_async_db = None

def get_async_db():
    """
    Get or create async DB wrapper.

    This wrapper provides lock-protected async access to the DuckDB connection,
    allowing concurrent execution while preventing race conditions.
    """
    global _async_db
    if _async_db is None:
        _async_db = AsyncDuckDBWrapper(sql_db.conn)
    return _async_db

# ============================================================
# Auto-load Required CSV Files
# ============================================================

REQUIRED_CSV_FILES = [
    "COMMON-SOLVENTS-DATABASE.csv",  # Main solubility data
    "Solvent_Data.csv",               # Solvent properties (BP, LogP, Energy)
]

def auto_load_csv_files():
    """
    Auto-load required CSV files at startup.
    Searches in DATA_DIR and common locations.
    """
    loaded = []
    not_found = []
    
    # Search paths
    search_paths = [
        DATA_DIR,
        ".",
        "./data",
        os.path.expanduser("~"),
        "/content",  # Google Colab
        "/content/drive/MyDrive",  # Google Drive in Colab
    ]
    
    for csv_file in REQUIRED_CSV_FILES:
        found = False
        
        # Check if already in DATA_DIR
        target_path = os.path.join(DATA_DIR, csv_file)
        if os.path.exists(target_path):
            found = True
            loaded.append(csv_file)
            logger.info(f"✅ Found {csv_file} in {DATA_DIR}")
            continue
        
        # Search other locations
        for search_dir in search_paths:
            source_path = os.path.join(search_dir, csv_file)
            if os.path.exists(source_path) and source_path != target_path:
                try:
                    # Copy to DATA_DIR
                    import shutil
                    shutil.copy(source_path, target_path)
                    loaded.append(csv_file)
                    logger.info(f"✅ Copied {csv_file} from {search_dir} to {DATA_DIR}")
                    found = True
                    break
                except Exception as e:
                    logger.warning(f"Could not copy {csv_file}: {e}")
        
        if not found:
            not_found.append(csv_file)
            logger.warning(f"⚠️ {csv_file} not found - upload it via the Data Management tab")
    
    # Load all CSVs into database
    sql_db.load_csv_files()
    
    return loaded, not_found

# Run auto-loading
_loaded_files, _missing_files = auto_load_csv_files()

if _missing_files:
    logger.warning(f"Missing CSV files: {_missing_files}")
    logger.warning("Upload these files via the Data Management tab for full functionality")

logger.info(f"📊 Loaded {len(sql_db.table_schemas)} tables: {list(sql_db.table_schemas.keys())}")


# ============================================================
# Helper Functions (PATCHED)
# ============================================================

def save_plot(fig, plot_name: str, plot_type: str = "matplotlib") -> str:
    """Save plot to file with memory cleanup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_name}_{timestamp}.png"
    filepath = os.path.join(PLOTS_DIR, filename)

    try:
        if plot_type == "matplotlib":
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plt.close('all')
        elif plot_type == "plotly":
            fig.write_image(filepath, width=1200, height=800)
        
        gc.collect()
        logger.info(f"Plot saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        try:
            plt.close('all')
        except:
            pass
        return f"Error: {e}"


def get_plot_url(filepath: str) -> str:
    """Convert filepath to displayable format"""
    return f"📊 Plot saved: `{filepath}`"


# Solvent name normalization mapping (common names → database names)
SOLVENT_NAME_MAPPING = {
    # Xylene variants - expand to both isomers
    'xylene': ['1,2-dimethylbenzene', '1,4-dimethylbenzene'],
    'xylenes': ['1,2-dimethylbenzene', '1,4-dimethylbenzene'],
    'o-xylene': ['1,2-dimethylbenzene'],
    'p-xylene': ['1,4-dimethylbenzene'],
    'm-xylene': ['1,4-dimethylbenzene'],  # closest available
    'ortho-xylene': ['1,2-dimethylbenzene'],
    'para-xylene': ['1,4-dimethylbenzene'],

    # Alkanes - common names to database names
    'heptane': ['n-heptane'],
    'n-heptane': ['n-heptane'],
    'hexane': ['hexane'],
    'n-hexane': ['hexane'],
    'pentane': ['pentane'],
    'n-pentane': ['pentane'],
    'octane': ['octane'],
    'n-octane': ['octane'],

    # Polar aprotic solvents - common abbreviations
    'dmso': ['dimethylsulfoxide'],
    'dimethyl sulfoxide': ['dimethylsulfoxide'],
    'dmf': ['dimethylformamide'],
    'dimethyl formamide': ['dimethylformamide'],
    'nmp': ['n-methylpyrrolidone'],
    'n-methyl-2-pyrrolidone': ['n-methylpyrrolidone'],

    # Ketones
    'acetone': ['propanone'],
    '2-propanone': ['propanone'],
    'mek': ['butanone'],
    'methyl ethyl ketone': ['butanone'],

    # Alcohols
    'ipa': ['2-propanol'],
    'isopropanol': ['2-propanol'],
    'isopropyl alcohol': ['2-propanol'],
    'n-propanol': ['propanol'],
    '1-propanol': ['propanol'],
    'meoh': ['methanol'],
    'etoh': ['ethanol'],

    # Ethers
    'tetrahydrofuran': ['thf'],
    'tetrahydropyran': ['thp'],
    'dihydropyran': ['2,3-dihydropyran'],

    # Halogenated
    'dcm': ['ch2cl2'],
    'dichloromethane': ['ch2cl2'],
    'methylene chloride': ['ch2cl2'],
    'chloroform': ['chcl3'],
    'trichloromethane': ['chcl3'],

    # Others
    'dmf': ['dimethylformamide'],
    'dmso': ['dimethylsulfoxide'],
    'ethyl acetate': ['ethylacetate'],
    'methyl acetate': ['methylacetate'],
    'water': ['h2o'],
    'n-hexane': ['hexane'],
    'n-heptane': ['n-heptane'],
    'ethylene glycol': ['glycol'],
    'propylene glycol': ['propyleneglycol'],
}

# Patterns to reconstruct solvent names that were incorrectly split by commas
# Maps fragment patterns to the correct full name
SOLVENT_FRAGMENT_RECONSTRUCTION = {
    # Pattern: (preceding_fragment, current_fragment) -> full_name
    # When we see "2" followed by "3-dihydropyran", reconstruct to "2,3-dihydropyran"
    ('2', '3-dihydropyran'): '2,3-dihydropyran',
    ('1', '2-dimethylbenzene'): '1,2-dimethylbenzene',
    ('1', '4-dimethylbenzene'): '1,4-dimethylbenzene',
    ('1', '3-dimethylbenzene'): '1,3-dimethylbenzene',
    ('1', '2-dichloroethane'): '1,2-dichloroethane',
    ('1', '1-dichloroethane'): '1,1-dichloroethane',
    ('1', '2-dichlorobenzene'): '1,2-dichlorobenzene',
    ('1', '4-dichlorobenzene'): '1,4-dichlorobenzene',
    ('1', '2-ethanediol'): '1,2-ethanediol',
    ('1', '3-propanediol'): '1,3-propanediol',
    ('1', '4-dioxane'): '1,4-dioxane',
    ('2', '2-dimethylbutane'): '2,2-dimethylbutane',
    ('2', '3-butanediol'): '2,3-butanediol',
    ('2', '4-pentanedione'): '2,4-pentanedione',
}


def normalize_solvent_names(solvents: List[str]) -> List[str]:
    """
    Normalize solvent names to match database entries.
    Expands common names like 'xylene' to actual database names.
    Also converts to lowercase since database uses lowercase names.

    Handles reconstruction of solvent names that were incorrectly split by commas,
    e.g., "2,3-dihydropyran" becoming ["2", "3-dihydropyran"].

    Args:
        solvents: List of solvent names (may include common names)

    Returns:
        List of normalized solvent names matching database entries (lowercase)
    """
    # First, reconstruct any fragmented solvent names
    reconstructed = []
    i = 0
    while i < len(solvents):
        solvent = solvents[i].strip().lower()

        # Check if this could be the start of a fragmented name
        if i + 1 < len(solvents):
            next_solvent = solvents[i + 1].strip().lower()
            fragment_key = (solvent, next_solvent)

            if fragment_key in SOLVENT_FRAGMENT_RECONSTRUCTION:
                # Reconstruct the full name
                full_name = SOLVENT_FRAGMENT_RECONSTRUCTION[fragment_key]
                reconstructed.append(full_name)
                i += 2  # Skip both fragments
                continue

        reconstructed.append(solvent)
        i += 1

    # Now apply the standard normalization mapping
    normalized = []
    for solvent in reconstructed:
        solvent_lower = solvent.strip().lower()
        if solvent_lower in SOLVENT_NAME_MAPPING:
            # Expand to mapped name(s) - already lowercase in mapping
            normalized.extend(SOLVENT_NAME_MAPPING[solvent_lower])
        else:
            # Convert to lowercase to match database
            normalized.append(solvent_lower)
    return normalized


def verify_inputs(table_name: str, columns: Dict[str, str],
                  values: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, str]:
    """Comprehensive input verification."""
    issues = []
    warnings = []

    # Verify table
    table_val = sql_db.validator.verify_table_exists(table_name)
    if not table_val.is_valid:
        return False, f"❌ Table '{table_name}' not found. {table_val.warnings}"

    # Get schema once
    try:
        schema = sql_db.conn.execute(f"DESCRIBE {table_name}").fetchdf()
        available_cols = set(schema['column_name'].values)
    except Exception as e:
        return False, f"❌ Could not get schema: {e}"

    # Verify all columns
    for purpose, col_name in columns.items():
        if col_name not in available_cols:
            issues.append(f"Column '{col_name}' ({purpose}) not found")
            similar = [c for c in available_cols if col_name.lower() in c.lower()]
            if similar:
                warnings.append(f"Did you mean: {similar}?")

    if issues:
        msg = "❌ Verification failed:\n- " + "\n- ".join(issues)
        if warnings:
            msg += "\n\n💡 " + "\n💡 ".join(warnings)
        return False, msg

    # Verify values if provided
    if values:
        for col_name, expected_vals in values.items():
            if col_name not in available_cols:
                continue
            for val in expected_vals:
                val_result = sql_db.validator.verify_value_exists(table_name, col_name, val)
                if not val_result.is_valid:
                    issues.append(f"Value '{val}' not found in {col_name}")
                    if val_result.warnings:
                        warnings.extend(val_result.warnings[:1])

    if issues:
        msg = "❌ Value verification failed:\n- " + "\n- ".join(issues)
        if warnings:
            msg += "\n\n💡 Available: " + str(warnings[0]) if warnings else ""
        return False, msg

    return True, "✅ All inputs verified"


# ============================================================
# Core Database Tools (PATCHED with @safe_tool_wrapper)
# ============================================================

@tool
@safe_tool_wrapper
def list_tables() -> str:
    """List all available SQL tables with schemas, row counts, and data quality info."""
    return sql_db.get_table_info()


@tool
@safe_tool_wrapper
def describe_table(table_name: str) -> str:
    """Get detailed information about a specific table including sample data and statistics."""
    if table_name not in sql_db.table_schemas:
        available = list(sql_db.table_schemas.keys())
        return f"Error: Table '{table_name}' not found. Available tables: {available}"

    schema = sql_db.table_schemas[table_name]
    output = [f"**Table: {table_name}**\n", f"Rows: {schema['row_count']}\n", "Columns:"]

    for col, dtype in schema['types'].items():
        try:
            if 'INT' in str(dtype).upper() or 'DOUBLE' in str(dtype).upper() or 'FLOAT' in str(dtype).upper():
                stats = sql_db.conn.execute(
                    f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}"
                ).fetchone()
                output.append(f"  - {col}: {dtype} [min={stats[0]:.4f}, max={stats[1]:.4f}, avg={stats[2]:.4f}]")
            else:
                unique_count = sql_db.conn.execute(
                    f"SELECT COUNT(DISTINCT {col}) FROM {table_name}"
                ).fetchone()[0]
                output.append(f"  - {col}: {dtype} [{unique_count} unique values]")
        except:
            output.append(f"  - {col}: {dtype}")

    output.append(f"\n**Sample data (5 rows):**")
    output.append(sql_db.get_sample_data(table_name, 5))

    return "\n".join(output)


@tool
@safe_tool_wrapper
def check_column_values(table_name: str, column_name: str, limit: int = 50) -> str:
    """Check what values exist in a specific column with frequency counts."""
    is_valid, msg = verify_inputs(table_name, {"column": column_name})
    if not is_valid:
        return msg

    query = f"""
    SELECT {column_name}, COUNT(*) as count
    FROM {table_name}
    GROUP BY {column_name}
    ORDER BY count DESC
    LIMIT {limit}
    """
    result_df = sql_db.conn.execute(query).fetchdf()

    output = f"**Unique values in {table_name}.{column_name}:**\n\n"
    output += result_df.to_markdown(index=False)
    output += f"\n\nTotal unique values: {len(result_df)}"

    total_rows = sql_db.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    output += f"\nTotal rows in table: {total_rows}"

    del result_df
    return output


@tool
@safe_tool_wrapper
def query_database(sql_query: str, export_csv: bool = False) -> str:
    """Execute a SQL query with enhanced validation and error reporting.

    Args:
        sql_query: SQL query to execute
        export_csv: If True, creates a CSV export of the results (default: False)

    Returns:
        Query results as formatted text, with optional CSV export link
    """
    result = sql_db.execute_query(sql_query)

    if result["success"]:
        df = result["dataframe"]

        # Generate CSV export if requested
        export_id = None
        if export_csv and result['rows'] > 0:
            try:
                from export_manager import export_manager
                export_id = export_manager.create_export(
                    data=df.to_dict(orient="records"),
                    tool_name="query_database",
                    columns=df.columns.tolist()
                )
            except Exception as e:
                logger.error(f"Failed to create CSV export: {e}")

        # Format output
        output = f"**Query Results**\n\nQuery: `{result['query']}`\n\nRows returned: {result['rows']}\n\n"

        if export_id:
            output += f"📥 **CSV Export Available:** `/api/export/{export_id}`\n\n"

        if result['rows'] > 0:
            output += "**Data:**\n" + result["preview"]
            if result['rows'] > 10:
                output += f"\n\n_(Showing first 10 of {result['rows']} rows)_"
        else:
            output += "⚠️ No rows matched the query."
        return output
    else:
        return f"**Query Error**\n\nQuery: `{result['query']}`\n\nError: {result['error']}\n\n💡 Tip: Use check_column_values() to verify column names and values."


@tool
@safe_tool_wrapper
def verify_data_accuracy(table_name: str, filters: Optional[str] = None) -> str:
    """Verify data accuracy by checking actual row counts and sample data."""
    where_clause = f"WHERE {filters}" if filters else ""

    count_query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
    count = sql_db.conn.execute(count_query).fetchone()[0]

    sample_query = f"SELECT * FROM {table_name} {where_clause} LIMIT 5"
    sample_df = sql_db.conn.execute(sample_query).fetchdf()

    output = f"**Data Verification for {table_name}**\n\n"
    output += f"Filter: {filters or 'None'}\n"
    output += f"Total matching rows: {count}\n\n"

    if count > 0:
        output += "Sample data:\n"
        output += sample_df.to_markdown(index=False)
    else:
        output += "⚠️ **No data matches these criteria!**\n"
        output += "Please verify:\n"
        output += "1. Column names are correct\n"
        output += "2. Filter values exist in the data\n"
        output += "3. Data types match (e.g., strings need quotes)\n"

    del sample_df
    return output


@tool
@safe_tool_wrapper
def validate_and_query(
    table_name: str,
    required_columns: str,
    filter_column: Optional[str] = None,
    filter_values: Optional[str] = None,
    sql_query: Optional[str] = None
) -> str:
    """Validate inputs BEFORE executing a query. Use this to prevent hallucinations."""
    output = ["**Input Validation Report**\n"]
    all_valid = True

    columns = [c.strip() for c in required_columns.split(',')]

    table_val = sql_db.validator.verify_table_exists(table_name)
    if table_val.is_valid:
        output.append(f"✅ Table '{table_name}' exists ({table_val.verified_row_count} rows)")
    else:
        output.append(f"❌ Table issue: {table_val.issues}")
        all_valid = False

    for col in columns:
        col_val = sql_db.validator.verify_column_exists(table_name, col)
        if col_val.is_valid:
            output.append(f"✅ Column '{col}' exists")
        else:
            output.append(f"❌ Column '{col}': {col_val.issues}")
            if col_val.warnings:
                output.append(f"   💡 {col_val.warnings[0]}")
            all_valid = False

    if filter_column and filter_values:
        values = [v.strip() for v in filter_values.split(',')]
        for val in values:
            val_result = sql_db.validator.verify_value_exists(table_name, filter_column, val)
            if val_result.is_valid:
                output.append(f"✅ Value '{val}' found in {filter_column} ({val_result.verified_row_count} rows)")
            else:
                output.append(f"❌ Value '{val}' NOT found in {filter_column}")
                if val_result.warnings:
                    output.append(f"   💡 {val_result.warnings[0]}")
                all_valid = False

    if sql_query and all_valid:
        output.append("\n**Query Execution:**")
        result = sql_db.execute_query(sql_query)
        if result["success"]:
            output.append(f"✅ Query successful: {result['rows']} rows returned")
            if result['rows'] > 0:
                output.append("\n" + result["preview"])
        else:
            output.append(f"❌ Query failed: {result['error']}")
    elif sql_query and not all_valid:
        output.append("\n⚠️ Query not executed due to validation failures")

    return "\n".join(output)


# ============================================================
# Adaptive Analysis Tools (PATCHED)
# ============================================================

@tool
@safe_tool_wrapper
def find_optimal_separation_conditions(
    target_polymer: str,
    comparison_polymers: str,
    start_temperature: float = 25.0,
    initial_selectivity: float = 30.0,
    export_csv: bool = False,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Find optimal solvent and temperature to separate target polymer from others.

    Searches for solvents that selectively dissolve the target polymer while leaving
    comparison polymers undissolved. Uses adaptive temperature sweeps.

    Parameters:
    - target_polymer: The polymer you want to dissolve (e.g., "LDPE", "PP")
    - comparison_polymers: Polymers to NOT dissolve, comma-separated (e.g., "HDPE,PP,PS")
    - start_temperature: Starting temperature for search in °C (default: 25.0)
    - initial_selectivity: Required selectivity in percentage points (default: 30.0)
    - export_csv: If True, export detailed results to CSV (default: False)

    WHEN TO USE:
    - "Find optimal conditions to separate LDPE from HDPE and PP"
    - "What solvent and temperature selectively dissolves PS?"
    - "Optimal separation conditions for PET from mixed plastics"
    """
    
    # Safely parse comparison_polymers
    if isinstance(comparison_polymers, str):
        comp_polymers = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
    elif isinstance(comparison_polymers, list):
        comp_polymers = comparison_polymers
    else:
        return f"Error: comparison_polymers must be a comma-separated string, got {type(comparison_polymers)}"
    
    if not comp_polymers:
        return "Error: No comparison polymers specified."
    
    all_polymers = [target_polymer] + comp_polymers
    
    is_valid, msg = verify_inputs(
        table_name,
        {
            "polymer": polymer_column,
            "solvent": solvent_column,
            "temperature": temperature_column,
            "solubility": solubility_column
        },
        {polymer_column: all_polymers}
    )

    if not is_valid:
        return f"❌ Input validation failed:\n{msg}"

    output = [f"**Adaptive Separation Analysis**\n"]
    output.append(f"Target: Dissolve {target_polymer}")
    output.append(f"Separate from: {', '.join(comp_polymers)}")
    output.append(f"Starting conditions: T={start_temperature}°C, selectivity threshold={initial_selectivity}%\n")

    result = sql_db.analyzer.adaptive_separation_analysis(
        table_name, polymer_column, solvent_column,
        temperature_column, solubility_column,
        target_polymer, comp_polymers,
        initial_temp=start_temperature,
        initial_selectivity=initial_selectivity
    )

    if result.is_feasible:
        output.append("✅ **Separation IS FEASIBLE**\n")
        output.append(f"**Optimal Conditions:**")
        output.append(f"  - Temperature: {result.conditions['temperature']}°C")
        output.append(f"  - Solvent: {result.conditions['best_solvent']}")
        output.append(f"  - Selectivity: {result.selectivity:.1f}%")
        output.append(f"  - Target solubility: {result.conditions['target_solubility']:.1f}%")
        output.append(f"  - Max other solubility: {result.conditions['max_other_solubility']:.1f}%")
        output.append(f"  - Confidence: {result.confidence:.1%}")

        if result.alternative_conditions:
            output.append("\n**Alternative Conditions:**")
            for i, alt in enumerate(result.alternative_conditions[:3], 1):
                output.append(f"  {i}. T={alt['temperature']}°C, {alt['best_solvent']} (selectivity={alt['selectivity']:.1f}%)")
    else:
        output.append("⚠️ **Separation NOT FEASIBLE** with current data\n")

    output.append("\n**Recommendations:**")
    for rec in result.recommendations:
        output.append(f"  - {rec}")

    # Generate CSV export if requested
    export_id = None
    if export_csv and result.is_feasible:
        try:
            from export_manager import export_manager

            # Prepare export data
            export_data = []

            # Add optimal condition
            optimal = {
                "rank": 1,
                "solvent": result.conditions['best_solvent'],
                "temperature": result.conditions['temperature'],
                "selectivity": result.selectivity,
                "target_solubility": result.conditions['target_solubility'],
                "max_other_solubility": result.conditions['max_other_solubility'],
                "confidence": result.confidence
            }
            export_data.append(optimal)

            # Add alternative conditions
            if result.alternative_conditions:
                for i, alt in enumerate(result.alternative_conditions, 2):
                    alt_data = {
                        "rank": i,
                        "solvent": alt['best_solvent'],
                        "temperature": alt['temperature'],
                        "selectivity": alt['selectivity'],
                        "target_solubility": alt.get('target_solubility', 0),
                        "max_other_solubility": alt.get('max_other_solubility', 0),
                        "confidence": alt.get('confidence', 0)
                    }
                    export_data.append(alt_data)

            export_id = export_manager.create_export(
                data=export_data,
                tool_name="separation_analysis",
                columns=["rank", "solvent", "temperature", "selectivity", "target_solubility", "max_other_solubility", "confidence"]
            )

            output.append(f"\n📥 **CSV Export Available:** `/api/export/{export_id}`")
        except Exception as e:
            logger.error(f"Failed to create CSV export: {e}")

    return "\n".join(output)


@tool
@safe_tool_wrapper
def adaptive_threshold_search(
    target_polymer: str,
    comparison_polymers: Optional[str] = None,
    temperature: float = 25.0,
    start_threshold: float = 0.5,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Search for selective solvents using adaptive thresholds.

    Automatically adjusts selectivity thresholds to find the best solvents
    that dissolve target polymer while avoiding others.

    Parameters:
    - target_polymer: The polymer you want to dissolve (e.g., "LDPE")
    - comparison_polymers: Polymers to avoid, comma-separated (optional)
    - temperature: Target temperature in °C (default: 25.0)
    - start_threshold: Starting selectivity threshold (default: 0.5)

    WHEN TO USE:
    - "Find selective solvents for LDPE using adaptive search"
    - "Search for solvents that dissolve PP selectively"
    """
    
    # Ensure comp_list is always a list
    comp_list = []
    if comparison_polymers:
        if isinstance(comparison_polymers, str):
            comp_list = [p.strip() for p in comparison_polymers.split(',')]
        elif isinstance(comparison_polymers, list):
            comp_list = comparison_polymers
    else:
        try:
            all_polymers_query = f"SELECT DISTINCT {polymer_column} FROM {table_name}"
            result = sql_db.conn.execute(all_polymers_query).fetchdf()
            if len(result) > 0 and polymer_column in result.columns:
                comp_list = [p for p in result[polymer_column].tolist() if p != target_polymer]
        except Exception as e:
            logger.warning(f"Could not get polymers: {e}")
            return f"Error: Could not retrieve polymer list. Please verify table '{table_name}' exists and has data."
    
    if not comp_list:
        return f"Error: No comparison polymers found. Please specify comparison_polymers or ensure data exists."

    output = [f"**Adaptive Threshold Search**\n"]
    output.append(f"Target: {target_polymer}")
    output.append(f"Comparing against: {', '.join(comp_list)}")
    output.append(f"Temperature: {temperature}°C")
    output.append(f"Starting threshold: {start_threshold}\n")

    def search_at_threshold(threshold: float) -> List[Dict]:
        results = []
        temp_tolerance = 5.0

        all_polymers = [target_polymer] + comp_list
        polymer_filter = "', '".join(all_polymers)

        query = f"""
        SELECT {solvent_column}, {polymer_column}, AVG({solubility_column}) as avg_sol
        FROM {table_name}
        WHERE {polymer_column} IN ('{polymer_filter}')
        AND {temperature_column} BETWEEN {temperature - temp_tolerance} AND {temperature + temp_tolerance}
        GROUP BY {solvent_column}, {polymer_column}
        """

        df = sql_db.conn.execute(query).fetchdf()

        for solvent in df[solvent_column].unique():
            solvent_data = df[df[solvent_column] == solvent]

            target_data = solvent_data[solvent_data[polymer_column] == target_polymer]
            if len(target_data) == 0:
                continue
            target_sol = target_data['avg_sol'].values[0]

            other_data = solvent_data[solvent_data[polymer_column].isin(comp_list)]
            if len(other_data) == 0:
                max_other = 0
            else:
                max_other = other_data['avg_sol'].max()

            selectivity = target_sol - max_other
            if selectivity >= threshold:
                results.append({
                    'solvent': solvent,
                    'selectivity': selectivity,
                    'target_solubility': target_sol,
                    'max_other_solubility': max_other
                })

        return sorted(results, key=lambda x: x['selectivity'], reverse=True)

    thresholds = [t for t in AdaptiveAnalyzer.SELECTIVITY_THRESHOLDS if t <= start_threshold]
    search_result = sql_db.analyzer.find_threshold_with_results(
        search_at_threshold, thresholds, min_results=1
    )

    output.append(f"**Search Path:** {search_result.search_path}\n")
    output.append(f"Thresholds tried: {search_result.thresholds_tried}")

    if search_result.found:
        output.append(f"\n✅ **Found {len(search_result.results)} selective solvent(s)** at threshold {search_result.threshold_used}\n")
        output.append("**Results:**")
        for i, r in enumerate(search_result.results[:10], 1):
            output.append(f"  {i}. {r['solvent']}")
            output.append(f"     Selectivity: {r['selectivity']:.4f}")
            output.append(f"     {target_polymer} solubility: {r['target_solubility']:.4f}")
            output.append(f"     Max other solubility: {r['max_other_solubility']:.4f}")
    else:
        output.append(f"\n❌ **No selective solvents found** even at threshold {thresholds[-1]}")
        output.append("\nConsider:")
        output.append("  - Exploring higher temperatures")
        output.append("  - Using find_optimal_separation_conditions for comprehensive search")

    return "\n".join(output)


@tool
@safe_tool_wrapper
def analyze_selective_solubility_enhanced(
    target_polymer: str,
    comparison_polymers: Optional[str] = None,
    temperature_range: str = "25-120",
    auto_threshold: bool = True,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Enhanced selective solubility analysis with adaptive thresholds.

    Find solvents that selectively dissolve the target polymer while NOT dissolving others.

    Parameters:
    - target_polymer: The polymer you want to dissolve (e.g., "LDPE", "PP", "PET")
    - comparison_polymers: Polymers to avoid dissolving, comma-separated (e.g., "HDPE,PP,PS")
                          If not provided, compares against ALL other polymers
    - temperature_range: Temperature range as "min-max" or single temperature (e.g., "100" or "80-120")
    - auto_threshold: Whether to use adaptive thresholds (default: True)

    WHEN TO USE:
    - "Find a solvent that dissolves LDPE but not HDPE"
    - "What solvent is selective for PP over PET at 100°C?"
    - "Find selective solvents for PS separation from mixed plastics"
    """
    # Handle single temperature or range
    if '-' in str(temperature_range):
        parts = str(temperature_range).split('-')
        if len(parts) >= 2:
            temp_min, temp_max = float(parts[0]), float(parts[1])
        else:
            temp_min = temp_max = float(parts[0])
    else:
        # Single temperature - use as both min and max (±5°C range)
        temp = float(temperature_range)
        temp_min, temp_max = temp - 5, temp + 5

    # Safely build comp_list
    comp_list = []
    if comparison_polymers:
        if isinstance(comparison_polymers, str):
            comp_list = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
        elif isinstance(comparison_polymers, list):
            comp_list = comparison_polymers
        output = [f"**Selective Solubility Analysis (Targeted Comparison)**\n"]
    else:
        try:
            all_query = f"SELECT DISTINCT {polymer_column} FROM {table_name}"
            result = sql_db.conn.execute(all_query).fetchdf()
            if len(result) > 0 and polymer_column in result.columns:
                comp_list = [p for p in result[polymer_column].tolist() if p != target_polymer]
        except Exception as e:
            logger.warning(f"Could not get polymers: {e}")
            return f"Error: Could not retrieve polymer list from '{table_name}'"
        output = [f"**Selective Solubility Analysis (All Polymers)**\n"]
    
    if not comp_list:
        return f"Error: No comparison polymers found."

    output.append(f"Target: {target_polymer}")
    output.append(f"Comparing against: {', '.join(comp_list)}")
    output.append(f"Temperature range: {temp_min}°C - {temp_max}°C\n")

    val_result = sql_db.validator.verify_value_exists(table_name, polymer_column, target_polymer)
    if not val_result.is_valid:
        return f"❌ Target polymer '{target_polymer}' not found. {val_result.warnings}"

    all_polymers = [target_polymer] + comp_list
    polymer_filter = "', '".join(all_polymers)

    query = f"""
    SELECT {solvent_column}, {polymer_column},
           AVG({solubility_column}) as avg_solubility,
           MIN({solubility_column}) as min_solubility,
           MAX({solubility_column}) as max_solubility,
           COUNT(*) as n_points
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {temperature_column} >= {temp_min} AND {temperature_column} <= {temp_max}
    GROUP BY {solvent_column}, {polymer_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]
    output.append(f"Data points analyzed: {len(df)}\n")

    solvents = df[solvent_column].unique()
    selectivity_data = []

    for solvent in solvents:
        solvent_data = df[df[solvent_column] == solvent]

        target_sol = solvent_data[solvent_data[polymer_column] == target_polymer]
        if len(target_sol) == 0:
            continue
        target_avg = target_sol['avg_solubility'].values[0]
        target_n = target_sol['n_points'].values[0]

        other_data = solvent_data[solvent_data[polymer_column].isin(comp_list)]
        if len(other_data) == 0:
            max_other = 0
            avg_other = 0
        else:
            max_other = other_data['avg_solubility'].max()
            avg_other = other_data['avg_solubility'].mean()

        selectivity = target_avg - max_other
        selectivity_ratio = target_avg / max_other if max_other > 0.001 else float('inf')

        selectivity_data.append({
            'solvent': solvent,
            'target_solubility': target_avg,
            'max_other_solubility': max_other,
            'avg_other_solubility': avg_other,
            'selectivity_difference': selectivity,
            'selectivity_ratio': selectivity_ratio,
            'n_data_points': target_n
        })

    selectivity_data.sort(key=lambda x: x['selectivity_difference'], reverse=True)

    if not selectivity_data:
        return f"❌ No selectivity data found for {target_polymer}"

    if auto_threshold:
        thresholds_tried = []
        for threshold in AdaptiveAnalyzer.SELECTIVITY_THRESHOLDS:
            selective_solvents = [s for s in selectivity_data if s['selectivity_difference'] >= threshold]
            thresholds_tried.append((threshold, len(selective_solvents)))
            if len(selective_solvents) > 0:
                output.append(f"**Adaptive Threshold:** Found {len(selective_solvents)} solvent(s) at threshold {threshold}")
                break

        output.append(f"Thresholds searched: {[t[0] for t in thresholds_tried]}\n")

    output.append("**Selective Solvents (ranked by selectivity):**\n")
    for i, data in enumerate(selectivity_data[:15], 1):
        sel_symbol = "✅" if data['selectivity_difference'] > 10 else "⚠️" if data['selectivity_difference'] > 0 else "❌"
        output.append(f"{i}. {sel_symbol} **{data['solvent']}**")
        output.append(f"   - {target_polymer} solubility: {data['target_solubility']:.4f}")
        output.append(f"   - Max comparison solubility: {data['max_other_solubility']:.4f}")
        output.append(f"   - Selectivity: {data['selectivity_difference']:.4f} ({data['selectivity_ratio']:.1f}x)")
        output.append(f"   - Data points: {data['n_data_points']}")

    # Create visualization
    if len(selectivity_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        top_n = min(12, len(selectivity_data))
        solvent_names = [d['solvent'] for d in selectivity_data[:top_n]]
        target_sols = [d['target_solubility'] for d in selectivity_data[:top_n]]
        other_sols = [d['max_other_solubility'] for d in selectivity_data[:top_n]]

        x = np.arange(len(solvent_names))
        width = 0.35

        axes[0].bar(x - width/2, target_sols, width, label=target_polymer, color='green', alpha=0.8)
        axes[0].bar(x + width/2, other_sols, width, label='Max Comparison', color='red', alpha=0.8)
        axes[0].set_xlabel('Solvent', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Average Solubility', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Selective Solvents for {target_polymer}', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(solvent_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        selectivity_diffs = [d['selectivity_difference'] for d in selectivity_data[:top_n]]
        colors = ['green' if s > 10 else 'orange' if s > 0 else 'red' for s in selectivity_diffs]
        axes[1].barh(solvent_names, selectivity_diffs, color=colors, alpha=0.8)
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].axvline(x=10, color='green', linestyle='--', linewidth=1, label='Good selectivity (10%)')
        axes[1].set_xlabel('Selectivity Difference', fontsize=12, fontweight='bold')
        axes[1].set_title('Selectivity Ranking', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filepath = save_plot(fig, f"selective_solubility_{target_polymer}", "matplotlib")
        output.append(f"\n{get_plot_url(filepath)}")

    del df
    gc.collect()

    display = "\n".join(output)

    # Build structured data for programmatic access
    top_solvents = selectivity_data[:10] if selectivity_data else []
    structured_data = {
        "tool_name": "analyze_selective_solubility_enhanced",
        "success": True,
        "polymers_analyzed": [target_polymer] + comp_list,
        "solvents": [s['solvent'] for s in top_solvents],
        "selectivities": [s['selectivity_difference'] for s in top_solvents],
        "best_solvent": top_solvents[0]['solvent'] if top_solvents else None,
        "best_selectivity": top_solvents[0]['selectivity_difference'] if top_solvents else None,
        "temperature": (temp_min + temp_max) / 2,
        "temperature_range": [temp_min, temp_max],
        "target_polymer": target_polymer,
        "comparison_polymers": comp_list,
        "algorithm_used": "selective_solubility",
        "coverage_complete": len(top_solvents) > 0,
    }

    # Return structured JSON
    import json
    return json.dumps({"display": display, "data": structured_data})


# ============================================================
# Statistical Analysis Tools (PATCHED)
# ============================================================

@tool
@safe_tool_wrapper
def statistical_summary(
    table_name: str,
    value_column: str,
    group_by_column: Optional[str] = None,
    filters: Optional[str] = None
) -> str:
    """Comprehensive statistical summary with confidence intervals."""
    where_clause = f"WHERE {filters}" if filters else ""

    if group_by_column:
        query = f"""
        SELECT {group_by_column},
               COUNT({value_column}) as n,
               AVG({value_column}) as mean,
               STDDEV({value_column}) as std,
               MIN({value_column}) as min,
               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {value_column}) as q1,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {value_column}) as median,
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {value_column}) as q3,
               MAX({value_column}) as max
        FROM {table_name}
        {where_clause}
        GROUP BY {group_by_column}
        ORDER BY {group_by_column}
        """
    else:
        query = f"""
        SELECT COUNT({value_column}) as n,
               AVG({value_column}) as mean,
               STDDEV({value_column}) as std,
               MIN({value_column}) as min,
               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {value_column}) as q1,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {value_column}) as median,
               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {value_column}) as q3,
               MAX({value_column}) as max
        FROM {table_name}
        {where_clause}
        """

    result = sql_db.execute_query(query, limit=1000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]

    output = [f"**Statistical Summary: {value_column}**\n"]
    if filters:
        output.append(f"Filters: {filters}\n")

    output.append(df.to_markdown(index=False))

    if group_by_column:
        output.append("\n**95% Confidence Intervals:**")
        for _, row in df.iterrows():
            if row['n'] > 1 and row['std'] is not None and not pd.isna(row['std']):
                ci = 1.96 * row['std'] / np.sqrt(row['n'])
                output.append(f"  - {row[group_by_column]}: {row['mean']:.4f} ± {ci:.4f}")
    else:
        if df['n'].iloc[0] > 1 and df['std'].iloc[0] is not None and not pd.isna(df['std'].iloc[0]):
            ci = 1.96 * df['std'].iloc[0] / np.sqrt(df['n'].iloc[0])
            output.append(f"\n**95% CI:** {df['mean'].iloc[0]:.4f} ± {ci:.4f}")

    del df
    return "\n".join(output)


@tool
@safe_tool_wrapper
def correlation_analysis(
    table_name: str,
    columns: str,
    filters: Optional[str] = None,
    method: str = "pearson"
) -> str:
    """Analyze correlations between multiple columns."""
    col_list = [c.strip() for c in columns.split(',')]
    where_clause = f"WHERE {filters}" if filters else ""

    query = f"SELECT {', '.join(col_list)} FROM {table_name} {where_clause}"
    result = sql_db.execute_query(query, limit=100000)

    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"].dropna()

    if len(df) < 3:
        return f"❌ Insufficient data for correlation analysis (n={len(df)})"

    corr_matrix = df.corr(method=method)

    output = [f"**Correlation Analysis ({method.title()})**\n"]
    output.append(f"Data points: {len(df)}\n")
    output.append("**Correlation Matrix:**")
    output.append(corr_matrix.round(3).to_markdown())

    output.append("\n**Significant Correlations (p < 0.05):**")
    for i, col1 in enumerate(col_list):
        for col2 in col_list[i+1:]:
            try:
                if method == 'pearson':
                    r, p = stats.pearsonr(df[col1], df[col2])
                elif method == 'spearman':
                    r, p = stats.spearmanr(df[col1], df[col2])
                else:
                    r, p = stats.kendalltau(df[col1], df[col2])

                if p < 0.05:
                    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                    direction = "positive" if r > 0 else "negative"
                    output.append(f"  - {col1} vs {col2}: r={r:.3f}, p={p:.4f} ({strength} {direction})")
            except:
                pass

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f'{method.title()} Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = save_plot(fig, "correlation_matrix", "matplotlib")
    output.append(f"\n{get_plot_url(filepath)}")

    del df
    return "\n".join(output)


@tool
@safe_tool_wrapper
async def compare_groups_statistically(
    table_name: str,
    value_column: str,
    group_column: str,
    group1: str,
    group2: str,
    filters: Optional[str] = None
) -> str:
    """Statistical comparison between two groups with hypothesis testing (ASYNC)."""
    async_db = get_async_db()
    where_clause = f"WHERE {filters} AND" if filters else "WHERE"

    query1 = f"SELECT {value_column} FROM {table_name} {where_clause} LOWER({group_column}) = LOWER('{group1}')"
    query2 = f"SELECT {value_column} FROM {table_name} {where_clause} LOWER({group_column}) = LOWER('{group2}')"

    # PARALLEL EXECUTION - Run both queries concurrently
    try:
        df1, df2 = await async_db.execute_many_async([query1, query2])
    except Exception as e:
        return f"❌ Query failed: {str(e)[:300]}"

    if len(df1) == 0 or len(df2) == 0:
        return f"❌ No data returned for groups: {group1} ({len(df1)} rows), {group2} ({len(df2)} rows)"

    data1 = df1[value_column].dropna()
    data2 = df2[value_column].dropna()

    if len(data1) < 3 or len(data2) < 3:
        return f"❌ Insufficient data: {group1} has {len(data1)}, {group2} has {len(data2)} samples"

    output = [f"**Statistical Comparison: {group1} vs {group2}**\n"]

    output.append("**Descriptive Statistics:**")
    output.append(f"| Metric | {group1} | {group2} |")
    output.append("|--------|----------|----------|")
    output.append(f"| N | {len(data1)} | {len(data2)} |")
    output.append(f"| Mean | {data1.mean():.4f} | {data2.mean():.4f} |")
    output.append(f"| Std | {data1.std():.4f} | {data2.std():.4f} |")
    output.append(f"| Median | {data1.median():.4f} | {data2.median():.4f} |")

    # Hypothesis tests
    output.append("\n**Hypothesis Tests:**")
    t_stat, t_p = stats.ttest_ind(data1, data2)
    output.append(f"  - Independent t-test: t={t_stat:.3f}, p={t_p:.4f}")

    u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    output.append(f"  - Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

    # Effect size
    pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) /
                        (len(data1)+len(data2)-2))
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
    effect_size = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    output.append(f"\n**Effect Size:** Cohen's d = {cohens_d:.3f} ({effect_size})")

    # Interpretation
    output.append("\n**Interpretation:**")
    if t_p < 0.05:
        direction = "higher" if data1.mean() > data2.mean() else "lower"
        output.append(f"  ✅ Significant difference (p < 0.05): {group1} has {direction} values")
    else:
        output.append(f"  ⚠️ No significant difference (p = {t_p:.4f})")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bp = axes[0].boxplot([data1, data2], labels=[group1, group2], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel(value_column.replace('_', ' ').title())
    axes[0].set_title('Distribution Comparison', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].hist(data1, bins=30, alpha=0.6, label=group1, density=True)
    axes[1].hist(data2, bins=30, alpha=0.6, label=group2, density=True)
    axes[1].set_xlabel(value_column.replace('_', ' ').title())
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution Overlap', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = save_plot(fig, "group_comparison", "matplotlib")
    output.append(f"\n{get_plot_url(filepath)}")

    return "\n".join(output)


@tool
@safe_tool_wrapper
def regression_analysis(
    table_name: str,
    x_column: str,
    y_column: str,
    group_by: Optional[str] = None,
    filters: Optional[str] = None,
    degree: int = 1
) -> str:
    """Perform regression analysis with model fitting and diagnostics."""
    where_clause = f"WHERE {filters}" if filters else ""

    if group_by:
        query = f"SELECT {x_column}, {y_column}, {group_by} FROM {table_name} {where_clause}"
    else:
        query = f"SELECT {x_column}, {y_column} FROM {table_name} {where_clause}"

    result = sql_db.execute_query(query, limit=100000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"].dropna()

    output = [f"**Regression Analysis: {y_column} ~ {x_column}**\n"]
    output.append(f"Model: Polynomial degree {degree}")
    output.append(f"Data points: {len(df)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

        output.append("**Regression Results by Group:**\n")

        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group]
            x = group_data[x_column].values
            y = group_data[y_column].values

            if len(x) < degree + 1:
                continue

            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            output.append(f"**{group}:** R²={r2:.4f}, RMSE={rmse:.4f}")

            axes[0].scatter(x, y, alpha=0.5, color=colors[i], label=f'{group} (R²={r2:.3f})')
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[0].plot(x_line, poly(x_line), color=colors[i], linewidth=2)

        axes[0].legend(fontsize=9)
    else:
        x = df[x_column].values
        y = df[y_column].values

        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        if degree == 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            output.append(f"**Linear Regression:**")
            output.append(f"  - Slope: {slope:.4f} (SE: {std_err:.4f})")
            output.append(f"  - Intercept: {intercept:.4f}")
            output.append(f"  - p-value: {p_value:.4e}")

        output.append(f"\n**Model Fit:** R²={r2:.4f}, RMSE={rmse:.4f}")

        axes[0].scatter(x, y, alpha=0.5, color='steelblue')
        x_line = np.linspace(x.min(), x.max(), 100)
        axes[0].plot(x_line, poly(x_line), 'r-', linewidth=2, label=f'Fit (R²={r2:.3f})')
        axes[0].legend()

        residuals = y - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    axes[0].set_xlabel(x_column.replace('_', ' ').title(), fontweight='bold')
    axes[0].set_ylabel(y_column.replace('_', ' ').title(), fontweight='bold')
    axes[0].set_title(f'Regression: {y_column} vs {x_column}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = save_plot(fig, "regression_analysis", "matplotlib")
    output.append(f"\n{get_plot_url(filepath)}")

    del df
    return "\n".join(output)


# ============================================================
# Visualization Tools (PATCHED)
# ============================================================

@tool
@safe_tool_wrapper
def plot_solubility_vs_temperature(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    solvents: str,
    plot_title: Optional[str] = None,
    include_confidence_bands: bool = True,
    temperature_min: Optional[float] = None,
    temperature_max: Optional[float] = None
) -> str:
    """
    Create temperature vs solubility curves with validation and confidence bands.

    Args:
        table_name: Database table name
        polymer_column: Column containing polymer names
        solvent_column: Column containing solvent names
        temperature_column: Column containing temperature values
        solubility_column: Column containing solubility values
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents
        plot_title: Optional custom plot title
        include_confidence_bands: Whether to show confidence intervals (default: True)
        temperature_min: Minimum temperature to plot (optional)
        temperature_max: Maximum temperature to plot (optional)

    Returns:
        Formatted output with plot URL
    """
    polymer_list = [p.strip() for p in polymers.split(',')]
    solvent_list = [s.strip() for s in solvents.split(',')]

    # Normalize solvent names (e.g., "xylene" → "1,2-dimethylbenzene", "1,4-dimethylbenzene")
    solvent_list = normalize_solvent_names(solvent_list)

    is_valid, msg = verify_inputs(
        table_name,
        {
            "polymer": polymer_column,
            "solvent": solvent_column,
            "temperature": temperature_column,
            "solubility": solubility_column
        },
        {polymer_column: polymer_list, solvent_column: solvent_list}
    )

    if not is_valid:
        return f"❌ Validation failed:\n{msg}"

    polymer_filter = "', '".join(polymer_list)
    solvent_filter = "', '".join(solvent_list)

    # Build temperature filter if specified
    temp_filter = ""
    if temperature_min is not None and temperature_max is not None:
        temp_filter = f"AND {temperature_column} BETWEEN {temperature_min} AND {temperature_max}"
    elif temperature_min is not None:
        temp_filter = f"AND {temperature_column} >= {temperature_min}"
    elif temperature_max is not None:
        temp_filter = f"AND {temperature_column} <= {temperature_max}"

    query = f"""
    SELECT {polymer_column}, {solvent_column}, {temperature_column},
           AVG({solubility_column}) as avg_sol,
           STDDEV({solubility_column}) as std_sol,
           COUNT(*) as n
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {solvent_column} IN ('{solvent_filter}')
    {temp_filter}
    GROUP BY {polymer_column}, {solvent_column}, {temperature_column}
    ORDER BY {polymer_column}, {solvent_column}, {temperature_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"] or result["rows"] == 0:
        return f"❌ No data found. Error: {result.get('error', 'No matching rows')}"

    df = result["dataframe"]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(polymer_list) * len(solvent_list)))
    color_idx = 0

    for polymer in polymer_list:
        for solvent in solvent_list:
            mask = (df[polymer_column] == polymer) & (df[solvent_column] == solvent)
            data = df[mask].sort_values(temperature_column)

            if len(data) > 0:
                temps = data[temperature_column]
                sols = data['avg_sol']

                line, = ax.plot(temps, sols, marker='o', linewidth=2, markersize=6,
                               label=f"{polymer} in {solvent}", color=colors[color_idx])

                if include_confidence_bands and 'std_sol' in data.columns:
                    std = data['std_sol'].fillna(0)
                    n = data['n']
                    se = std / np.sqrt(n.replace(0, 1))
                    ax.fill_between(temps, sols - 1.96*se, sols + 1.96*se,
                                    alpha=0.2, color=colors[color_idx])

                color_idx += 1

    ax.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solubility', fontsize=12, fontweight='bold')
    title = plot_title or 'Solubility vs Temperature'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis limits based on temperature range if specified
    if temperature_min is not None or temperature_max is not None:
        current_xlim = ax.get_xlim()
        new_min = temperature_min if temperature_min is not None else current_xlim[0]
        new_max = temperature_max if temperature_max is not None else current_xlim[1]
        ax.set_xlim(new_min, new_max)

    plt.tight_layout()

    filepath = save_plot(fig, "solubility_temp_curve", "matplotlib")

    # Clean up figure to prevent memory leaks
    plt.close(fig)

    output = f"✅ **Solubility vs Temperature Plot Created**\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}°C - {temperature_max}°C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}°C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}°C\n"
    output += f"Data points: {result['rows']}\n"
    if include_confidence_bands:
        output += "Shaded regions show 95% confidence intervals\n"
    output += f"\n{get_plot_url(filepath)}"

    del df
    gc.collect()  # Force garbage collection
    return output


@tool
@safe_tool_wrapper
def plot_solubility_vs_temperature_interactive(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    solvents: str,
    plot_title: Optional[str] = None,
    temperature_min: Optional[float] = None,
    temperature_max: Optional[float] = None
) -> str:
    """
    Create INTERACTIVE temperature vs solubility curves with sliders and toggleable lines.

    Features:
    - Interactive hover tooltips showing exact values
    - Range slider to zoom into temperature ranges
    - Clickable legend to show/hide individual solvent/polymer combinations
    - Zoom, pan, and screenshot tools
    - Opens in new tab as HTML file

    Args:
        table_name: Database table name
        polymer_column: Column containing polymer names
        solvent_column: Column containing solvent names
        temperature_column: Column containing temperature values
        solubility_column: Column containing solubility values
        polymers: Comma-separated list of polymers
        solvents: Comma-separated list of solvents
        plot_title: Optional custom plot title
        temperature_min: Minimum temperature to plot (optional)
        temperature_max: Maximum temperature to plot (optional)

    Returns:
        Link to interactive HTML visualization
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    polymer_list = [p.strip() for p in polymers.split(',')]
    solvent_list = [s.strip() for s in solvents.split(',')]

    # Normalize solvent names (e.g., "xylene" → "1,2-dimethylbenzene", "1,4-dimethylbenzene")
    solvent_list = normalize_solvent_names(solvent_list)

    is_valid, msg = verify_inputs(
        table_name,
        {
            "polymer": polymer_column,
            "solvent": solvent_column,
            "temperature": temperature_column,
            "solubility": solubility_column
        },
        {polymer_column: polymer_list, solvent_column: solvent_list}
    )

    if not is_valid:
        return f"❌ Validation failed:\n{msg}"

    polymer_filter = "', '".join(polymer_list)
    solvent_filter = "', '".join(solvent_list)

    # Build temperature filter if specified
    temp_filter = ""
    if temperature_min is not None and temperature_max is not None:
        temp_filter = f"AND {temperature_column} BETWEEN {temperature_min} AND {temperature_max}"
    elif temperature_min is not None:
        temp_filter = f"AND {temperature_column} >= {temperature_min}"
    elif temperature_max is not None:
        temp_filter = f"AND {temperature_column} <= {temperature_max}"

    query = f"""
    SELECT {polymer_column}, {solvent_column}, {temperature_column},
           AVG({solubility_column}) as avg_sol,
           STDDEV({solubility_column}) as std_sol,
           COUNT(*) as n
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {solvent_column} IN ('{solvent_filter}')
    {temp_filter}
    GROUP BY {polymer_column}, {solvent_column}, {temperature_column}
    ORDER BY {polymer_column}, {solvent_column}, {temperature_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"] or result["rows"] == 0:
        return f"❌ No data found. Error: {result.get('error', 'No matching rows')}"

    df = result["dataframe"]

    # Create interactive Plotly figure
    fig = go.Figure()

    # Use a nice color palette
    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for polymer in polymer_list:
        for solvent in solvent_list:
            mask = (df[polymer_column] == polymer) & (df[solvent_column] == solvent)
            data = df[mask].sort_values(temperature_column)

            if len(data) > 0:
                temps = data[temperature_column]
                sols = data['avg_sol']

                # Add main line trace
                fig.add_trace(go.Scatter(
                    x=temps,
                    y=sols,
                    mode='lines+markers',
                    name=f"{polymer} in {solvent}",
                    line=dict(width=3, color=colors[color_idx % len(colors)]),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate=(
                        f"<b>{polymer} in {solvent}</b><br>" +
                        "Temperature: %{x:.1f}°C<br>" +
                        "Solubility: %{y:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))

                # Add confidence band if available
                if 'std_sol' in data.columns:
                    std = data['std_sol'].fillna(0)
                    n = data['n']
                    se = std / np.sqrt(n.replace(0, 1))
                    upper = sols + 1.96*se
                    lower = sols - 1.96*se

                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=temps,
                        y=upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=temps,
                        y=lower,
                        mode='lines',
                        line=dict(width=0),
                        fillcolor=colors[color_idx % len(colors)],
                        fill='tonexty',
                        opacity=0.2,
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                color_idx += 1

    # Update layout with interactive features
    title = plot_title or 'Interactive Solubility vs Temperature'
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Temperature (°C)', font=dict(size=16, family='Arial')),
            rangeslider=dict(visible=True, thickness=0.05),  # Interactive range slider!
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text='Solubility', font=dict(size=16, family='Arial')),
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='closest',
        height=700,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12)
        )
    )

    # Add interactive tools config
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'solubility_vs_temp',
            'height': 700,
            'width': 1200,
            'scale': 2
        },
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'displaylogo': False
    }

    # Save as HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interactive_solubility_temp_{timestamp}.html"
    filepath = os.path.join(PLOTS_DIR, filename)

    fig.write_html(filepath, config=config)

    # Create output message
    output = f"✅ **Interactive Solubility vs Temperature Visualization Created**\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}°C - {temperature_max}°C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}°C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}°C\n"
    output += f"Data points: {result['rows']}\n\n"

    output += f"## 🎮 Interactive Features:\n"
    output += f"- **Click legend items** to show/hide individual curves\n"
    output += f"- **Drag the range slider** below the plot to zoom into temperature ranges\n"
    output += f"- **Hover over points** to see exact values\n"
    output += f"- **Use toolbar** to zoom, pan, reset, or download as PNG\n"
    output += f"- **Double-click legend** to isolate a single curve\n\n"

    html_url = f"/plots/{filename}"
    output += f"**[Click here to open the interactive plot]({html_url})**\n"
    output += f"(Opens in a new tab with full interactivity)\n"

    del df
    gc.collect()
    return output


@tool
@safe_tool_wrapper
def plot_selectivity_heatmap(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: Optional[str] = None,
    temperature: float = 120.0,
    temperature_tolerance: float = 10.0,
    show_selectivity: bool = False,
    max_solvents: int = 30
) -> str:
    """
    Create heatmap showing solubility across polymer-solvent combinations.

    Args:
        table_name: Database table name
        polymer_column: Column with polymer names
        solvent_column: Column with solvent names
        temperature_column: Column with temperature values
        solubility_column: Column with solubility values
        target_polymer: Optional - filter to show only this polymer's data
        temperature: Target temperature (default: 120°C for better polymer dissolution)
        temperature_tolerance: Temperature range ± (default: 10°C)
        show_selectivity: If True and target_polymer set, show selectivity view instead
        max_solvents: Maximum solvents to show (default: 30 for readability)

    Returns:
        Heatmap visualization with solubility data
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Build query - optionally filter to target polymer
    if target_polymer:
        polymer_filter = f"AND UPPER({polymer_column}) = '{target_polymer.upper()}'"
    else:
        polymer_filter = ""

    query = f"""
    SELECT {polymer_column}, {solvent_column},
           AVG({solubility_column}) as avg_solubility
    FROM {table_name}
    WHERE {temperature_column} BETWEEN {temperature - temperature_tolerance}
          AND {temperature + temperature_tolerance}
    {polymer_filter}
    GROUP BY {polymer_column}, {solvent_column}
    ORDER BY avg_solubility DESC
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]

    if len(df) == 0:
        return f"❌ No data found at {temperature}°C ± {temperature_tolerance}°C. Try a different temperature."

    pivot_df = df.pivot(index=polymer_column, columns=solvent_column, values='avg_solubility')

    # Limit solvents for readability - keep top ones by average solubility
    if len(pivot_df.columns) > max_solvents:
        top_solvents = df.groupby(solvent_column)['avg_solubility'].mean().nlargest(max_solvents).index
        pivot_df = pivot_df[top_solvents]

    # Determine if we should show annotations based on size
    n_cells = pivot_df.shape[0] * pivot_df.shape[1]
    show_annot = n_cells <= 150
    annot_fontsize = 10 if n_cells <= 50 else 8 if n_cells <= 100 else 6

    # Create custom colormap that emphasizes 0-20% range
    # More color variation in 0-20% range, then gradual to high values
    colors_low = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6']  # Blues for 0-20%
    colors_high = ['#4292c6', '#2171b5', '#08519c', '#08306b']  # Darker blues for >20%

    # Custom colormap with emphasis on low values
    cmap = LinearSegmentedColormap.from_list('solubility_emphasis',
        colors_low + colors_high, N=256)

    # Single clean figure
    fig, ax = plt.subplots(figsize=(max(14, len(pivot_df.columns) * 0.5),
                                     max(8, len(pivot_df) * 0.8)))

    # Use logarithmic-like normalization to emphasize low values
    from matplotlib.colors import PowerNorm
    vmax = pivot_df.max().max()
    if vmax > 20:
        # Power norm with gamma < 1 emphasizes lower values
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    else:
        norm = None

    sns.heatmap(pivot_df, annot=show_annot, fmt='.1f', cmap=cmap,
               cbar_kws={'label': 'Solubility (%)', 'shrink': 0.8},
               linewidths=0.3, ax=ax, annot_kws={'size': annot_fontsize},
               norm=norm)

    title = f'Solubility Heatmap at {temperature}°C'
    if target_polymer:
        title = f'{target_polymer} Solubility at {temperature}°C'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Solvent', fontsize=14, fontweight='bold')
    ax.set_ylabel('Polymer', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    plt.tight_layout()
    filepath = save_plot(fig, "solubility_heatmap", "matplotlib")

    output = f"✅ **Heatmap Created**\n\n"
    output += f"Temperature: {temperature}°C ± {temperature_tolerance}°C\n"
    output += f"Polymers: {len(pivot_df)}\n"
    output += f"Solvents: {len(pivot_df.columns)}\n"

    # Show top solvents summary
    if target_polymer and target_polymer.upper() in [p.upper() for p in pivot_df.index]:
        idx = [p for p in pivot_df.index if p.upper() == target_polymer.upper()][0]
        row = pivot_df.loc[idx].sort_values(ascending=False)
        output += f"\n**Top solvents for {target_polymer}:**\n"
        for solvent, sol in list(row.items())[:10]:
            if pd.notna(sol):
                symbol = "✅" if sol > 20 else "⚠️" if sol > 5 else "❌"
                output += f"  {symbol} {solvent}: {sol:.1f}%\n"

    output += f"\n{get_plot_url(filepath)}"

    # Note about color scale
    output += f"\n\n💡 *Color scale emphasizes 0-20% range for better differentiation of low-solubility solvents.*"

    del df
    return output


@tool
@safe_tool_wrapper
def plot_multi_panel_analysis(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: str,
    solvent: str
) -> str:
    """Create comprehensive multi-panel visualization for separation analysis."""
    
    # Safely parse comparison_polymers
    if isinstance(comparison_polymers, str):
        comp_list = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
    elif isinstance(comparison_polymers, list):
        comp_list = comparison_polymers
    else:
        return f"Error: comparison_polymers must be a comma-separated string"
    
    if not comp_list:
        return "Error: No comparison polymers specified."
    
    all_polymers = [target_polymer] + comp_list
    polymer_filter = "', '".join(all_polymers)

    query = f"""
    SELECT {polymer_column}, {temperature_column},
           AVG({solubility_column}) as avg_sol,
           STDDEV({solubility_column}) as std_sol,
           COUNT(*) as n
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {solvent_column} = '{solvent}'
    GROUP BY {polymer_column}, {temperature_column}
    ORDER BY {polymer_column}, {temperature_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    colors_others = plt.cm.Reds(np.linspace(0.4, 0.8, len(comp_list)))

    # Panel 1: Solubility curves
    ax1 = fig.add_subplot(gs[0, 0])
    target_data = df[df[polymer_column] == target_polymer].sort_values(temperature_column)
    if len(target_data) > 0:
        ax1.plot(target_data[temperature_column], target_data['avg_sol'],
                'o-', color='green', linewidth=3, markersize=8, label=target_polymer)

    for i, comp in enumerate(comp_list):
        comp_data = df[df[polymer_column] == comp].sort_values(temperature_column)
        if len(comp_data) > 0:
            ax1.plot(comp_data[temperature_column], comp_data['avg_sol'],
                    's--', color=colors_others[i], linewidth=2, markersize=6, label=comp)

    ax1.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Solubility', fontsize=11, fontweight='bold')
    ax1.set_title(f'Solubility in {solvent}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Selectivity vs Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    if len(target_data) > 0:
        temps = target_data[temperature_column].values

        for i, comp in enumerate(comp_list):
            comp_data = df[df[polymer_column] == comp].sort_values(temperature_column)
            selectivity = []
            for temp in temps:
                target_sol = target_data[target_data[temperature_column] == temp]['avg_sol']
                comp_sol = comp_data[comp_data[temperature_column] == temp]['avg_sol']
                if len(target_sol) > 0 and len(comp_sol) > 0:
                    selectivity.append(target_sol.values[0] - comp_sol.values[0])
                else:
                    selectivity.append(np.nan)

            ax2.plot(temps, selectivity, 'o-', color=colors_others[i],
                    linewidth=2, markersize=6, label=f'vs {comp}')

        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=10, color='green', linestyle=':', alpha=0.7, label='Good selectivity (10%)')

    ax2.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Selectivity (%) (target - other)', fontsize=11, fontweight='bold')
    ax2.set_title('Selectivity vs Temperature', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Separation window
    ax3 = fig.add_subplot(gs[1, 0])
    good_separation_temps = []
    if len(target_data) > 0:
        for temp in temps:
            target_sol_val = target_data[target_data[temperature_column] == temp]['avg_sol']
            if len(target_sol_val) == 0:
                continue
            target_sol_val = target_sol_val.values[0]

            max_other = 0
            for comp in comp_list:
                comp_data = df[df[polymer_column] == comp]
                comp_sol = comp_data[comp_data[temperature_column] == temp]['avg_sol']
                if len(comp_sol) > 0:
                    max_other = max(max_other, comp_sol.values[0])

            if target_sol_val - max_other > 5:  # 5% threshold for good separation
                good_separation_temps.append(temp)

        all_temps = sorted(temps)
        bar_colors = ['green' if t in good_separation_temps else 'lightgray' for t in all_temps]
        ax3.bar(range(len(all_temps)), [1]*len(all_temps), color=bar_colors, edgecolor='black')
        ax3.set_xticks(range(len(all_temps)))
        ax3.set_xticklabels([f'{int(t)}°C' for t in all_temps], rotation=45, ha='right')

    ax3.set_ylabel('Separation Feasibility', fontsize=11, fontweight='bold')
    ax3.set_title('Separation Window', fontsize=12, fontweight='bold')
    ax3.set_yticks([])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Good separation'),
                      Patch(facecolor='lightgray', label='Poor separation')]
    ax3.legend(handles=legend_elements, loc='upper right')

    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_text = f"**Analysis Summary**\n\n"
    summary_text += f"Target: {target_polymer}\n"
    summary_text += f"Solvent: {solvent}\n"
    summary_text += f"Comparisons: {', '.join(comp_list)}\n\n"

    if good_separation_temps:
        summary_text += f"✅ Separation possible at:\n"
        summary_text += f"   {', '.join([f'{int(t)}°C' for t in good_separation_temps])}\n"
    else:
        summary_text += f"⚠️ No clear separation window\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    filepath = save_plot(fig, "multi_panel_analysis", "matplotlib")

    output = f"✅ **Multi-Panel Analysis Created**\n\n"
    output += f"Target: {target_polymer}\n"
    output += f"Solvent: {solvent}\n"
    output += f"Comparisons: {', '.join(comp_list)}\n\n"

    if good_separation_temps:
        output += f"**Separation possible at:** {', '.join([f'{int(t)}°C' for t in good_separation_temps])}\n"

    output += f"\n{get_plot_url(filepath)}"

    del df
    return output


@tool
@safe_tool_wrapper
def plot_comparison_dashboard(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    temperature: float = 25.0
) -> str:
    """Create a comprehensive comparison dashboard for multiple polymers."""
    polymer_list = [p.strip() for p in polymers.split(',')]
    polymer_filter = "', '".join(polymer_list)

    query = f"""
    SELECT {polymer_column}, {solvent_column},
           AVG({solubility_column}) as avg_sol,
           MAX({solubility_column}) as max_sol,
           MIN({solubility_column}) as min_sol
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {temperature_column} BETWEEN {temperature - 5} AND {temperature + 5}
    GROUP BY {polymer_column}, {solvent_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]
    solvents = df[solvent_column].unique()

    # Limit number of solvents for readability
    max_solvents = 15
    if len(solvents) > max_solvents:
        # Keep top solvents by average solubility
        solvent_means = df.groupby(solvent_column)['avg_sol'].mean().sort_values(ascending=False)
        solvents = solvent_means.head(max_solvents).index.tolist()
        df = df[df[solvent_column].isin(solvents)]

    fig = plt.figure(figsize=(20, 12))

    # Panel 1: Grouped bar chart - IMPROVED
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(solvents))
    width = 0.8 / len(polymer_list)
    colors = plt.cm.Set2(np.linspace(0, 1, len(polymer_list)))

    for i, polymer in enumerate(polymer_list):
        poly_data = df[df[polymer_column] == polymer]
        values = []
        for solvent in solvents:
            sol_data = poly_data[poly_data[solvent_column] == solvent]['avg_sol']
            values.append(sol_data.values[0] if len(sol_data) > 0 else 0)
        ax1.bar(x + i * width, values, width, label=polymer, color=colors[i], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Solvent', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Solubility (%)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Solubility Comparison at {temperature}°C', fontsize=15, fontweight='bold', pad=10)
    ax1.set_xticks(x + width * (len(polymer_list) - 1) / 2)
    # Truncate long solvent names and rotate for readability
    short_labels = [s[:20] + '...' if len(s) > 20 else s for s in solvents]
    ax1.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Heatmap - IMPROVED
    ax2 = fig.add_subplot(2, 2, 2)
    pivot = df.pivot(index=polymer_column, columns=solvent_column, values='avg_sol')
    # Determine annotation size based on data
    n_cells = pivot.shape[0] * pivot.shape[1]
    annot_size = 11 if n_cells <= 20 else 9 if n_cells <= 40 else 7
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                annot_kws={'size': annot_size}, linewidths=0.5,
                cbar_kws={'label': 'Solubility (%)', 'shrink': 0.8})
    ax2.set_title('Solubility Heatmap', fontsize=15, fontweight='bold', pad=10)
    ax2.set_xlabel('Solvent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Polymer', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=11)

    # Panel 3: Box plot - IMPROVED
    ax3 = fig.add_subplot(2, 2, 3)
    data_for_box = [df[df[polymer_column] == p]['avg_sol'].values for p in polymer_list]
    bp = ax3.boxplot(data_for_box, labels=polymer_list, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    ax3.set_xlabel('Polymer', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Solubility Distribution (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Solubility Distribution by Polymer', fontsize=15, fontweight='bold', pad=10)
    ax3.tick_params(axis='both', labelsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Rankings - IMPROVED
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    ranking_text = "POLYMER RANKINGS\n" + "="*25 + "\n\n"
    mean_sols = {p: df[df[polymer_column] == p]['avg_sol'].mean() for p in polymer_list}
    sorted_polymers = sorted(mean_sols.items(), key=lambda x: x[1], reverse=True)

    for i, (polymer, sol) in enumerate(sorted_polymers, 1):
        ranking_text += f"{i}. {polymer}: {sol:.2f}%\n"

    ax4.text(0.1, 0.85, ranking_text, transform=ax4.transAxes, fontsize=14,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    filepath = save_plot(fig, "comparison_dashboard", "matplotlib")

    output = f"✅ **Comparison Dashboard Created**\n\n"
    output += f"Temperature: {temperature}°C\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {len(solvents)}\n\n"
    output += get_plot_url(filepath)

    del df
    return output


@tool
@safe_tool_wrapper
async def plot_solvent_properties(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    solubility_column: str,
    polymer: str,
    property_to_plot: str,
    temperature_column: Optional[str] = None,
    temperature: Optional[float] = 25.0,
    min_solubility: Optional[float] = 0.0,
    max_solvents: int = 20,
    plot_type: str = "bar"
) -> str:
    """
    Plot solvent properties (BP, LogP, Energy, Cp) for solvents that dissolve a polymer.

    This tool combines solubility data with solvent properties to create visualizations
    showing physical/chemical characteristics of effective solvents.

    Args:
        table_name: Solubility database table name
        polymer_column: Column containing polymer names
        solvent_column: Column containing solvent names
        solubility_column: Column containing solubility values
        polymer: Polymer to analyze
        property_to_plot: Property to visualize ('bp', 'energy', 'logp', or 'cp')
        temperature_column: Column containing temperature (optional)
        temperature: Target temperature in °C (default: 25.0)
        min_solubility: Minimum solubility threshold (default: 0.0)
        max_solvents: Maximum number of solvents to show (default: 20)
        plot_type: Type of plot ('bar' or 'scatter') (default: 'bar')

    Returns:
        Plot description and URL

    Examples:
        - "Plot boiling points of solvents that dissolve PET" → property_to_plot='bp'
        - "Show energy costs for PS solvents" → property_to_plot='energy'
        - "Compare LogP values for PVDF solvents" → property_to_plot='logp'
    """
    # Validate property
    valid_properties = {'bp', 'energy', 'logp', 'cp'}
    property_lower = property_to_plot.lower().strip()
    if property_lower not in valid_properties:
        return f"❌ Invalid property '{property_to_plot}'. Must be one of: {', '.join(valid_properties)}"

    property_labels = {
        'bp': 'Boiling Point (°C)',
        'energy': 'Energy Cost (J/g)',
        'logp': 'LogP (Lipophilicity)',
        'cp': 'Heat Capacity Cp (J/g·K)'
    }

    # Query for solvents that dissolve the polymer
    temp_filter = ""
    if temperature_column and temperature is not None:
        temp_filter = f"AND {temperature_column} BETWEEN {temperature - 5} AND {temperature + 5}"

    query = f"""
    SELECT {solvent_column}, AVG({solubility_column}) as avg_solubility
    FROM {table_name}
    WHERE {polymer_column} = '{polymer}'
    AND {solubility_column} >= {min_solubility}
    {temp_filter}
    GROUP BY {solvent_column}
    ORDER BY avg_solubility DESC
    LIMIT {max_solvents}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"] or result["rows"] == 0:
        return f"❌ No solvents found for {polymer} with solubility >= {min_solubility}%"

    df = result["dataframe"]
    solvents = df[solvent_column].tolist()

    # Look up properties using robust matching
    solvent_table = get_solvent_table_name()
    if not solvent_table:
        return "❌ Solvent property database (solvent_data) not found. Cannot retrieve properties."

    logger.info(f"Looking up {property_lower} for {len(solvents)} solvents")
    props = await lookup_solvent_properties(solvents, solvent_table)

    # Extract the requested property
    solvent_data = []
    for solvent in solvents:
        if solvent in props and props[solvent][property_lower] is not None:
            solubility = df[df[solvent_column] == solvent]['avg_solubility'].values[0]
            solvent_data.append({
                'solvent': solvent,
                'property_value': props[solvent][property_lower],
                'solubility': solubility
            })

    if not solvent_data:
        return f"❌ No {property_lower.upper()} data found for solvents that dissolve {polymer}.\n\nThis may be due to naming mismatches between databases. Found {len(solvents)} solvents but none had {property_lower.upper()} data."

    # Sort by property value
    solvent_data.sort(key=lambda x: x['property_value'])

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    if plot_type.lower() == 'bar':
        # Bar chart
        names = [d['solvent'] for d in solvent_data]
        values = [d['property_value'] for d in solvent_data]
        solubilities = [d['solubility'] for d in solvent_data]

        # Color by solubility (darker = higher solubility)
        colors = plt.cm.YlOrRd(np.array(solubilities) / max(solubilities))

        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1.5)

        # Add solubility labels on top of bars
        for i, (bar, sol) in enumerate(zip(bars, solubilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{sol:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel(property_labels[property_lower], fontsize=13, fontweight='bold')
        ax.set_xlabel('Solvent', fontsize=13, fontweight='bold')
        ax.set_title(f'{property_labels[property_lower]} for Solvents Dissolving {polymer}\n(Color intensity = solubility)',
                    fontsize=15, fontweight='bold', pad=15)

    else:  # scatter plot
        # Scatter: property vs solubility
        values = [d['property_value'] for d in solvent_data]
        solubilities = [d['solubility'] for d in solvent_data]
        names = [d['solvent'] for d in solvent_data]

        ax.scatter(values, solubilities, s=150, alpha=0.6, edgecolors='black', linewidth=2, c=values, cmap='viridis')

        # Add labels for each point
        for x, y, name in zip(values, solubilities, names):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8, fontweight='bold')

        ax.set_xlabel(property_labels[property_lower], fontsize=13, fontweight='bold')
        ax.set_ylabel('Solubility (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{property_labels[property_lower]} vs Solubility for {polymer}',
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = save_plot(fig, f"solvent_properties_{property_lower}")
    plt.close(fig)  # Clean up
    gc.collect()

    # Build output message
    output = [f"✅ **Solvent Property Plot Created**\n"]
    output.append(f"**Polymer:** {polymer}")
    output.append(f"**Property:** {property_labels[property_lower]}")
    output.append(f"**Solvents analyzed:** {len(solvent_data)} (from {len(solvents)} total)")

    if len(solvents) > len(solvent_data):
        missing = len(solvents) - len(solvent_data)
        output.append(f"⚠️ Note: {missing} solvents had no {property_lower.upper()} data")

    output.append(f"\n**Top 5 by {property_labels[property_lower]}:**")
    for i, d in enumerate(solvent_data[:5], 1):
        prop_val = d['property_value']
        sol = d['solubility']
        output.append(f"{i}. **{d['solvent']}**: {prop_val:.2f} (solubility: {sol:.1f}%)")

    output.append(f"\n{get_plot_url(filepath)}")

    return "\n".join(output)


# ============================================================
# Greedy Separation Algorithm for Large Polymer Sets
# ============================================================

async def _greedy_separation_planning(
    polymer_list: list,
    temperature: float,
    top_k_solvents: int,
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str
) -> str:
    """
    Greedy algorithm for separation planning when n > 3 polymers.

    At each step, selects the polymer that can be most selectively separated
    from all remaining polymers. This is O(n²) instead of O(n!).

    Algorithm:
    1. For each remaining polymer, find the best solvent to separate it from others
    2. Pick the polymer with highest selectivity (easiest to separate)
    3. Remove it from the mixture and repeat
    """
    import math

    async_db = get_async_db()
    n_polymers = len(polymer_list)

    output = []
    output.append("# 🧪 Greedy Separation Planning\n")
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Count:** {n_polymers} polymers")
    output.append(f"**Algorithm:** Greedy (O(n²) ≈ {n_polymers**2} evaluations)")
    output.append(f"**vs Exhaustive:** {n_polymers}! = {math.factorial(n_polymers):,} permutations avoided")
    output.append(f"**Temperature:** {temperature}°C\n")

    output.append("## Algorithm Explanation\n")
    output.append("At each step, we select the polymer that can be **most selectively** separated")
    output.append("from all remaining polymers. This greedy approach finds a good (not necessarily optimal)")
    output.append("sequence efficiently.\n")

    # Track the greedy sequence
    remaining = list(polymer_list)
    sequence = []
    steps = []
    used_solvents = set()

    output.append("## Step-by-Step Greedy Selection\n")

    step_num = 0
    while len(remaining) > 1:
        step_num += 1
        output.append(f"### Step {step_num}: Evaluating {len(remaining)} candidates\n")
        output.append(f"**Remaining mixture:** {{{', '.join(remaining)}}}\n")

        # Evaluate each polymer as a potential target
        candidates = []

        for target in remaining:
            others = [p for p in remaining if p != target]

            # Build query to find best solvent for this target
            others_filter = "', '".join(others)
            all_polymers_filter = "', '".join([target] + others)

            query = f"""
            WITH target_sol AS (
                SELECT {solvent_column} as solvent, AVG({solubility_column}) as t_sol
                FROM {table_name}
                WHERE {polymer_column} = '{target}'
                AND {temperature_column} BETWEEN {temperature - 10} AND {temperature + 10}
                GROUP BY {solvent_column}
                HAVING AVG({solubility_column}) > 0
            ),
            others_max AS (
                SELECT {solvent_column} as solvent, MAX({solubility_column}) as o_max
                FROM {table_name}
                WHERE {polymer_column} IN ('{others_filter}')
                AND {temperature_column} BETWEEN {temperature - 10} AND {temperature + 10}
                GROUP BY {solvent_column}
            )
            SELECT
                t.solvent,
                t.t_sol as target_solubility,
                COALESCE(o.o_max, 0) as max_other_solubility,
                (t.t_sol - COALESCE(o.o_max, 0)) as selectivity
            FROM target_sol t
            LEFT JOIN others_max o ON LOWER(t.solvent) = LOWER(o.solvent)
            ORDER BY selectivity DESC
            LIMIT 1
            """

            try:
                result_df = await async_db.execute_async(query)
                if len(result_df) > 0:
                    row = result_df.iloc[0]
                    candidates.append({
                        'polymer': target,
                        'solvent': row['solvent'],
                        'selectivity': row['selectivity'],
                        'target_sol': row['target_solubility'],
                        'others': others
                    })
                else:
                    candidates.append({
                        'polymer': target,
                        'solvent': 'N/A',
                        'selectivity': -999,
                        'target_sol': 0,
                        'others': others
                    })
            except Exception as e:
                candidates.append({
                    'polymer': target,
                    'solvent': 'error',
                    'selectivity': -999,
                    'target_sol': 0,
                    'others': others,
                    'error': str(e)
                })

        # Show candidate evaluations
        output.append("| Polymer | Best Solvent | Selectivity |")
        output.append("|---------|--------------|-------------|")
        for c in sorted(candidates, key=lambda x: x['selectivity'], reverse=True):
            sel_str = f"{c['selectivity']:.1f}%" if c['selectivity'] > -900 else "N/A"
            output.append(f"| {c['polymer']} | {c['solvent']} | {sel_str} |")
        output.append("")

        # Pick the best candidate (highest selectivity)
        best = max(candidates, key=lambda x: x['selectivity'])

        if best['selectivity'] > -900:
            output.append(f"✅ **Selected: {best['polymer']}** with {best['solvent']} (selectivity: {best['selectivity']:.1f}%)\n")
        else:
            output.append(f"⚠️ **Selected: {best['polymer']}** (no solubility data available)\n")

        # Record the step
        sequence.append(best['polymer'])
        steps.append({
            'step': step_num,
            'target': best['polymer'],
            'solvent': best['solvent'],
            'selectivity': best['selectivity'],
            'remaining_before': list(remaining)
        })
        used_solvents.add(best['solvent'])
        remaining.remove(best['polymer'])

    # Add the last polymer
    if remaining:
        sequence.append(remaining[0])
        output.append(f"### Step {step_num + 1}: {remaining[0]} is isolated ✓\n")

    # Summary
    output.append("---\n")
    output.append("## 📋 Greedy Separation Sequence Summary\n")
    output.append(f"**Optimized Sequence:** {' → '.join(sequence)}\n")

    output.append("### Step-by-Step Protocol\n")
    output.append("| Step | Separate | Using Solvent | Selectivity |")
    output.append("|------|----------|---------------|-------------|")

    valid_steps = [s for s in steps if s['selectivity'] > -900]
    for s in steps:
        sel_str = f"{s['selectivity']:.1f}%" if s['selectivity'] > -900 else "N/A"
        output.append(f"| {s['step']} | {s['target']} | {s['solvent']} | {sel_str} |")
    output.append(f"| {len(steps) + 1} | {sequence[-1]} | (isolated) | ✓ |")
    output.append("")

    # Metrics
    if valid_steps:
        min_sel = min(s['selectivity'] for s in valid_steps)
        avg_sel = sum(s['selectivity'] for s in valid_steps) / len(valid_steps)
        unique_solvents = len(set(s['solvent'] for s in valid_steps if s['solvent'] != 'N/A'))

        output.append("### Metrics\n")
        output.append(f"- **Minimum selectivity:** {min_sel:.1f}%")
        output.append(f"- **Average selectivity:** {avg_sel:.1f}%")
        output.append(f"- **Unique solvents needed:** {unique_solvents}")
        output.append(f"- **Evaluations performed:** ~{n_polymers * (n_polymers + 1) // 2}")

    output.append("\n---\n")
    output.append("*Note: Greedy algorithm finds a good sequence efficiently but may not be globally optimal.*")
    output.append("*For ≤3 polymers, exhaustive search is used to find the true optimum.*")

    display = "\n".join(output)

    # Build structured data for programmatic access
    valid_steps = [s for s in steps if s['selectivity'] > -900]
    solvents_used = list(set(s['solvent'] for s in valid_steps if s['solvent'] != 'N/A'))

    structured_data = {
        "tool_name": "plan_sequential_separation",
        "success": True,
        "polymers_analyzed": polymer_list,
        "best_sequence": sequence,
        "solvents": solvents_used,
        "selectivities": [s['selectivity'] for s in valid_steps],
        "temperature": temperature,
        "algorithm_used": "greedy",
        "steps": [
            {
                "step": s['step'],
                "target": s['target'],
                "solvent": s['solvent'],
                "selectivity": s['selectivity']
            } for s in steps
        ],
        "min_selectivity": min(s['selectivity'] for s in valid_steps) if valid_steps else None,
        "max_selectivity": max(s['selectivity'] for s in valid_steps) if valid_steps else None,
        "coverage_complete": len(sequence) == len(polymer_list),
    }

    # Return structured JSON
    import json
    return json.dumps({"display": display, "data": structured_data})


# ============================================================
# Collect all tools
# ============================================================

@tool
@safe_tool_wrapper
async def plan_sequential_separation(
    polymers: str,
    temperature: float = 120.0,
    top_k_solvents: int = 5,
    create_decision_tree: bool = True,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Plan optimal sequential separation sequences for multiple polymers.

    Algorithm selection (automatic):
    - ≤3 polymers: Exhaustive search (all n! permutations)
    - >3 polymers: Greedy algorithm (O(n²) - fast and scalable)

    Creates a decision tree visualization showing separation paths.
    Enforces solvent diversity - each step uses a DIFFERENT solvent for physical feasibility.

    Parameters:
    - polymers: Comma-separated list of polymers to separate (e.g., "LDPE,HDPE,PP,PS")
    - temperature: Target temperature in °C (default: 120.0 - typical for polymer dissolution)
    - top_k_solvents: Number of top solvents to show per step (default: 5)
    - create_decision_tree: Whether to generate decision tree plot (default: True)

    WHEN TO USE:
    - "Plan sequential separation for LDPE, HDPE, PP, and PS"
    - "What's the best order to separate mixed plastics?"
    - "Design a multi-step polymer separation process"
    - Any number of polymers (automatically uses greedy for >3)

    Returns: Comprehensive separation plan with rankings and decision tree visualization
    """
    from itertools import permutations

    async_db = get_async_db()

    # Parse polymers
    polymer_list = [p.strip() for p in polymers.split(',') if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "Error: Need at least 2 polymers for separation planning."

    # For >3 polymers, use greedy algorithm instead of exhaustive search
    # (4! = 24 permutations is manageable, but 5! = 120 and 6! = 720 are too slow)
    USE_GREEDY = n_polymers > 3

    if USE_GREEDY:
        # Greedy algorithm: O(n²) instead of O(n!)
        return await _greedy_separation_planning(
            polymer_list, temperature, top_k_solvents,
            table_name, polymer_column, solvent_column,
            temperature_column, solubility_column
        )

    # Generate all permutations (only for ≤3 polymers)
    all_sequences = list(permutations(polymer_list))
    n_sequences = len(all_sequences)

    output = [f"# Sequential Separation Planning\n"]
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Number of possible sequences:** {n_polymers}! = {n_sequences}")
    output.append(f"**Temperature:** {temperature}°C")
    output.append(f"**Top solvents per step:** {top_k_solvents}\n")

    # List all sequences
    output.append("## All Possible Sequences\n")
    for i, seq in enumerate(all_sequences, 1):
        output.append(f"{i}. {' → '.join(seq)}")
    output.append("")

    # Minimum selectivity threshold for viable separation
    MIN_SELECTIVITY = 5.0

    # Async function to find top-k solvents for separating target from remaining polymers
    async def find_top_solvents(target: str, remaining: list, k: int = 5, used_solvents: set = None) -> list:
        """Find top-k solvents for separating target from remaining polymers (ASYNC).

        Enforces solvent diversity by excluding solvents already used in previous steps.
        """
        if used_solvents is None:
            used_solvents = set()

        if not remaining:
            return [{"solvent": "N/A", "selectivity": float('inf'), "target_sol": 100, "max_other": 0, "note": "Last polymer - no separation needed"}]

        all_polymers = [target] + remaining
        polymer_filter = "', '".join(all_polymers)

        query = f"""
        SELECT {solvent_column}, {polymer_column}, AVG({solubility_column}) as avg_sol
        FROM {table_name}
        WHERE {polymer_column} IN ('{polymer_filter}')
        AND {temperature_column} BETWEEN {temperature - 5} AND {temperature + 5}
        GROUP BY {solvent_column}, {polymer_column}
        """

        try:
            df = await async_db.execute_async(query)
        except Exception as e:
            return [{"solvent": "Error", "selectivity": 0, "target_sol": 0, "max_other": 0, "error": str(e)}]

        if len(df) == 0:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

        results = []
        for solvent in df[solvent_column].unique():
            solvent_data = df[df[solvent_column] == solvent]

            target_data = solvent_data[solvent_data[polymer_column] == target]
            if len(target_data) == 0:
                continue
            target_sol = target_data['avg_sol'].values[0]

            other_data = solvent_data[solvent_data[polymer_column].isin(remaining)]
            if len(other_data) == 0:
                max_other = 0
            else:
                max_other = other_data['avg_sol'].max()

            selectivity = target_sol - max_other
            results.append({
                "solvent": solvent,
                "selectivity": selectivity,
                "target_sol": target_sol,
                "max_other": max_other
            })

        # Sort by selectivity (descending)
        results.sort(key=lambda x: x["selectivity"], reverse=True)

        # CRITICAL: Filter out solvents already used in previous steps to ensure diversity
        if used_solvents:
            used_lower = {s.lower() for s in used_solvents}
            unused_results = [r for r in results if r["solvent"].lower() not in used_lower]
            if unused_results:
                results = unused_results
            else:
                # Mark reused solvents if no alternatives
                for r in results:
                    if r["solvent"].lower() in used_lower:
                        r["reused"] = True

        # Filter by minimum selectivity threshold
        viable_results = [r for r in results if r.get("selectivity", 0) >= MIN_SELECTIVITY]
        if viable_results:
            results = viable_results

        # Add solvent properties if available (ASYNC)
        solvent_table = get_solvent_table_name()
        if solvent_table and results:
            try:
                # Use async lookup for solvent properties
                solvent_names = [r["solvent"] for r in results[:k]]
                prop_lookup = await lookup_solvent_properties(solvent_names, solvent_table)

                # Add properties to results
                for r in results:
                    if r["solvent"] in prop_lookup:
                        r.update({k: v for k, v in prop_lookup[r["solvent"]].items() if v is not None})
            except Exception as e:
                logger.debug(f"Could not fetch solvent properties: {e}")

        return results[:k] if results else [{"solvent": "None found", "selectivity": 0, "target_sol": 0, "max_other": 0}]

    # Async function to analyze a single sequence with solvent diversity tracking
    async def analyze_sequence(sequence, seq_idx):
        """Analyze a single sequence, enforcing different solvents for each step."""
        seq_output = []
        seq_output.append(f"### Sequence {seq_idx}: {' → '.join(sequence)}\n")

        # Track solvents used in previous steps for diversity
        used_solvents = set()

        # Process steps SEQUENTIALLY to track used solvents
        total_min_selectivity = float('inf')
        seq_steps = []

        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])  # Polymers after this one

            # Find top solvents, excluding those already used
            top_solvents = await find_top_solvents(target, remaining, top_k_solvents, used_solvents)

            # Track the best solvent for this step (first in list)
            if top_solvents and top_solvents[0].get("solvent") not in ["N/A", "No data", "None found", "Error"]:
                used_solvents.add(top_solvents[0]["solvent"])

            seq_output.append(f"**Step {step}: Separate {target} from {{{', '.join(remaining)}}}**")

            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents
            }
            seq_steps.append(step_data)

            for rank, sol_info in enumerate(top_solvents, 1):
                if "error" in sol_info:
                    seq_output.append(f"  {rank}. Error: {sol_info['error']}")
                elif sol_info.get("solvent") in ["No data", "None found", "No viable solvent"]:
                    seq_output.append(f"  {rank}. No data available")
                else:
                    sel = sol_info.get("selectivity", 0)
                    symbol = "✅" if sel > 10 else "⚠️" if sel > 0 else "❌"
                    reused_marker = " *(REUSED)*" if sol_info.get("reused") else ""
                    target_sol = sol_info.get('target_sol', 0)
                    max_other = sol_info.get('max_other', 0)
                    line = f"  {rank}. {symbol} **{sol_info['solvent']}**{reused_marker}: selectivity={sel:.1f}% (target={target_sol:.1f}%, max_other={max_other:.1f}%)"

                    # Add properties if available
                    props = []
                    if sol_info.get('logp') is not None:
                        toxicity = "Low" if sol_info['logp'] < 0 else "Med" if sol_info['logp'] < 2 else "High"
                        props.append(f"LogP:{sol_info['logp']:.1f}({toxicity})")
                    if sol_info.get('energy') is not None:
                        props.append(f"Energy:{sol_info['energy']:.0f}J/g")
                    if sol_info.get('bp') is not None:
                        props.append(f"BP:{sol_info['bp']:.0f}°C")

                    if props:
                        line += f" | {' '.join(props)}"

                    seq_output.append(line)

            # Track minimum selectivity across best solvents
            if top_solvents and "selectivity" in top_solvents[0]:
                best_selectivity = top_solvents[0]["selectivity"]
                total_min_selectivity = min(total_min_selectivity, best_selectivity)

            seq_output.append("")

        # Final polymer
        seq_output.append(f"**Step {len(sequence)}: {sequence[-1]} is isolated** ✅\n")

        # Solvent diversity summary
        best_solvents = [s["solvents"][0]["solvent"] for s in seq_steps
                        if s["solvents"] and s["solvents"][0].get("solvent") not in ["N/A", "No data", "None found", "Error"]]
        unique_solvents = set(best_solvents)
        if len(best_solvents) > len(unique_solvents):
            seq_output.append(f"⚠️ **Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps (some reused)\n")
        else:
            seq_output.append(f"✅ **Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(best_solvents)} steps\n")

        seq_output.append("---\n")

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps,
            "output": seq_output,
            "unique_solvents": len(unique_solvents)
        }

    # Analyze all sequences in parallel with limited concurrency
    output.append("## Detailed Analysis of Each Sequence\n")
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent sequence analyses

    async def analyze_with_limit(sequence, seq_idx):
        async with semaphore:
            return await analyze_sequence(sequence, seq_idx)

    # Execute all sequence analyses in parallel
    sequence_results = await asyncio.gather(*[
        analyze_with_limit(seq, idx)
        for idx, seq in enumerate(all_sequences, 1)
    ])

    # Extract results and build output
    sequence_scores = []
    sequence_details = []
    for result in sequence_results:
        sequence_scores.append({
            "sequence": result["sequence"],
            "min_selectivity": result["min_selectivity"],
            "steps": result["steps"]
        })
        sequence_details.append(result["steps"])
        output.extend(result["output"])
    
    # Rank sequences by minimum selectivity (higher is better)
    sequence_scores.sort(key=lambda x: x["min_selectivity"], reverse=True)
    
    output.append("## Sequence Ranking (by worst-case selectivity)\n")
    output.append("*Higher minimum selectivity = more robust separation*\n")
    
    for rank, score_data in enumerate(sequence_scores[:10], 1):  # Top 10
        seq_str = ' → '.join(score_data["sequence"])
        min_sel = score_data["min_selectivity"]
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        output.append(f"{symbol} **{seq_str}** (min selectivity: {min_sel:.1f}%)")
    
    output.append("")

    # Create visualization - SHOW TOP SEQUENCE ONLY (clear and easy to read)
    if create_decision_tree and sequence_scores:
        output.append("## Top Recommended Separation Sequence\n")

        try:
            # Get the best sequence
            best_seq = sequence_scores[0]
            sequence = best_seq["sequence"]
            steps = best_seq["steps"]
            min_sel = best_seq["min_selectivity"]

            def get_color(selectivity):
                """Get color based on selectivity percentage (0-100 scale)."""
                if selectivity > 30:
                    return '#2ecc71'  # Green - excellent
                elif selectivity > 10:
                    return '#f1c40f'  # Yellow - good
                elif selectivity > 0:
                    return '#e67e22'  # Orange - marginal
                else:
                    return '#e74c3c'  # Red - poor

            # Create figure - VERTICAL FLOWCHART (easy to read top-to-bottom)
            n_steps = len(steps)
            # Increase spacing for more polymers to prevent overlap
            fig_height = max(3 + n_steps * 2.5, 8)
            fig_width = 12
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Title with ranking info
            ax.set_title(
                f'RECOMMENDED SEPARATION SEQUENCE (Rank #1 of {len(sequence_scores)})\n' +
                f'Sequence: {" → ".join(sequence)} | Min Selectivity: {min_sel:.1f}% | Temp: {temperature}°C',
                fontsize=16, fontweight='bold', pad=20
            )

            ax.set_xlim(0, 10)
            # Add extra space at top to prevent overlap with solvent boxes
            ax.set_ylim(-0.5, n_steps + 2.5)
            ax.axis('off')

            # Starting mixture at top (moved higher to prevent overlap)
            y_pos = n_steps + 1.5
            ax.add_patch(plt.Rectangle((2, y_pos - 0.3), 6, 0.6,
                                       facecolor='#3498db', edgecolor='black', linewidth=2))
            ax.text(5, y_pos, f'STARTING MIXTURE: {", ".join(polymer_list)}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')

            # Draw each separation step
            for idx, step in enumerate(steps):
                y_pos = n_steps - idx
                target = step["target"]
                remaining = step["remaining"]
                top_solvent = step["solvents"][0] if step["solvents"] else {"solvent": "N/A", "selectivity": 0}
                solvent_name = top_solvent["solvent"]
                selectivity = top_solvent.get("selectivity", 0)
                color = get_color(selectivity)

                # Arrow down with solvent info
                ax.annotate('', xy=(3.5, y_pos + 0.4), xytext=(3.5, y_pos + 0.9),
                           arrowprops=dict(arrowstyle='->', lw=4, color=color))

                # Step box (left side - narrower to avoid overlap)
                ax.add_patch(plt.Rectangle((1.2, y_pos - 0.35), 4.6, 0.7,
                                          facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.3))

                # Step number and target
                ax.add_patch(plt.Circle((1.9, y_pos), 0.25, facecolor=color, edgecolor='black', linewidth=2))
                ax.text(1.9, y_pos, str(idx + 1), ha='center', va='center',
                       fontsize=14, fontweight='bold', color='white')

                # Separated polymer (large text)
                ax.text(2.7, y_pos, f'SEPARATE: {target}',
                       ha='left', va='center', fontsize=14, fontweight='bold')

                # Solvent label box (right side - clear separation from step box)
                ax.add_patch(plt.Rectangle((6.2, y_pos + 0.55), 3.3, 0.5,
                                          facecolor='white', edgecolor=color, linewidth=2))
                ax.text(7.85, y_pos + 0.8, f'Solvent: {solvent_name}',
                       ha='center', va='center', fontsize=11, fontweight='bold')
                ax.text(7.85, y_pos + 0.6, f'Selectivity: {selectivity:.1f}%',
                       ha='center', va='center', fontsize=10, color=color, fontweight='bold')

                # Remaining polymers (positioned below step box, no overlap)
                if remaining:
                    remaining_text = f'Remaining: {", ".join(remaining)}'
                    ax.text(5.7, y_pos - 0.15, remaining_text,
                           ha='right', va='center', fontsize=10, color='#34495e',
                           style='italic', weight='bold')
                else:
                    ax.text(5.7, y_pos - 0.15, '(Last polymer - isolated)',
                           ha='right', va='center', fontsize=10, color='#27ae60',
                           style='italic', weight='bold')

            # Final result box at bottom
            ax.add_patch(plt.Rectangle((2, -0.3), 6, 0.6,
                                      facecolor='#2ecc71', edgecolor='black', linewidth=2.5))
            ax.text(5, 0, '✓ ALL POLYMERS SEPARATED',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')

            # Legend
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                          markersize=15, markeredgecolor='black', linewidth=2, label='Excellent (>30%)'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#f1c40f',
                          markersize=15, markeredgecolor='black', linewidth=2, label='Good (10-30%)'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22',
                          markersize=15, markeredgecolor='black', linewidth=2, label='Marginal (0-10%)'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                          markersize=15, markeredgecolor='black', linewidth=2, label='Poor (<0%)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                     frameon=True, fancybox=True, title='Selectivity Quality', title_fontsize=12)

            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            filepath = save_plot(fig, f"separation_sequence_rank1")
            plt.close(fig)  # Clean up figure to prevent memory leaks
            output.append(f"📊 Visualization saved: {get_plot_url(filepath)}\n")

            # Add note about alternative sequences
            if len(sequence_scores) > 1:
                output.append(f"💡 **Note:** This shows the top-ranked sequence. There are {len(sequence_scores) - 1} other possible sequences.")
                output.append(f"    To view alternatives, ask: 'Show me the 2nd best sequence' or 'Show me {polymer_list[1]}-first separation'")

        except Exception as e:
            logger.error(f"Decision tree error: {e}", exc_info=True)
            output.append(f"⚠️ Could not create visualization: {e}")

    # Summary recommendations
    output.append("\n## Recommendations\n")
    if sequence_scores and sequence_scores[0]["min_selectivity"] > 10:
        best = sequence_scores[0]
        output.append(f"✅ **Best sequence:** {' → '.join(best['sequence'])}")
        output.append(f"   - Minimum selectivity: {best['min_selectivity']:.1f}%")
        output.append(f"   - All steps have positive selectivity")
        if len(sequence_scores) > 1:
            output.append(f"\n📋 **Alternative sequences available:** {len(sequence_scores) - 1} more options")
            output.append(f"   Ask to see specific sequences (e.g., 'Show 2nd best' or 'Show {polymer_list[0]}-first')")
    elif sequence_scores:
        output.append("⚠️ **No sequence has all high-selectivity steps.**")
        output.append("Consider:")
        output.append("  - Exploring different temperatures")
        output.append("  - Using multi-stage extraction")
        output.append("  - Combining solvents")
    
    return "\n".join(output)


@tool
@safe_tool_wrapper
async def analyze_integrated_separation(
    polymers: str,
    rank_by: str = "selectivity",
    top_k: int = 10,
    temperature_min: float = 25.0,
    temperature_max: float = 160.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Comprehensive multi-polymer separation analysis with optimal temperatures and integrated properties.

    Analyzes ALL possible separation sequences for multiple polymers, finding the OPTIMAL
    TEMPERATURE for each separation step. Includes selectivity, safety (G-score),
    cost (energy), toxicity (LogP), and boiling point for each recommended solvent.

    Key Features:
    - Searches across ALL temperatures (25-160°C) for each step
    - Finds optimal temp-solvent combination per separation
    - Includes GSK safety scores, energy costs, toxicity, boiling points
    - Ranks by selectivity, cost, safety, or boiling point
    - Creates visualization showing different temps per step

    Args:
        polymers: Comma-separated list of polymers to separate (e.g., "LDPE,EVOH,PET,PVC")
        rank_by: How to rank solvents - 'selectivity', 'cost'/'energy', 'safety'/'gscore', 'toxicity'/'logp', 'bp'/'boiling'
        top_k: Number of top solvents to show per step (default: 10)
        temperature_min: Minimum temperature to search (default: 25°C)
        temperature_max: Maximum temperature to search (default: 160°C)
        table_name: Solubility data table (default: common_solvents_database)
        polymer_column: Column with polymer names (default: polymer)
        solvent_column: Column with solvent names (default: solvent)
        temperature_column: Column with temperature values (default: temperature___c_)
        solubility_column: Column with solubility values (default: solubility____)

    Returns:
        Comprehensive separation plan with optimal temps, solvents, and all properties
    """
    from itertools import permutations

    async_db = get_async_db()

    # Parse polymers
    polymer_list = [p.strip().upper() for p in polymers.split(',') if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "❌ Need at least 2 polymers for separation analysis."

    # For >3 polymers, recommend using greedy-based plan_sequential_separation instead
    if n_polymers > 3:
        return f"⚠️ For {n_polymers} polymers, use `plan_sequential_separation` which uses efficient greedy algorithm. This exhaustive analysis tool is limited to ≤3 polymers."

    # Get available temperatures from database
    temp_query = f"""
    SELECT DISTINCT {temperature_column} as temp
    FROM {table_name}
    WHERE {temperature_column} BETWEEN {temperature_min} AND {temperature_max}
    ORDER BY temp
    """
    try:
        temp_df = await async_db.execute_async(temp_query)
        available_temps = sorted(temp_df['temp'].unique())
    except Exception as e:
        return f"❌ Error getting temperatures: {e}"

    if not available_temps:
        return f"❌ No temperature data found between {temperature_min}°C and {temperature_max}°C"

    output = [f"# 🔬 Integrated Multi-Polymer Separation Analysis\n"]
    output.append(f"**Polymers:** {', '.join(polymer_list)}")
    output.append(f"**Temperature Range:** {temperature_min}°C - {temperature_max}°C ({len(available_temps)} temperatures)")
    output.append(f"**Ranking Criterion:** {rank_by}")
    output.append(f"**Number of Sequences:** {n_polymers}! = {len(list(permutations(polymer_list)))}\n")

    # Helper to get solvent properties including GSK G-score
    async def get_full_properties(solvent_names: list) -> dict:
        """Get all properties for solvents including GSK G-scores."""
        prop_lookup = {}

        # Get basic properties from solvent_data
        solvent_table = get_solvent_table_name()
        if solvent_table:
            try:
                prop_dict = await lookup_solvent_properties(solvent_names, solvent_table)
                if prop_dict:
                    prop_lookup.update(prop_dict)
            except Exception:
                pass

        # Get GSK G-scores
        try:
            solvent_filter = "', '".join(solvent_names)
            gscore_query = f"""
            SELECT solvent_common_name, g_score, classification
            FROM gsk_dataset
            WHERE LOWER(solvent_common_name) IN ('{solvent_filter.lower()}')
            """
            gscore_df = await async_db.execute_async(gscore_query)
            if len(gscore_df) > 0:
                for _, row in gscore_df.iterrows():
                    name = row['solvent_common_name']
                    # Match by lowercase
                    for orig_name in solvent_names:
                        if orig_name.lower() == name.lower():
                            if orig_name not in prop_lookup:
                                prop_lookup[orig_name] = {}
                            prop_lookup[orig_name]['g_score'] = row['g_score']
                            prop_lookup[orig_name]['gsk_class'] = row['classification']
                            break
        except Exception:
            pass

        return prop_lookup

    # Minimum selectivity threshold - solvents with lower selectivity are not practical
    MIN_SELECTIVITY_THRESHOLD = 5.0  # At least 5% selectivity required

    # Find optimal separation for target polymer from remaining at ANY temperature
    async def find_optimal_separation(target: str, remaining: list, used_solvents: set = None) -> dict:
        """Find the best temperature-solvent combination for separating target from remaining.

        Args:
            target: Polymer to dissolve/separate
            remaining: List of polymers that should NOT dissolve
            used_solvents: Set of solvents already used in previous steps (to enforce diversity)
        """
        if used_solvents is None:
            used_solvents = set()

        if not remaining:
            return {
                "solvent": "N/A",
                "temperature": 0,
                "selectivity": float('inf'),
                "target_sol": 100,
                "max_other": 0,
                "note": "Last polymer - no separation needed"
            }

        all_polymers = [target] + remaining
        polymer_filter = "', '".join(all_polymers)

        # Query across ALL temperatures
        query = f"""
        SELECT {solvent_column}, {polymer_column}, {temperature_column} as temp,
               AVG({solubility_column}) as avg_sol
        FROM {table_name}
        WHERE {polymer_column} IN ('{polymer_filter}')
        AND {temperature_column} BETWEEN {temperature_min} AND {temperature_max}
        GROUP BY {solvent_column}, {polymer_column}, {temperature_column}
        """

        try:
            df = await async_db.execute_async(query)
        except Exception as e:
            return {"solvent": "Error", "temperature": 0, "selectivity": 0, "error": str(e)}

        if len(df) == 0:
            return {"solvent": "No data", "temperature": 0, "selectivity": 0}

        # Analyze each temperature-solvent combination
        results = []
        for temp in df['temp'].unique():
            temp_df = df[df['temp'] == temp]

            for solvent in temp_df[solvent_column].unique():
                solvent_data = temp_df[temp_df[solvent_column] == solvent]

                target_data = solvent_data[solvent_data[polymer_column] == target]
                if len(target_data) == 0:
                    continue
                target_sol = target_data['avg_sol'].values[0]

                other_data = solvent_data[solvent_data[polymer_column].isin(remaining)]
                if len(other_data) == 0:
                    max_other = 0
                else:
                    max_other = other_data['avg_sol'].max()

                selectivity = target_sol - max_other

                results.append({
                    "solvent": solvent,
                    "temperature": temp,
                    "selectivity": selectivity,
                    "target_sol": target_sol,
                    "max_other": max_other
                })

        if not results:
            return {"solvent": "None found", "temperature": 0, "selectivity": 0}

        # CRITICAL: Filter out solvents already used in previous steps to ensure diversity
        # This prevents the physically-impossible scenario of using the same solvent for all steps
        if used_solvents:
            # First try: exclude used solvents entirely
            unused_results = [r for r in results if r["solvent"].lower() not in {s.lower() for s in used_solvents}]
            if unused_results:
                results = unused_results
            else:
                # Fallback: if all good solvents are used, keep results but mark as reused
                for r in results:
                    if r["solvent"].lower() in {s.lower() for s in used_solvents}:
                        r["reused_solvent"] = True

        # Filter by minimum selectivity threshold
        results = [r for r in results if r.get('selectivity', 0) >= MIN_SELECTIVITY_THRESHOLD]
        if not results:
            return {"solvent": "No viable solvent", "temperature": 0, "selectivity": 0,
                    "note": f"No solvent found with selectivity >= {MIN_SELECTIVITY_THRESHOLD}%"}

        # Get properties for all solvents found
        solvent_names = list(set(r["solvent"] for r in results))
        prop_lookup = await get_full_properties(solvent_names)

        # Add properties to results
        for r in results:
            if r["solvent"] in prop_lookup:
                r.update(prop_lookup[r["solvent"]])

        # Sort based on rank_by criterion
        rank_lower = rank_by.lower()
        if rank_lower in ['cost', 'energy']:
            # Filter to positive selectivity, then sort by energy (lower = better)
            valid = [r for r in results if r.get('selectivity', 0) > 0 and r.get('energy') is not None]
            if valid:
                valid.sort(key=lambda x: x['energy'])
                return valid[0]
        elif rank_lower in ['safety', 'gscore', 'g_score']:
            # Filter to positive selectivity, sort by G-score (higher = safer)
            valid = [r for r in results if r.get('selectivity', 0) > 0 and r.get('g_score') is not None]
            if valid:
                valid.sort(key=lambda x: x['g_score'], reverse=True)
                return valid[0]
        elif rank_lower in ['toxicity', 'logp']:
            # Filter to positive selectivity, sort by LogP (lower = less toxic)
            valid = [r for r in results if r.get('selectivity', 0) > 0 and r.get('logp') is not None]
            if valid:
                valid.sort(key=lambda x: x['logp'])
                return valid[0]
        elif rank_lower in ['bp', 'boiling', 'boiling_point']:
            # Filter to positive selectivity, sort by boiling point
            valid = [r for r in results if r.get('selectivity', 0) > 0 and r.get('bp') is not None]
            if valid:
                valid.sort(key=lambda x: x['bp'])
                return valid[0]

        # Default: sort by selectivity (higher = better)
        results.sort(key=lambda x: x.get('selectivity', 0), reverse=True)
        return results[0]

    # Analyze a single sequence
    async def analyze_sequence(sequence: tuple) -> dict:
        """Analyze one separation sequence finding optimal temp for each step.

        Tracks used solvents to ensure each step uses a DIFFERENT solvent,
        which is required for physically feasible sequential separation.
        """
        steps = []
        total_score = 0
        used_solvents = set()  # Track solvents used in previous steps

        for step_idx, target in enumerate(sequence[:-1]):
            remaining = list(sequence[step_idx + 1:])
            best = await find_optimal_separation(target, remaining, used_solvents)

            # Track the solvent used in this step (if valid)
            if best.get('solvent') and best['solvent'] not in ['None found', 'No data', 'Error', 'N/A', 'No viable solvent']:
                used_solvents.add(best['solvent'])

            step_data = {
                "step": step_idx + 1,
                "target": target,
                "remaining": remaining,
                "best": best
            }
            steps.append(step_data)

            # Score based on selectivity (or other criteria)
            sel = best.get('selectivity', 0)
            if sel != float('inf'):
                total_score += sel

        # Add final polymer
        steps.append({
            "step": len(sequence),
            "target": sequence[-1],
            "remaining": [],
            "best": {"solvent": "N/A", "temperature": 0, "selectivity": float('inf'), "note": "Isolated"}
        })

        # Calculate minimum selectivity (bottleneck)
        min_sel = min(
            s["best"].get("selectivity", 0)
            for s in steps[:-1]
            if s["best"].get("selectivity", 0) != float('inf')
        ) if len(steps) > 1 else 0

        return {
            "sequence": sequence,
            "steps": steps,
            "total_score": total_score,
            "min_selectivity": min_sel
        }

    # Generate all permutations and analyze in parallel
    all_sequences = list(permutations(polymer_list))

    output.append("## 📋 Analyzing All Sequences...\n")

    # Analyze all sequences with limited concurrency
    semaphore = asyncio.Semaphore(5)

    async def analyze_with_limit(seq):
        async with semaphore:
            return await analyze_sequence(seq)

    all_results = await asyncio.gather(*[analyze_with_limit(seq) for seq in all_sequences])

    # Sort by minimum selectivity (bottleneck approach)
    all_results.sort(key=lambda x: x["min_selectivity"], reverse=True)

    # Show top 3 sequences in detail
    output.append("## 🏆 Top 3 Recommended Separation Sequences\n")

    for rank, result in enumerate(all_results[:3], 1):
        seq = result["sequence"]
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"

        output.append(f"### {medal} Rank #{rank}: {' → '.join(seq)}")
        output.append(f"**Minimum Selectivity (Bottleneck):** {result['min_selectivity']:.1f}%\n")

        for step in result["steps"][:-1]:  # Exclude final "isolated" step
            best = step["best"]
            target = step["target"]
            remaining = step["remaining"]

            sel = best.get('selectivity', 0)
            symbol = "✅" if sel > 30 else "🟡" if sel > 10 else "⚠️" if sel > 0 else "❌"

            output.append(f"**Step {step['step']}: Separate {target} from {{{', '.join(remaining)}}}**")

            # Check for warnings
            solvent_name = best.get('solvent', 'N/A')
            if best.get('reused_solvent'):
                output.append(f"  ⚠️ **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f}°C** *(REUSED - limited options)*")
            elif best.get('note'):
                output.append(f"  ❌ **Solvent:** {solvent_name} - {best.get('note')}")
            else:
                output.append(f"  {symbol} **Solvent:** {solvent_name} @ **{best.get('temperature', 0):.0f}°C**")
            output.append(f"  - Selectivity: {sel:.1f}% (target: {best.get('target_sol', 0):.1f}%, max_other: {best.get('max_other', 0):.1f}%)")

            # Properties
            props = []
            if best.get('g_score') is not None:
                gs = best['g_score']
                rating = "✅ Excellent" if gs >= 8 else "🟢 Good" if gs >= 6 else "🟡 Problematic" if gs >= 4 else "🔴 Hazardous"
                props.append(f"G-Score: {gs:.1f}/10 ({rating})")
            if best.get('logp') is not None:
                lp = best['logp']
                tox = "Low" if lp < 0 else "Medium" if lp < 2 else "High"
                props.append(f"LogP: {lp:.2f} ({tox} toxicity)")
            if best.get('energy') is not None:
                props.append(f"Energy: {best['energy']:.1f} J/g")
            if best.get('bp') is not None:
                props.append(f"BP: {best['bp']:.0f}°C")

            if props:
                output.append(f"  - Properties: {' | '.join(props)}")
            output.append("")

        # Final polymer
        output.append(f"**Step {len(seq)}: {seq[-1]} is isolated** ✅\n")

        # Summary: check for solvent diversity
        solvents_used = [s["best"].get("solvent", "N/A") for s in result["steps"][:-1]
                        if s["best"].get("solvent") not in ["N/A", "None found", "No data", "Error", "No viable solvent"]]
        unique_solvents = set(solvents_used)
        if len(solvents_used) > len(unique_solvents):
            duplicates = [s for s in unique_solvents if solvents_used.count(s) > 1]
            output.append(f"⚠️ **Warning:** Solvent(s) {duplicates} used multiple times. This may indicate limited data or challenging separation.\n")
        else:
            output.append(f"✅ **Solvent Diversity:** {len(unique_solvents)} unique solvents for {len(solvents_used)} steps\n")

        output.append("---\n")

    # Create visualization for top sequence
    try:
        best_result = all_results[0]
        seq = best_result["sequence"]
        steps = best_result["steps"][:-1]  # Exclude final step

        n_steps = len(steps)
        fig_height = max(5 + n_steps * 3.5, 12)
        fig, ax = plt.subplots(figsize=(16, fig_height), dpi=150)

        ax.set_title(
            f'OPTIMAL SEPARATION SEQUENCE: {" → ".join(seq)}\n' +
            f'Ranked by: {rank_by} | Min Selectivity: {best_result["min_selectivity"]:.1f}%',
            fontsize=18, fontweight='bold', pad=25
        )

        ax.set_xlim(0, 14)
        ax.set_ylim(-1.5, n_steps + 3.5)
        ax.axis('off')

        def get_color(selectivity):
            if selectivity > 30:
                return '#2ecc71'  # Green
            elif selectivity > 10:
                return '#f1c40f'  # Yellow
            elif selectivity > 0:
                return '#e67e22'  # Orange
            else:
                return '#e74c3c'  # Red

        # Starting mixture
        y_pos = n_steps + 2
        ax.add_patch(plt.Rectangle((1.5, y_pos - 0.5), 11, 1.0,
                                   facecolor='#3498db', edgecolor='black', linewidth=2.5))
        ax.text(7, y_pos, f'MIXTURE: {", ".join(polymer_list)}',
               ha='center', va='center', fontsize=16, fontweight='bold', color='white')

        # Draw each step
        for idx, step in enumerate(steps):
            y_pos = n_steps + 1 - idx
            best = step["best"]
            target = step["target"]
            remaining = step["remaining"]
            sel = best.get('selectivity', 0)
            temp = best.get('temperature', 0)
            color = get_color(sel)

            # Arrow
            ax.annotate('', xy=(3.5, y_pos + 0.4), xytext=(3.5, y_pos + 0.9),
                       arrowprops=dict(arrowstyle='->', lw=3.5, color=color))

            # Step box (left side)
            ax.add_patch(plt.Rectangle((1, y_pos - 0.6), 5.5, 1.2,
                                      facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.25))

            # Step number circle
            ax.add_patch(plt.Circle((1.6, y_pos), 0.35, facecolor=color, edgecolor='black', linewidth=2.5))
            ax.text(1.6, y_pos, str(idx + 1), ha='center', va='center',
                   fontsize=15, fontweight='bold', color='white')

            # Target polymer
            ax.text(2.4, y_pos + 0.25, f'SEPARATE: {target}',
                   ha='left', va='center', fontsize=14, fontweight='bold')
            ax.text(2.4, y_pos - 0.25, f'From: {", ".join(remaining)}',
                   ha='left', va='center', fontsize=12, color='#333')

            # Solvent & Temperature box (right side) - LARGER for better readability
            ax.add_patch(plt.Rectangle((7, y_pos - 0.6), 5.5, 1.2,
                                      facecolor='white', edgecolor=color, linewidth=2.5))
            ax.text(9.75, y_pos + 0.25, f'{best.get("solvent", "N/A")}',
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(9.75, y_pos - 0.15, f'{temp:.0f}°C  |  Selectivity: {sel:.1f}%',
                   ha='center', va='center', fontsize=13, color=color, fontweight='bold')

            # Properties below - LARGER AND BOLDER for publication quality
            props_text = []
            if best.get('g_score') is not None:
                props_text.append(f"G-Score: {best['g_score']:.0f}")
            if best.get('energy') is not None:
                props_text.append(f"Energy: {best['energy']:.0f} J/g")
            if best.get('bp') is not None:
                props_text.append(f"BP: {best['bp']:.0f}°C")
            if props_text:
                ax.text(9.75, y_pos - 0.52, '  |  '.join(props_text),
                       ha='center', va='top', fontsize=12, fontweight='semibold', color='#222')

        # Final result
        ax.add_patch(plt.Rectangle((1.5, -0.5), 11, 1.0,
                                  facecolor='#2ecc71', edgecolor='black', linewidth=2.5))
        ax.text(7, 0, f'✓ ALL POLYMERS SEPARATED',
               ha='center', va='center', fontsize=16, fontweight='bold', color='white')

        # Legend - larger and more readable
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                      markersize=14, label='Excellent (>30%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#f1c40f',
                      markersize=14, label='Good (10-30%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22',
                      markersize=14, label='Marginal (0-10%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                      markersize=14, label='Poor (<0%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                 framealpha=0.95, edgecolor='#333', fancybox=True)

        plt.tight_layout(pad=2.0)
        filepath = save_plot(fig, "integrated_separation_analysis")
        plt.close(fig)

        output.append(f"## 📊 Visualization\n")
        output.append(f"![Separation Sequence]({get_plot_url(filepath)})\n")

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        output.append(f"⚠️ Could not create visualization: {e}\n")

    # Summary and recommendations
    output.append("## 📝 Summary & Recommendations\n")

    best = all_results[0]
    output.append(f"**Best Sequence:** {' → '.join(best['sequence'])}")
    output.append(f"**Bottleneck Selectivity:** {best['min_selectivity']:.1f}%\n")

    output.append("**Optimal Conditions per Step:**")
    for step in best["steps"][:-1]:
        b = step["best"]
        output.append(f"  - Step {step['step']}: {step['target']} → {b.get('solvent', 'N/A')} @ {b.get('temperature', 0):.0f}°C")

    # Alternative ranking recommendations
    if rank_by.lower() == 'selectivity':
        output.append("\n💡 **Tip:** Re-run with `rank_by='cost'` for cheapest solvents, `rank_by='safety'` for safest (highest G-score), or `rank_by='toxicity'` for least toxic (lowest LogP).")

    return "\n".join(output)


@tool
@safe_tool_wrapper
async def get_solubility_for_solvents(
    polymer: str,
    solvents: str,
    temperature: float = 90.0,
    temperature_range: float = 10.0
) -> str:
    """
    Get solubility data for SPECIFIC solvents with a given polymer.

    Use this when the user asks to compare specific solvents by name,
    rather than searching for the best solvents.

    Parameters:
    - polymer: The polymer name (e.g., "EVOH", "LDPE", "PET")
    - solvents: Comma-separated list of specific solvents (e.g., "DMSO,DMF,NMP,ethylene glycol")
    - temperature: Target temperature in °C (default: 90)
    - temperature_range: Temperature tolerance +/- (default: 10°C)

    WHEN TO USE:
    - "Get solubility of EVOH in DMSO, DMF, and NMP at 90°C"
    - "Compare these 5 solvents for PET: toluene, xylene, benzene, acetone, ethanol"
    - "What's the solubility of LDPE in heptane and hexane?"

    Returns: Solubility data for each specified solvent
    """
    async_db = get_async_db()
    polymer_upper = polymer.strip().upper()
    solvent_list = [s.strip().lower() for s in solvents.split(',')]

    output = [f"SOLUBILITY DATA: {polymer_upper} in specific solvents at {temperature}°C\n"]

    results = []
    missing_solvents = []

    for solvent in solvent_list:
        # Try exact match first
        query = f"""
        SELECT solvent, AVG(solubility____) as avg_solubility,
               MIN(solubility____) as min_sol, MAX(solubility____) as max_sol,
               COUNT(*) as data_points
        FROM common_solvents_database
        WHERE UPPER(polymer) = '{polymer_upper}'
        AND LOWER(solvent) LIKE '%{solvent}%'
        AND temperature___c_ BETWEEN {temperature - temperature_range} AND {temperature + temperature_range}
        GROUP BY solvent
        """

        try:
            df = await async_db.execute_async(query)
            if len(df) > 0:
                for _, row in df.iterrows():
                    results.append({
                        'solvent': row['solvent'],
                        'solubility': row['avg_solubility'],
                        'min_sol': row['min_sol'],
                        'max_sol': row['max_sol'],
                        'data_points': row['data_points']
                    })
            else:
                missing_solvents.append(solvent)
        except Exception as e:
            logger.warning(f"Error querying {solvent}: {e}")
            missing_solvents.append(solvent)

    if results:
        output.append("RESULTS:")
        for r in sorted(results, key=lambda x: x['solubility'], reverse=True):
            output.append(f"{r['solvent']}: {r['solubility']:.1f}% (range: {r['min_sol']:.1f}-{r['max_sol']:.1f}%, n={r['data_points']})")
    else:
        output.append("No solubility data found for any of the specified solvents.")

    if missing_solvents:
        output.append(f"\nNO DATA for: {', '.join(missing_solvents)}")
        output.append("(These solvents may not have solubility data for this polymer at this temperature)")

    return "\n".join(output)


@tool
@safe_tool_wrapper
async def analyze_polymer_dissolution(
    polymer: str,
    temperature: float = 120.0,
    min_solubility: float = 5.0,
    rank_by: str = "solubility",
    top_k: int = 15,
    temperature_range: float = 10.0,
    adaptive: bool = True
) -> str:
    """
    Find solvents that dissolve a polymer at a given temperature AND show their properties.

    This is the PRIMARY tool for questions like:
    - "What solvents dissolve PET at 120°C?"
    - "Find solvents for LDPE ranked by boiling point"
    - "What are the safest solvents for PS dissolution?"
    - "Cheapest solvents for dissolving HDPE"

    Automatically includes: boiling point, cost (energy), toxicity (LogP), G-score safety rating.

    Args:
        polymer: The polymer to dissolve (e.g., "PET", "LDPE", "PS", "HDPE")
        temperature: Target temperature in °C (default: 120 - most polymers need elevated temps)
        min_solubility: Minimum solubility % to include (default: 5% - adaptive mode will adjust if needed)
        rank_by: How to rank - 'solubility', 'bp'/'boiling', 'cost'/'energy', 'safety'/'gscore', 'toxicity'/'logp'
        top_k: Number of results (default: 15)
        temperature_range: Temperature tolerance +/- (default: 10°C)
        adaptive: If True, automatically searches higher temps and lower thresholds if needed (default: True)

    Returns:
        Ranked list of solvents with solubility AND all properties (BP, cost, safety, toxicity)
    """
    async_db = get_async_db()
    polymer_upper = polymer.strip().upper()

    # Query for solvents that dissolve the polymer
    query = f"""
    SELECT solvent, AVG(solubility____) as avg_solubility,
           MIN(solubility____) as min_sol, MAX(solubility____) as max_sol,
           COUNT(*) as data_points
    FROM common_solvents_database
    WHERE UPPER(polymer) = '{polymer_upper}'
    AND temperature___c_ BETWEEN {temperature - temperature_range} AND {temperature + temperature_range}
    GROUP BY solvent
    HAVING avg_solubility >= {min_solubility}
    ORDER BY avg_solubility DESC
    """

    try:
        df = await async_db.execute_async(query)
    except Exception as e:
        return f"❌ Query error: {e}"

    if len(df) == 0 and adaptive:
        # ADAPTIVE MODE: Try higher temperatures and lower thresholds
        adaptive_note = f"⚠️ No solvents found with >{min_solubility}% solubility for {polymer_upper} at {temperature}°C.\n\n"
        adaptive_note += "**Adaptive search engaged** - trying higher temperatures...\n\n"

        # Try higher temperatures (up to 160°C)
        for try_temp in [140, 150, 160]:
            if try_temp <= temperature:
                continue
            adaptive_query = f"""
            SELECT solvent, AVG(solubility____) as avg_solubility,
                   MIN(solubility____) as min_sol, MAX(solubility____) as max_sol,
                   COUNT(*) as data_points
            FROM common_solvents_database
            WHERE UPPER(polymer) = '{polymer_upper}'
            AND temperature___c_ BETWEEN {try_temp - 5} AND {try_temp + 5}
            GROUP BY solvent
            HAVING avg_solubility >= 1.0
            ORDER BY avg_solubility DESC
            LIMIT {top_k}
            """
            try:
                adaptive_df = await async_db.execute_async(adaptive_query)
                if len(adaptive_df) > 0:
                    df = adaptive_df
                    temperature = try_temp
                    adaptive_note += f"✅ Found results at **{try_temp}°C**\n\n"
                    break
            except:
                pass

        if len(df) == 0:
            # Still nothing - get whatever is available
            fallback_query = f"""
            SELECT solvent, temperature___c_ as temp, AVG(solubility____) as avg_solubility
            FROM common_solvents_database
            WHERE UPPER(polymer) = '{polymer_upper}'
            GROUP BY solvent, temperature___c_
            ORDER BY avg_solubility DESC
            LIMIT 10
            """
            try:
                fallback_df = await async_db.execute_async(fallback_query)
                if len(fallback_df) > 0:
                    best = fallback_df.iloc[0]
                    return (
                        f"❌ No high-solubility solvents found for {polymer_upper}.\n\n"
                        f"**Best available:** {best['solvent']} at {best['avg_solubility']:.1f}% "
                        f"solubility at {best['temp']}°C.\n\n"
                        f"💡 **Suggestions:**\n"
                        f"- This polymer may have limited solubility in common solvents\n"
                        f"- Try ML prediction for specialized solvents\n"
                        f"- Check structurally similar polymers"
                    )
            except:
                pass

            return (
                f"❌ No solubility data found for {polymer_upper}.\n\n"
                f"💡 This polymer may not be in the database. Try:\n"
                f"- `list_available_polymers()` to see available polymers\n"
                f"- ML prediction for polymers not in database"
            )
    elif len(df) == 0:
        return (
            f"❌ No solvents found with >{min_solubility}% solubility for {polymer_upper} at {temperature}°C.\n\n"
            f"💡 Try setting `adaptive=True` or increasing temperature."
        )

    # Get properties for each solvent using cross-database lookup
    results = []
    for _, row in df.iterrows():
        solvent_name = row['solvent']
        avg_sol = row['avg_solubility']

        # Get cross-database properties
        props = get_cross_database_properties(solvent_name, sql_db.conn)

        results.append({
            'solvent': solvent_name,
            'solubility': avg_sol,
            'min_sol': row['min_sol'],
            'max_sol': row['max_sol'],
            'data_points': row['data_points'],
            **props
        })

    # Sort based on rank_by criterion
    rank_lower = rank_by.lower()
    if rank_lower in ['bp', 'boiling', 'boiling_point']:
        # Sort by BP (lower first = easier recovery), put None at end
        results.sort(key=lambda x: (x['bp'] is None, x['bp'] if x['bp'] is not None else float('inf')))
    elif rank_lower in ['cost', 'energy']:
        # Sort by energy (lower = cheaper)
        results.sort(key=lambda x: (x['energy'] is None, x['energy'] if x['energy'] is not None else float('inf')))
    elif rank_lower in ['safety', 'gscore', 'g_score']:
        # Sort by G-score (higher = safer)
        results.sort(key=lambda x: (x['g_score'] is None, -(x['g_score'] if x['g_score'] is not None else -float('inf'))))
    elif rank_lower in ['toxicity', 'logp']:
        # Sort by LogP (lower = less toxic)
        results.sort(key=lambda x: (x['logp'] is None, x['logp'] if x['logp'] is not None else float('inf')))
    else:
        # Default: sort by solubility (higher = better)
        results.sort(key=lambda x: -x['solubility'])

    # Build output
    output = [f"# 🧪 Solvents for {polymer_upper} Dissolution\n"]
    output.append(f"**Temperature:** {temperature}°C (±{temperature_range}°C)")
    output.append(f"**Minimum Solubility:** {min_solubility}%")
    output.append(f"**Ranked By:** {rank_by}")
    output.append(f"**Results:** {len(results)} solvents found\n")

    output.append("## Results\n")
    output.append("| # | Solvent | Solubility | BP (°C) | G-Score | LogP | Energy (J/g) |")
    output.append("|---|---------|------------|---------|---------|------|--------------|")

    for i, r in enumerate(results[:top_k], 1):
        sol = r['solubility']
        bp = f"{r['bp']:.0f}" if r['bp'] is not None else "—"
        gs = f"{r['g_score']:.1f}" if r['g_score'] is not None else "—"
        lp = f"{r['logp']:.2f}" if r['logp'] is not None else "—"
        en = f"{r['energy']:.0f}" if r['energy'] is not None else "—"

        # Add quality indicators
        sol_icon = "✅" if sol > 80 else "🟡" if sol > 50 else "⚠️"
        gs_icon = ""
        if r['g_score'] is not None:
            gs_icon = "✅" if r['g_score'] >= 7 else "🟡" if r['g_score'] >= 5 else "🔴"

        output.append(f"| {i} | **{r['solvent']}** | {sol_icon} {sol:.1f}% | {bp} | {gs_icon}{gs} | {lp} | {en} |")

    # Add recommendations
    output.append("\n## 📝 Recommendations\n")

    # Best by different criteria
    if results:
        best_sol = max(results, key=lambda x: x['solubility'])
        output.append(f"**Best Solubility:** {best_sol['solvent']} ({best_sol['solubility']:.1f}%)")

        with_bp = [r for r in results if r['bp'] is not None]
        if with_bp:
            lowest_bp = min(with_bp, key=lambda x: x['bp'])
            output.append(f"**Lowest Boiling Point:** {lowest_bp['solvent']} ({lowest_bp['bp']:.0f}°C) - easiest recovery")

        with_gs = [r for r in results if r['g_score'] is not None]
        if with_gs:
            safest = max(with_gs, key=lambda x: x['g_score'])
            output.append(f"**Safest (G-Score):** {safest['solvent']} (G={safest['g_score']:.1f}/10)")

        with_energy = [r for r in results if r['energy'] is not None]
        if with_energy:
            cheapest = min(with_energy, key=lambda x: x['energy'])
            output.append(f"**Cheapest (Energy):** {cheapest['solvent']} ({cheapest['energy']:.0f} J/g)")

        with_logp = [r for r in results if r['logp'] is not None]
        if with_logp:
            least_toxic = min(with_logp, key=lambda x: x['logp'])
            output.append(f"**Least Toxic (LogP):** {least_toxic['solvent']} (LogP={least_toxic['logp']:.2f})")

    # Legend
    output.append("\n## Legend")
    output.append("- **G-Score:** 1-10 scale (higher = safer). ✅≥7, 🟡5-7, 🔴<5")
    output.append("- **LogP:** Lower/negative = less toxic, more water soluble")
    output.append("- **Energy:** Lower = cheaper operating cost")
    output.append("- **BP:** Lower = easier solvent recovery")

    return "\n".join(output)


@tool
@safe_tool_wrapper
async def view_alternative_separation_sequence(
    polymers: str,
    sequence_rank: Optional[int] = None,
    starting_polymer: Optional[str] = None,
    top_k_solvents: int = 5,
    temperature: float = 25.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    View a specific alternative separation sequence with clear visualization.

    Use after plan_sequential_separation to explore different sequence options.

    Parameters:
    - polymers: Comma-separated list of polymers (e.g., "LDPE,HDPE,PP,PS")
    - sequence_rank: Rank of sequence to view (1=best, 2=2nd best, etc.)
    - starting_polymer: Name of polymer to start with (alternative to rank)
    - top_k_solvents: Number of top solvents to show per step (default: 5)
    - temperature: Target temperature in °C (default: 25.0)

    WHEN TO USE:
    - "Show me the 2nd best separation sequence"
    - "What if we start with PET instead?"
    - "View LDPE-first separation option"
    """
    from itertools import permutations

    async_db = get_async_db()

    # Parse polymers
    polymer_list = [p.strip() for p in polymers.split(',') if p.strip()]
    n_polymers = len(polymer_list)

    if n_polymers < 2:
        return "Error: Need at least 2 polymers."

    # Generate and analyze all sequences (same as plan_sequential_separation)
    all_sequences = list(permutations(polymer_list))

    # Reuse the analysis logic from plan_sequential_separation
    async def find_top_solvents(target: str, remaining: list, k: int = 5) -> list:
        """Find top-k solvents for separating target from remaining polymers."""
        if not remaining:
            return [{"solvent": "N/A", "selectivity": float('inf'), "target_sol": 100, "max_other": 0}]

        all_polymers_for_query = [target] + remaining
        polymer_filter = "', '".join(all_polymers_for_query)

        query = f"""
        SELECT {solvent_column}, {polymer_column}, AVG({solubility_column}) as avg_sol
        FROM {table_name}
        WHERE {polymer_column} IN ('{polymer_filter}')
        AND {temperature_column} BETWEEN {temperature - 5} AND {temperature + 5}
        GROUP BY {solvent_column}, {polymer_column}
        """

        try:
            df = await async_db.execute_async(query)
        except Exception as e:
            return [{"solvent": "Error", "selectivity": 0, "target_sol": 0, "max_other": 0, "error": str(e)}]

        if len(df) == 0:
            return [{"solvent": "No data", "selectivity": 0, "target_sol": 0, "max_other": 0}]

        results = []
        for solvent in df[solvent_column].unique():
            solvent_data = df[df[solvent_column] == solvent]
            target_data = solvent_data[solvent_data[polymer_column] == target]
            if len(target_data) == 0:
                continue
            target_sol = target_data['avg_sol'].values[0]

            other_data = solvent_data[solvent_data[polymer_column].isin(remaining)]
            max_other = other_data['avg_sol'].max() if len(other_data) > 0 else 0

            selectivity = target_sol - max_other
            results.append({
                "solvent": solvent,
                "selectivity": selectivity,
                "target_sol": target_sol,
                "max_other": max_other
            })

        results.sort(key=lambda x: x["selectivity"], reverse=True)
        return results[:k]

    async def analyze_sequence(sequence, seq_idx):
        """Analyze single sequence."""
        step_tasks = []
        step_info = []
        for step, target in enumerate(sequence[:-1], 1):
            remaining = list(sequence[step:])
            step_tasks.append(find_top_solvents(target, remaining, top_k_solvents))
            step_info.append((step, target, remaining))

        all_step_results = await asyncio.gather(*step_tasks)

        total_min_selectivity = float('inf')
        seq_steps = []

        for (step, target, remaining), top_solvents in zip(step_info, all_step_results):
            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents
            }
            seq_steps.append(step_data)

            if top_solvents and top_solvents[0]["selectivity"] < total_min_selectivity:
                total_min_selectivity = top_solvents[0]["selectivity"]

        return {
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps
        }

    # Analyze all sequences with limited concurrency
    semaphore = asyncio.Semaphore(10)

    async def analyze_with_limit(seq, idx):
        async with semaphore:
            return await analyze_sequence(seq, idx)

    sequence_analyses = await asyncio.gather(*[
        analyze_with_limit(seq, idx) for idx, seq in enumerate(all_sequences, 1)
    ])

    # Sort by min_selectivity
    sequence_scores = sorted(sequence_analyses, key=lambda x: x["min_selectivity"], reverse=True)

    # Find the requested sequence
    target_seq = None
    rank = None

    if sequence_rank is not None:
        # User specified a rank
        if 1 <= sequence_rank <= len(sequence_scores):
            target_seq = sequence_scores[sequence_rank - 1]
            rank = sequence_rank
        else:
            return f"Error: Rank {sequence_rank} is out of range (1-{len(sequence_scores)})"

    elif starting_polymer is not None:
        # User specified a starting polymer
        starting_polymer_normalized = starting_polymer.strip().upper()
        for idx, seq_data in enumerate(sequence_scores, 1):
            if seq_data["sequence"][0].upper() == starting_polymer_normalized:
                target_seq = seq_data
                rank = idx
                break

        if target_seq is None:
            return f"Error: No sequence found starting with '{starting_polymer}'. Available polymers: {', '.join(polymer_list)}"

    else:
        return "Error: Must specify either sequence_rank or starting_polymer"

    # Generate output with visualization
    output = []
    output.append(f"# Alternative Separation Sequence (Rank #{rank})\n")
    output.append(f"**Sequence:** {' → '.join(target_seq['sequence'])}")
    output.append(f"**Minimum Selectivity:** {target_seq['min_selectivity']:.1f}%")
    output.append(f"**Temperature:** {temperature}°C\n")

    # Create the same clear visualization as in plan_sequential_separation
    try:
        def get_color(selectivity):
            if selectivity > 30:
                return '#2ecc71'
            elif selectivity > 10:
                return '#f1c40f'
            elif selectivity > 0:
                return '#e67e22'
            else:
                return '#e74c3c'

        sequence = target_seq["sequence"]
        steps = target_seq["steps"]
        min_sel = target_seq["min_selectivity"]

        # Create figure - VERTICAL FLOWCHART
        n_steps = len(steps)
        fig_height = max(3 + n_steps * 2.5, 8)
        fig_width = 12
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Title
        ax.set_title(
            f'SEPARATION SEQUENCE (Rank #{rank} of {len(sequence_scores)})\n' +
            f'Sequence: {" → ".join(sequence)} | Min Selectivity: {min_sel:.1f}% | Temp: {temperature}°C',
            fontsize=16, fontweight='bold', pad=20
        )

        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, n_steps + 2)
        ax.axis('off')

        # Starting mixture
        y_pos = n_steps + 1
        ax.add_patch(plt.Rectangle((2, y_pos - 0.3), 6, 0.6,
                                   facecolor='#3498db', edgecolor='black', linewidth=2))
        ax.text(5, y_pos, f'STARTING MIXTURE: {", ".join(polymer_list)}',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Draw each step
        for idx, step in enumerate(steps):
            y_pos = n_steps - idx
            target = step["target"]
            remaining = step["remaining"]
            top_solvent = step["solvents"][0] if step["solvents"] else {"solvent": "N/A", "selectivity": 0}
            solvent_name = top_solvent["solvent"]
            selectivity = top_solvent.get("selectivity", 0)
            color = get_color(selectivity)

            # Arrow down with solvent info
            ax.annotate('', xy=(3.5, y_pos + 0.4), xytext=(3.5, y_pos + 0.9),
                       arrowprops=dict(arrowstyle='->', lw=4, color=color))

            # Step box (left side - narrower to avoid overlap)
            ax.add_patch(plt.Rectangle((1.2, y_pos - 0.35), 4.6, 0.7,
                                      facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.3))

            # Step number and target
            ax.add_patch(plt.Circle((1.9, y_pos), 0.25, facecolor=color, edgecolor='black', linewidth=2))
            ax.text(1.9, y_pos, str(idx + 1), ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')

            # Separated polymer (large text)
            ax.text(2.7, y_pos, f'SEPARATE: {target}',
                   ha='left', va='center', fontsize=14, fontweight='bold')

            # Solvent label box (right side - clear separation from step box)
            ax.add_patch(plt.Rectangle((6.2, y_pos + 0.55), 3.3, 0.5,
                                      facecolor='white', edgecolor=color, linewidth=2))
            ax.text(7.85, y_pos + 0.8, f'Solvent: {solvent_name}',
                   ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(7.85, y_pos + 0.6, f'Selectivity: {selectivity:.1f}%',
                   ha='center', va='center', fontsize=10, color=color, fontweight='bold')

            # Remaining polymers (positioned below step box, no overlap)
            if remaining:
                remaining_text = f'Remaining: {", ".join(remaining)}'
                ax.text(5.7, y_pos - 0.15, remaining_text,
                       ha='right', va='center', fontsize=10, color='#34495e',
                       style='italic', weight='bold')
            else:
                ax.text(5.7, y_pos - 0.15, '(Last polymer - isolated)',
                       ha='right', va='center', fontsize=10, color='#27ae60',
                       style='italic', weight='bold')

        # Final result
        ax.add_patch(plt.Rectangle((2, -0.3), 6, 0.6,
                                  facecolor='#2ecc71', edgecolor='black', linewidth=2.5))
        ax.text(5, 0, '✓ ALL POLYMERS SEPARATED',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                      markersize=15, markeredgecolor='black', linewidth=2, label='Excellent (>30%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#f1c40f',
                      markersize=15, markeredgecolor='black', linewidth=2, label='Good (10-30%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22',
                      markersize=15, markeredgecolor='black', linewidth=2, label='Marginal (0-10%)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                      markersize=15, markeredgecolor='black', linewidth=2, label='Poor (<0%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                 frameon=True, fancybox=True, title='Selectivity Quality', title_fontsize=12)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        filepath = save_plot(fig, f"separation_sequence_rank{rank}")
        plt.close(fig)  # Clean up figure to prevent memory leaks
        output.append(f"\n📊 Visualization saved: {get_plot_url(filepath)}")

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        output.append(f"\n⚠️ Could not create visualization: {e}")
        # Try to close figure even if error occurred
        try:
            plt.close(fig)
        except:
            pass

    # Step details
    output.append("\n## Separation Steps\n")
    for step_data in target_seq["steps"]:
        step_num = step_data["step"]
        target = step_data["target"]
        remaining = step_data["remaining"]
        solvents = step_data["solvents"]

        output.append(f"**Step {step_num}: Separate {target}**")
        if remaining:
            output.append(f"  - Remaining in mixture: {', '.join(remaining)}")
        output.append(f"  - Top solvents:")

        for rank_idx, sol in enumerate(solvents[:3], 1):
            sol_name = sol.get("solvent", "N/A")
            sel = sol.get("selectivity", 0)
            output.append(f"    {rank_idx}. {sol_name} (selectivity: {sel:.1f}%)")
        output.append("")

    # Comparison to best
    if rank > 1:
        best_seq = sequence_scores[0]
        output.append("## Comparison to Best Sequence\n")
        output.append(f"**Best sequence:** {' → '.join(best_seq['sequence'])} (min selectivity: {best_seq['min_selectivity']:.1f}%)")
        output.append(f"**This sequence:** {' → '.join(target_seq['sequence'])} (min selectivity: {target_seq['min_selectivity']:.1f}%)")
        output.append(f"**Difference:** {target_seq['min_selectivity'] - best_seq['min_selectivity']:.1f}% selectivity")

    return "\n".join(output)


# ============================================================
# Solvent Property Tools
# ============================================================

# Standard solvent data table name (will be auto-detected or can be set)
SOLVENT_DATA_TABLE = None  # Will be auto-detected from loaded tables

def get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent data table name."""
    global SOLVENT_DATA_TABLE
    
    if SOLVENT_DATA_TABLE and SOLVENT_DATA_TABLE in sql_db.table_schemas:
        return SOLVENT_DATA_TABLE
    
    # Try to find a table with solvent properties
    for table_name in sql_db.table_schemas.keys():
        if 'solvent' in table_name.lower() and 'solubility' not in table_name.lower():
            # Check if it has expected columns
            schema = sql_db.table_schemas[table_name]
            cols_lower = [c.lower() for c in schema['columns']]
            if any('bp' in c or 'boil' in c for c in cols_lower) or \
               any('logp' in c for c in cols_lower) or \
               any('energy' in c for c in cols_lower):
                SOLVENT_DATA_TABLE = table_name
                logger.info(f"Auto-detected solvent data table: {table_name}")
                return table_name
    
    return None


def get_solvent_name_column(table_name: str) -> Optional[str]:
    """Get the column name that contains solvent names."""
    if table_name not in sql_db.table_schemas:
        return None

    cols = sql_db.table_schemas[table_name]['columns']

    # Priority order for solvent name column
    priority_patterns = ['solvent_name', 'solvent', 'name', 'compound']

    for pattern in priority_patterns:
        for col in cols:
            if pattern in col.lower():
                return col

    # If no match, return first string column
    for col, dtype in sql_db.table_schemas[table_name]['types'].items():
        if 'VARCHAR' in str(dtype).upper() or 'TEXT' in str(dtype).upper():
            return col

    return cols[0] if cols else None


def get_cosmobase_column(table_name: str) -> Optional[str]:
    """Get the 'Solvent name in cosmobase' column for exact matching."""
    if table_name not in sql_db.table_schemas:
        return None

    cols = sql_db.table_schemas[table_name]['columns']

    # Look for cosmobase column specifically
    for col in cols:
        if 'cosmobase' in col.lower():
            return col

    return None


async def lookup_solvent_properties(solvent_names: list, solvent_table: str) -> dict:
    """
    Look up solvent properties for multiple solvents with robust fuzzy matching (ASYNC).

    Uses multiple strategies to match solvent names:
    1. Exact match on COSMOBASE/name column
    2. Common abbreviation mapping
    3. Bidirectional fuzzy matching
    4. Partial substring matching

    Returns a dict mapping solvent names to their properties.
    """
    if not solvent_table or solvent_table not in sql_db.table_schemas:
        return {}

    # Common solvent abbreviations mapping
    ABBREVIATION_MAP = {
        # Common abbreviations to full names
        'dmf': 'dimethylformamide',
        'thf': 'tetrahydrofuran',
        'dme': 'dimethoxyethane',
        'meoh': 'methanol',
        'etoh': 'ethanol',
        'ipa': 'isopropanol',
        'nmp': 'n-methyl-2-pyrrolidone',
        'dmso': 'dimethyl sulfoxide',
        'dcm': 'dichloromethane',
        'dce': 'dichloroethane',
        'mecn': 'acetonitrile',
        'etac': 'ethyl acetate',
        'acac': 'acetylacetone',
        'tfa': 'trifluoroacetic acid',
        'tfe': 'trifluoroethanol',
        'hfip': 'hexafluoroisopropanol',
        'chcl3': 'chloroform',
        'ccl4': 'carbon tetrachloride',
        'phme': 'toluene',
        'phh': 'benzene',
        'mtbe': 'methyl tert-butyl ether',
        'tbme': 'tert-butyl methyl ether',
        'dipa': 'diisopropylamine',
        'tea': 'triethylamine',
        'dbu': '1,8-diazabicyclo[5.4.0]undec-7-ene',
        'pyr': 'pyridine',
        'acn': 'acetonitrile',
        'mibk': 'methyl isobutyl ketone',
    }

    async_db = get_async_db()
    schema = sql_db.table_schemas[solvent_table]
    cols = schema['columns']
    cols_lower = {c.lower(): c for c in cols}

    # Get column names
    cosmobase_col = get_cosmobase_column(solvent_table)
    name_col = get_solvent_name_column(solvent_table)

    # Property columns
    logp_col = next((cols_lower[k] for k in cols_lower if 'logp' in k), None)
    bp_col = next((cols_lower[k] for k in cols_lower if 'bp' in k or 'boil' in k), None)
    energy_col = next((cols_lower[k] for k in cols_lower if 'energy' in k), None)
    cp_col = next((cols_lower[k] for k in cols_lower if 'cp' in k and 'logp' not in k), None)

    match_col = cosmobase_col or name_col
    if not match_col:
        return {}

    async def find_solvent_match(solvent: str):
        """Try multiple strategies to find solvent properties."""
        sol_lower = solvent.lower().strip()
        sol_normalized = sol_lower.replace('-', '').replace(' ', '').replace(',', '')

        # Strategy 1: Exact match
        query1 = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") = '{sol_lower}'"
        try:
            df = await async_db.execute_async(query1)
            if len(df) > 0:
                return df.iloc[0]
        except:
            pass

        # Strategy 2: Try abbreviation mapping
        if sol_lower in ABBREVIATION_MAP:
            full_name = ABBREVIATION_MAP[sol_lower]
            query2 = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{full_name}%' ORDER BY LENGTH(\"{match_col}\")"
            try:
                df = await async_db.execute_async(query2)
                if len(df) > 0:
                    return df.iloc[0]
            except:
                pass

        # Strategy 3: Reverse - check if solvent is abbreviation of a full name
        query3 = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{sol_lower}%' ORDER BY LENGTH(\"{match_col}\")"
        try:
            df = await async_db.execute_async(query3)
            if len(df) > 0:
                # Prefer shorter matches (more likely to be correct)
                return df.iloc[0]
        except:
            pass

        # Strategy 4: Normalized match (remove special characters)
        query4 = f"""
        SELECT * FROM {solvent_table}
        WHERE REPLACE(REPLACE(REPLACE(LOWER(\"{match_col}\"), '-', ''), ' ', ''), ',', '') LIKE '%{sol_normalized}%'
        ORDER BY LENGTH(\"{match_col}\")
        """
        try:
            df = await async_db.execute_async(query4)
            if len(df) > 0:
                return df.iloc[0]
        except:
            pass

        # Strategy 5: Check if full name contains the abbreviation as a word
        for abbrev, full in ABBREVIATION_MAP.items():
            if abbrev in sol_lower or sol_lower in full:
                query5 = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{full}%' ORDER BY LENGTH(\"{match_col}\")"
                try:
                    df = await async_db.execute_async(query5)
                    if len(df) > 0:
                        return df.iloc[0]
                except:
                    pass

        return None

    # Find matches for all solvents in parallel
    match_tasks = [find_solvent_match(solvent) for solvent in solvent_names]
    matches = await asyncio.gather(*match_tasks)

    # Extract properties from matches
    props_map = {}
    for solvent, row in zip(solvent_names, matches):
        props = {'logp': None, 'bp': None, 'energy': None, 'cp': None}

        if row is not None:
            props = {
                'logp': row[logp_col] if logp_col and logp_col in row.index else None,
                'bp': row[bp_col] if bp_col and bp_col in row.index else None,
                'energy': row[energy_col] if energy_col and energy_col in row.index else None,
                'cp': row[cp_col] if cp_col and cp_col in row.index else None,
            }

        props_map[solvent] = props

    return props_map


@tool
@safe_tool_wrapper
def list_solvent_properties() -> str:
    """
    List all solvents with their properties from the solvent database.
    Shows: name, CAS number, boiling point, LogP, heat capacity, energy cost.
    """
    table_name = get_solvent_table_name()
    
    if not table_name:
        return ("❌ No solvent properties table found.\n"
                "Please upload a CSV file named 'Solvent_Data.csv' with columns:\n"
                "- Solvent name, CAS number, Bp (C), LogP, Cp (J/gK), Energy (J/g)")
    
    # Get all solvent data
    query = f"SELECT * FROM {table_name} ORDER BY 1 LIMIT 100"
    result = sql_db.execute_query(query, limit=100)
    
    if not result["success"]:
        return f"❌ Error querying solvent data: {result.get('error')}"
    
    output = [f"**Solvent Properties Database**\n"]
    output.append(f"Table: `{table_name}`")
    output.append(f"Total solvents: {sql_db.table_schemas[table_name]['row_count']}")
    output.append(f"Columns: {', '.join(result['columns'])}\n")
    output.append(result["preview"])
    
    return "\n".join(output)


@tool
@safe_tool_wrapper
def get_solvent_properties(solvent_names: str) -> str:
    """
    Get detailed properties for specific solvents.
    
    Args:
        solvent_names: Comma-separated list of solvent names to look up
    
    Returns:
        Properties including boiling point, LogP (toxicity indicator), 
        heat capacity, and energy cost for each solvent.
    """
    table_name = get_solvent_table_name()
    
    if not table_name:
        return "❌ No solvent properties table found. Upload Solvent_Data.csv first."
    
    name_col = get_solvent_name_column(table_name)
    if not name_col:
        return "❌ Could not identify solvent name column."
    
    # Parse solvent names
    solvents = [s.strip() for s in solvent_names.split(',') if s.strip()]
    
    if not solvents:
        return "❌ No solvent names provided."
    
    # Build query with fuzzy matching
    conditions = []
    for solvent in solvents:
        conditions.append(f"LOWER({name_col}) LIKE '%{solvent.lower()}%'")
    
    where_clause = " OR ".join(conditions)
    query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    
    result = sql_db.execute_query(query, limit=50)
    
    if not result["success"]:
        return f"❌ Query error: {result.get('error')}"
    
    if result["rows"] == 0:
        # Try exact match
        exact_conditions = [f"LOWER({name_col}) = '{s.lower()}'" for s in solvents]
        query = f"SELECT * FROM {table_name} WHERE {' OR '.join(exact_conditions)}"
        result = sql_db.execute_query(query, limit=50)
        
        if result["rows"] == 0:
            return f"❌ No solvents found matching: {', '.join(solvents)}\n\nUse `list_solvent_properties()` to see available solvents."
    
    output = [f"**Solvent Properties**\n"]
    output.append(f"Requested: {', '.join(solvents)}")
    output.append(f"Found: {result['rows']} match(es)\n")
    output.append(result["preview"])
    
    # Add interpretation
    df = result["dataframe"]
    output.append("\n**Interpretation:**")
    
    # Find relevant columns
    cols = {c.lower(): c for c in df.columns}
    
    logp_col = next((cols[k] for k in cols if 'logp' in k), None)
    bp_col = next((cols[k] for k in cols if 'bp' in k or 'boil' in k), None)
    energy_col = next((cols[k] for k in cols if 'energy' in k), None)
    
    if logp_col:
        output.append(f"- **LogP** (toxicity): Lower/negative = less toxic, higher = more toxic")
    if bp_col:
        output.append(f"- **Boiling Point**: Higher = harder to remove/recycle")
    if energy_col:
        output.append(f"- **Energy**: Higher = more expensive to use")
    
    return "\n".join(output)


@tool
@safe_tool_wrapper
def rank_solvents_by_property(
    property_name: str,
    ascending: bool = True,
    limit: int = 20,
    filter_solvents: Optional[str] = None
) -> str:
    """
    Rank solvents by a specific property.
    
    Args:
        property_name: Property to rank by - 'bp', 'logp', 'energy', 'cp', or exact column name
        ascending: True for lowest first (good for cost/toxicity), False for highest first
        limit: Number of results to return (default 20)
        filter_solvents: Optional comma-separated list of solvents to filter to
    
    Returns:
        Ranked list of solvents with the specified property.
        
    Examples:
        - rank_solvents_by_property('energy', ascending=True) - cheapest solvents
        - rank_solvents_by_property('logp', ascending=True) - least toxic solvents
        - rank_solvents_by_property('bp', ascending=False) - highest boiling points
    """
    table_name = get_solvent_table_name()
    
    if not table_name:
        return "❌ No solvent properties table found."
    
    # Map common property names to likely column names
    property_map = {
        'bp': ['bp', 'bp_c', 'boiling_point', 'boilingpoint'],
        'boiling': ['bp', 'bp_c', 'boiling_point'],
        'logp': ['logp', 'log_p', 'logp_value'],
        'toxicity': ['logp', 'log_p'],  # LogP is proxy for toxicity
        'energy': ['energy', 'energy_j_g', 'energy_cost'],
        'cost': ['energy', 'energy_j_g', 'energy_cost'],
        'cp': ['cp', 'cp_j_gk', 'heat_capacity'],
        'heat_capacity': ['cp', 'cp_j_gk', 'heat_capacity'],
    }
    
    # Find the actual column name
    schema = sql_db.table_schemas[table_name]
    cols_lower = {c.lower().replace(' ', '_').replace('(', '_').replace(')', ''): c 
                  for c in schema['columns']}
    
    target_col = None
    prop_lower = property_name.lower().replace(' ', '_')
    
    # Direct match
    if prop_lower in cols_lower:
        target_col = cols_lower[prop_lower]
    else:
        # Try mapped names
        search_terms = property_map.get(prop_lower, [prop_lower])
        for term in search_terms:
            for col_key, col_name in cols_lower.items():
                if term in col_key:
                    target_col = col_name
                    break
            if target_col:
                break
    
    if not target_col:
        available = ', '.join(schema['columns'])
        return f"❌ Property '{property_name}' not found.\n\nAvailable columns: {available}"
    
    name_col = get_solvent_name_column(table_name)
    order = "ASC" if ascending else "DESC"
    
    # Build query
    if filter_solvents:
        solvents = [s.strip() for s in filter_solvents.split(',')]
        conditions = [f"LOWER({name_col}) LIKE '%{s.lower()}%'" for s in solvents]
        where_clause = f"WHERE ({' OR '.join(conditions)}) AND {target_col} IS NOT NULL"
    else:
        where_clause = f"WHERE {target_col} IS NOT NULL"
    
    query = f"""
    SELECT * FROM {table_name}
    {where_clause}
    ORDER BY {target_col} {order}
    LIMIT {limit}
    """
    
    result = sql_db.execute_query(query, limit=limit)
    
    if not result["success"]:
        return f"❌ Query error: {result.get('error')}"
    
    direction = "lowest" if ascending else "highest"
    output = [f"**Solvents Ranked by {target_col}** ({direction} first)\n"]
    
    if filter_solvents:
        output.append(f"Filtered to: {filter_solvents}")
    
    output.append(f"Results: {result['rows']}\n")
    output.append(result["preview"])
    
    # Add context
    output.append(f"\n**Note:** ")
    if 'logp' in target_col.lower():
        output.append("Lower/negative LogP generally indicates lower toxicity and higher water solubility.")
    elif 'energy' in target_col.lower():
        output.append("Lower energy typically means lower operating cost.")
    elif 'bp' in target_col.lower():
        output.append("Lower boiling point means easier solvent recovery but may require pressure vessels.")
    
    return "\n".join(output)


@tool
@safe_tool_wrapper
async def analyze_separation_with_properties(
    target_polymer: str,
    comparison_polymers: str,
    temperature: float = 120.0,
    rank_by: str = "selectivity",
    top_k: int = 10,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____"
) -> str:
    """
    Find selective solvents AND include their physical/economic properties.

    Combines separation analysis with solvent property data to help choose
    solvents based on both selectivity AND practical considerations.

    Args:
        target_polymer: Polymer to dissolve
        comparison_polymers: Comma-separated polymers to separate from
        temperature: Target temperature (default 120°C - polymers need elevated temps)
        rank_by: How to rank results - 'selectivity', 'energy' (cost), 'logp' (toxicity), 'bp'
        top_k: Number of top results to return
        table_name: Solubility data table (default: common_solvents_database)
        polymer_column: Column with polymer names (default: polymer)
        solvent_column: Column with solvent names (default: solvent)
        temperature_column: Column with temperature values (default: temperature___c_)
        solubility_column: Column with solubility values (default: solubility____)

    Returns:
        Ranked solvents with selectivity AND properties (cost, toxicity, bp)
    """
    # Parse comparison polymers
    if isinstance(comparison_polymers, str):
        comp_list = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
    else:
        comp_list = list(comparison_polymers) if comparison_polymers else []
    
    if not comp_list:
        return "❌ No comparison polymers specified."
    
    all_polymers = [target_polymer] + comp_list
    polymer_filter = "', '".join(all_polymers)
    
    # Query solubility data
    query = f"""
    SELECT {solvent_column}, {polymer_column}, AVG({solubility_column}) as avg_sol
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {temperature_column} BETWEEN {temperature - 5} AND {temperature + 5}
    GROUP BY {solvent_column}, {polymer_column}
    """
    
    try:
        df = sql_db.conn.execute(query).fetchdf()
    except Exception as e:
        return f"❌ Query error: {e}"
    
    if len(df) == 0:
        return f"❌ No solubility data found for these polymers at {temperature}°C"
    
    # Calculate selectivity for each solvent
    results = []
    for solvent in df[solvent_column].unique():
        solvent_data = df[df[solvent_column] == solvent]
        
        target_data = solvent_data[solvent_data[polymer_column] == target_polymer]
        if len(target_data) == 0:
            continue
        target_sol = target_data['avg_sol'].values[0]
        
        other_data = solvent_data[solvent_data[polymer_column].isin(comp_list)]
        if len(other_data) == 0:
            max_other = 0
        else:
            max_other = other_data['avg_sol'].max()
        
        selectivity = target_sol - max_other
        
        results.append({
            "solvent": solvent,
            "selectivity": selectivity,
            "target_solubility": target_sol,
            "max_other_solubility": max_other
        })
    
    if not results:
        return "❌ No solvents found with data for all specified polymers."

    # Get solvent properties if available using exact matching
    solvent_table = get_solvent_table_name()
    properties_available = False

    if solvent_table:
        solvent_names = [r["solvent"] for r in results]
        prop_lookup = await lookup_solvent_properties(solvent_names, solvent_table)

        if prop_lookup:
            properties_available = True
            for r in results:
                if r["solvent"] in prop_lookup:
                    r.update(prop_lookup[r["solvent"]])

    # Also get G-scores from GSK dataset if available
    try:
        solvent_names = [r["solvent"] for r in results]
        solvent_filter = "', '".join(solvent_names)
        gscore_query = f"""
        SELECT solvent_common_name, g_score
        FROM gsk_dataset
        WHERE solvent_common_name IN ('{solvent_filter}')
        """
        gscore_df = sql_db.conn.execute(gscore_query).fetchdf()

        if len(gscore_df) > 0:
            gscore_lookup = dict(zip(gscore_df['solvent_common_name'], gscore_df['g_score']))
            for r in results:
                # Try exact match
                if r["solvent"] in gscore_lookup:
                    r['g_score'] = gscore_lookup[r["solvent"]]
                else:
                    # Try fuzzy match
                    match_result = fuzzy_match_solvent_name(r["solvent"], dataset="gsk", threshold=85)
                    if match_result and match_result["matched_name"] in gscore_lookup:
                        r['g_score'] = gscore_lookup[match_result["matched_name"]]
    except Exception as e:
        logger.debug(f"Could not fetch G-scores: {e}")
    
    # Sort results based on rank_by parameter
    rank_by_lower = rank_by.lower()
    
    if rank_by_lower == 'selectivity':
        results.sort(key=lambda x: x.get('selectivity', 0), reverse=True)
    elif rank_by_lower in ['energy', 'cost']:
        # Lower energy = better (cheaper)
        results.sort(key=lambda x: (x.get('energy') is None, x.get('energy', float('inf'))))
    elif rank_by_lower in ['logp', 'toxicity']:
        # Lower LogP = less toxic
        results.sort(key=lambda x: (x.get('logp') is None, x.get('logp', float('inf'))))
    elif rank_by_lower in ['bp', 'boiling']:
        # Can sort either way - default to lower first (easier recovery)
        results.sort(key=lambda x: (x.get('bp') is None, x.get('bp', float('inf'))))
    else:
        # Default to selectivity
        results.sort(key=lambda x: x.get('selectivity', 0), reverse=True)
    
    # Format output
    output = [f"**Separation Analysis with Solvent Properties**\n"]
    output.append(f"Target: Dissolve **{target_polymer}**")
    output.append(f"Separate from: {', '.join(comp_list)}")
    output.append(f"Temperature: {temperature}°C")
    output.append(f"Ranked by: **{rank_by}**")
    output.append(f"Properties available: {'✅ Yes' if properties_available else '❌ No (upload Solvent_Data.csv)'}\n")
    
    output.append(f"**Top {min(top_k, len(results))} Solvents:**\n")
    
    for i, r in enumerate(results[:top_k], 1):
        selectivity = r.get('selectivity', 0)
        symbol = "✅" if selectivity > 10 else "⚠️" if selectivity > 0 else "❌"

        line = f"{i}. {symbol} **{r['solvent']}**"
        line += f"\n   - Selectivity: {selectivity:.1f}%"
        line += f" (target: {r.get('target_solubility', 0):.1f}%, max_other: {r.get('max_other_solubility', 0):.1f}%)"
        
        if properties_available:
            props = []
            if r.get('logp') is not None:
                toxicity = "Low" if r['logp'] < 0 else "Medium" if r['logp'] < 2 else "High"
                props.append(f"LogP: {r['logp']:.2f} ({toxicity} toxicity)")
            if r.get('g_score') is not None:
                g_score = r['g_score']
                if g_score >= 8.0:
                    g_rating = "✅ Excellent"
                elif g_score >= 6.0:
                    g_rating = "🟢 Good"
                elif g_score >= 4.0:
                    g_rating = "🟡 Problematic"
                else:
                    g_rating = "🔴 Hazardous"
                props.append(f"G-Score: {g_score:.2f}/10 ({g_rating})")
            if r.get('energy') is not None:
                props.append(f"Energy: {r['energy']:.1f} J/g")
            if r.get('bp') is not None:
                props.append(f"BP: {r['bp']:.1f}°C")
            if r.get('cp') is not None:
                props.append(f"Cp: {r['cp']:.2f} J/gK")

            if props:
                line += f"\n   - Properties: {' | '.join(props)}"
        
        output.append(line)
    
    # Summary recommendations
    output.append("\n**Recommendations:**")

    if results:
        best_selectivity = max(results, key=lambda x: x.get('selectivity', 0))
        output.append(f"- Best selectivity: **{best_selectivity['solvent']}** ({best_selectivity.get('selectivity', 0):.4f})")

        if properties_available:
            # Find best by different criteria
            with_energy = [r for r in results if r.get('energy') is not None and r.get('selectivity', 0) > 0]
            with_logp = [r for r in results if r.get('logp') is not None and r.get('selectivity', 0) > 0]
            with_gscore = [r for r in results if r.get('g_score') is not None and r.get('selectivity', 0) > 0]

            if with_energy:
                cheapest = min(with_energy, key=lambda x: x['energy'])
                output.append(f"- Lowest cost (with positive selectivity): **{cheapest['solvent']}** (Energy: {cheapest['energy']:.1f} J/g)")

            if with_logp:
                least_toxic = min(with_logp, key=lambda x: x['logp'])
                output.append(f"- Least toxic by LogP (with positive selectivity): **{least_toxic['solvent']}** (LogP: {least_toxic['logp']:.2f})")

            if with_gscore:
                safest = max(with_gscore, key=lambda x: x['g_score'])
                output.append(f"- Safest by G-Score (with positive selectivity): **{safest['solvent']}** (G-Score: {safest['g_score']:.2f}/10)")
    
    return "\n".join(output)


# ============================================================
# GSK Safety (G-Score) Analysis Tools
# ============================================================

@tool
@safe_tool_wrapper
async def get_solvent_gscore(solvent_name: str, use_fuzzy_matching: bool = True) -> str:
    """
    Get the GSK G-score (safety rating) for a solvent.

    The G-score is a composite safety metric from 0 (worst) to 10 (best),
    calculated as the geometric mean of EHSW scores:
    - E: Environmental impact
    - H: Health hazards
    - S: Safety concerns
    - W: Waste considerations

    Args:
        solvent_name: Name of the solvent to look up
        use_fuzzy_matching: If True, attempt fuzzy name matching if exact match fails

    Returns:
        G-score information including score, family classification, and matched name
    """
    try:
        async_db = get_async_db()

        # Try exact match first
        query = f"""
        SELECT solvent_common_name, classification, g_score, cas_number
        FROM gsk_dataset
        WHERE LOWER(solvent_common_name) = LOWER('{solvent_name}')
        """

        result = await async_db.execute_async(query)

        # If no exact match and fuzzy matching enabled, try fuzzy match
        if len(result) == 0 and use_fuzzy_matching:
            match_result = fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)

            if match_result:
                matched_name = match_result["matched_name"]
                query = f"""
                SELECT solvent_common_name, classification, g_score, cas_number
                FROM gsk_dataset
                WHERE LOWER(solvent_common_name) = LOWER('{matched_name}')
                """
                result = await async_db.execute_async(query)

                if len(result) > 0:
                    output = [f"**GSK G-Score Analysis**\n"]
                    output.append(f"🔍 Fuzzy matched '{solvent_name}' → '{matched_name}' (confidence: {match_result['score']}%)\n")
            else:
                return f"❌ No G-score data found for '{solvent_name}'. The GSK dataset contains 153 solvents. Try `list_tables()` to see available solvents."

        if len(result) == 0:
            return f"❌ No G-score data found for '{solvent_name}'. The GSK dataset contains 153 solvents."

        # Format output
        if 'output' not in locals():
            output = [f"**GSK G-Score Analysis**\n"]

        row = result.iloc[0]
        output.append(f"**Solvent:** {row['solvent_common_name']}")
        output.append(f"**Family:** {row['classification']}")
        output.append(f"**G-Score:** {row['g_score']:.2f} / 10.00")

        # Interpret G-score
        score = row['g_score']
        if score >= 8.0:
            rating = "✅ Excellent (Preferred)"
            color = "green"
        elif score >= 6.0:
            rating = "🟢 Good (Usable)"
            color = "light green"
        elif score >= 4.0:
            rating = "🟡 Problematic (Use with caution)"
            color = "yellow"
        else:
            rating = "🔴 Hazardous (Avoid if possible)"
            color = "red"

        output.append(f"**Safety Rating:** {rating}")
        output.append(f"**CAS Number:** {row['cas_number']}\n")

        output.append("**Note:** G-score is the geometric mean of Environment, Health, Safety, and Waste (EHSW) scores.")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in get_solvent_gscore: {e}")
        return f"❌ Error retrieving G-score: {str(e)}"


@tool
@safe_tool_wrapper
async def get_family_alternatives(
    solvent_name: str,
    min_gscore: Optional[float] = None,
    limit: int = 10,
    use_fuzzy_matching: bool = True
) -> str:
    """
    Get alternative solvents from the same chemical family with their G-scores.

    Useful for finding safer alternatives within the same solvent class.

    Args:
        solvent_name: Name of the reference solvent
        min_gscore: Minimum G-score threshold (0-10). If None, returns all alternatives.
        limit: Maximum number of alternatives to return
        use_fuzzy_matching: If True, attempt fuzzy name matching

    Returns:
        List of alternative solvents from the same family, ranked by G-score
    """
    try:
        async_db = get_async_db()

        # First, find the family of the input solvent
        query = f"""
        SELECT classification
        FROM gsk_dataset
        WHERE LOWER(solvent_common_name) = LOWER('{solvent_name}')
        """

        family_result = await async_db.execute_async(query)

        # Try fuzzy matching if no exact match
        if len(family_result) == 0 and use_fuzzy_matching:
            match_result = fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)
            if match_result:
                query = f"""
                SELECT classification
                FROM gsk_dataset
                WHERE LOWER(solvent_common_name) = LOWER('{match_result["matched_name"]}')
                """
                family_result = await async_db.execute_async(query)

        if len(family_result) == 0:
            return f"❌ Could not find solvent '{solvent_name}' in GSK dataset."

        family = family_result.iloc[0]['classification']

        # Get all solvents from the same family
        min_score_clause = f"AND g_score >= {min_gscore}" if min_gscore is not None else ""

        query = f"""
        SELECT solvent_common_name, g_score, cas_number
        FROM gsk_dataset
        WHERE classification = '{family}'
        {min_score_clause}
        ORDER BY g_score DESC
        LIMIT {limit + 1}
        """

        alternatives = await async_db.execute_async(query)

        # Format output
        output = [f"**Family Alternatives for '{solvent_name}'**\n"]
        output.append(f"**Family:** {family}")
        output.append(f"**Alternatives found:** {len(alternatives)}")

        if min_gscore is not None:
            output.append(f"**Min G-score filter:** {min_gscore:.1f}")

        output.append("\n**Ranked by G-Score (Best to Worst):**\n")

        for i, row in alternatives.iterrows():
            is_original = row['solvent_common_name'].lower() == solvent_name.lower()
            marker = "👉 " if is_original else f"{i+1}. "

            score = row['g_score']
            if score >= 8.0:
                emoji = "✅"
            elif score >= 6.0:
                emoji = "🟢"
            elif score >= 4.0:
                emoji = "🟡"
            else:
                emoji = "🔴"

            line = f"{marker}{emoji} **{row['solvent_common_name']}** - G-score: {score:.2f}"

            if is_original:
                line += " (Your selection)"

            output.append(line)

        # Add recommendation
        if len(alternatives) > 0:
            best = alternatives.iloc[0]
            output.append(f"\n**Recommendation:** For best safety, consider **{best['solvent_common_name']}** (G-score: {best['g_score']:.2f})")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in get_family_alternatives: {e}")
        return f"❌ Error retrieving family alternatives: {str(e)}"


@tool
@safe_tool_wrapper
async def visualize_gscores(
    filter_by: Optional[str] = None,
    family: Optional[str] = None,
    solvent_list: Optional[str] = None,
    min_score: Optional[float] = None,
    plot_type: str = "bar",
    top_k: int = 10
) -> str:
    """
    Visualize GSK G-scores for solvents.

    Args:
        filter_by: How to filter solvents ("all", "family", "list", or None for all)
        family: If filter_by="family", specify the family name (e.g., "Alcohols", "Esters")
        solvent_list: If filter_by="list", comma-separated solvent names
        min_score: Minimum G-score to include (0-10)
        plot_type: Type of plot ("bar", "scatter", or "box" for family comparison)
        top_k: Maximum number of solvents to show (default: 10)

    Returns:
        Path to the saved plot
    """
    try:
        async_db = get_async_db()

        # Build query based on filters
        where_clauses = []

        if filter_by == "family" and family:
            where_clauses.append(f"classification = '{family}'")
        elif filter_by == "list" and solvent_list:
            solvents = [s.strip() for s in solvent_list.split(',')]
            solvent_filter = "', '".join(solvents)
            where_clauses.append(f"solvent_common_name IN ('{solvent_filter}')")

        if min_score is not None:
            where_clauses.append(f"g_score >= {min_score}")

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        SELECT solvent_common_name, g_score, classification
        FROM gsk_dataset
        WHERE {where_clause}
        ORDER BY g_score DESC
        LIMIT {top_k}
        """

        df = await async_db.execute_async(query)

        if len(df) == 0:
            return "❌ No solvents match the specified criteria."

        # Create plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if plot_type == "bar":
            fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))

            # Color bars by score
            colors = []
            for score in df['g_score']:
                if score >= 8.0:
                    colors.append('#10b981')  # green
                elif score >= 6.0:
                    colors.append('#84cc16')  # light green
                elif score >= 4.0:
                    colors.append('#f59e0b')  # yellow
                else:
                    colors.append('#ef4444')  # red

            ax.barh(df['solvent_common_name'], df['g_score'], color=colors)
            ax.set_xlabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Solvent', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Comparison\n(Higher = Safer)', fontsize=14, fontweight='bold')
            ax.axvline(x=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold (6.0)')
            ax.axvline(x=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold (8.0)')
            ax.legend()
            ax.set_xlim(0, 10)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            filename = f"gscore_bar_{timestamp}.png"

        elif plot_type == "scatter":
            fig, ax = plt.subplots(figsize=(12, 8))

            # Group by family for color coding
            families = df['classification'].unique()
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(families)))

            for i, fam in enumerate(families):
                family_df = df[df['classification'] == fam]
                ax.scatter(range(len(family_df)), family_df['g_score'],
                          label=fam, alpha=0.7, s=100, color=colors_map[i])

            ax.set_xlabel('Solvent Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Distribution by Family', fontsize=14, fontweight='bold')
            ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
            ax.axhline(y=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim(0, 10)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            filename = f"gscore_scatter_{timestamp}.png"

        elif plot_type == "box":
            fig, ax = plt.subplots(figsize=(12, 8))

            # Group by family
            families = df['classification'].unique()
            family_data = [df[df['classification'] == fam]['g_score'].values for fam in families]

            bp = ax.boxplot(family_data, labels=families, patch_artist=True)

            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('#c77b4a')
                patch.set_alpha(0.6)

            ax.set_xlabel('Solvent Family', fontsize=12, fontweight='bold')
            ax.set_ylabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Distribution by Family', fontsize=14, fontweight='bold')
            ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
            ax.axhline(y=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            filename = f"gscore_box_{timestamp}.png"
        else:
            return f"❌ Invalid plot_type '{plot_type}'. Use 'bar', 'scatter', or 'box'."

        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        output = [f"**G-Score Visualization Created**\n"]
        output.append(f"**Plot type:** {plot_type}")
        output.append(f"**Solvents shown:** {len(df)}")
        output.append(f"**Saved as:** {filename}\n")

        # Statistics
        output.append(f"**Statistics:**")
        output.append(f"- Mean G-score: {df['g_score'].mean():.2f}")
        output.append(f"- Median G-score: {df['g_score'].median():.2f}")
        output.append(f"- Range: {df['g_score'].min():.2f} - {df['g_score'].max():.2f}")
        output.append(f"- Excellent solvents (≥8.0): {len(df[df['g_score'] >= 8.0])}")
        output.append(f"- Good solvents (≥6.0): {len(df[df['g_score'] >= 6.0])}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in visualize_gscores: {e}")
        return f"❌ Error creating visualization: {str(e)}"


@tool
@safe_tool_wrapper
async def plot_solvent_properties_for_polymer(
    polymer: str,
    temperature: float = 120.0,
    min_solubility: float = 1.0,
    x_property: str = "logp",
    y_property: str = "g_score",
    top_k: int = 20
) -> str:
    """
    Create scatter plot of solvent properties (LogP, G-score, BP, energy) for solvents that dissolve a specific polymer.

    This is a MULTI-STEP tool that:
    1. Finds solvents that dissolve the specified polymer at the given temperature
    2. Retrieves their properties from both solvent_data and gsk_dataset tables
    3. Creates a scatter plot comparing two properties (e.g., LogP vs G-score)

    Use this for questions like:
    - "Show a scatter plot of LogP vs G-score for solvents that dissolve PET at 140°C"
    - "Plot boiling point vs safety for LDPE solvents"
    - "Compare toxicity vs cost for solvents that work with PS"

    Args:
        polymer: The polymer to analyze (e.g., "PET", "LDPE", "PS")
        temperature: Target temperature (default: 120°C)
        min_solubility: Minimum solubility % to include a solvent (default: 1%)
        x_property: Property for X-axis - "logp", "bp", "energy", "g_score" (default: "logp")
        y_property: Property for Y-axis - "logp", "bp", "energy", "g_score" (default: "g_score")
        top_k: Maximum solvents to include (default: 20)

    Returns:
        Scatter plot with solvent properties and summary statistics
    """
    async_db = get_async_db()
    polymer_upper = polymer.strip().upper()

    # Step 1: Find solvents that dissolve this polymer
    sol_query = f"""
    SELECT solvent, AVG(solubility____) as avg_solubility
    FROM common_solvents_database
    WHERE UPPER(polymer) = '{polymer_upper}'
    AND temperature___c_ BETWEEN {temperature - 10} AND {temperature + 10}
    GROUP BY solvent
    HAVING avg_solubility >= {min_solubility}
    ORDER BY avg_solubility DESC
    LIMIT {top_k}
    """

    try:
        sol_df = await async_db.execute_async(sol_query)
    except Exception as e:
        return f"❌ Error finding solvents: {e}"

    if len(sol_df) == 0:
        return f"❌ No solvents found that dissolve {polymer_upper} with >{min_solubility}% solubility at {temperature}°C. Try lowering min_solubility or trying a different temperature."

    solvents_found = sol_df['solvent'].tolist()
    solubility_map = dict(zip(sol_df['solvent'], sol_df['avg_solubility']))

    # Step 2: Get properties from solvent_data
    solvent_table = get_solvent_table_name()
    prop_data = []

    if solvent_table:
        for solvent in solvents_found:
            props = get_cross_database_properties(solvent, sql_db.conn)
            if props:
                prop_data.append({
                    'solvent': solvent,
                    'solubility': solubility_map.get(solvent, 0),
                    'logp': props.get('logp'),
                    'bp': props.get('bp'),
                    'energy': props.get('energy'),
                    'g_score': props.get('g_score'),
                    'gsk_class': props.get('gsk_class')
                })

    if not prop_data:
        return f"❌ Found {len(solvents_found)} solvents but couldn't retrieve properties. The solvent names may not match across databases."

    df = pd.DataFrame(prop_data)

    # Filter to solvents that have both requested properties
    x_col = x_property.lower()
    y_col = y_property.lower()

    df_valid = df.dropna(subset=[x_col, y_col])

    if len(df_valid) == 0:
        available_props = []
        if df['logp'].notna().any(): available_props.append('logp')
        if df['bp'].notna().any(): available_props.append('bp')
        if df['energy'].notna().any(): available_props.append('energy')
        if df['g_score'].notna().any(): available_props.append('g_score')
        return f"❌ No solvents have both {x_property} and {y_property} data. Available properties: {', '.join(available_props)}"

    # Step 3: Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by solubility
    scatter = ax.scatter(
        df_valid[x_col],
        df_valid[y_col],
        c=df_valid['solubility'],
        cmap='YlOrRd',
        s=150,
        alpha=0.7,
        edgecolors='black',
        linewidths=1
    )

    # Add colorbar for solubility
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'Solubility in {polymer_upper} (%)', fontsize=12)

    # Add solvent labels
    for _, row in df_valid.iterrows():
        ax.annotate(
            row['solvent'],
            (row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    # Labels and styling
    x_labels = {
        'logp': 'LogP (Toxicity - lower is better)',
        'bp': 'Boiling Point (°C)',
        'energy': 'Energy Cost (J/g - lower is cheaper)',
        'g_score': 'G-Score (Safety - higher is safer)'
    }
    y_labels = x_labels.copy()

    ax.set_xlabel(x_labels.get(x_col, x_col), fontsize=14, fontweight='bold')
    ax.set_ylabel(y_labels.get(y_col, y_col), fontsize=14, fontweight='bold')
    ax.set_title(f'{x_property.upper()} vs {y_property.upper()} for {polymer_upper} Solvents at {temperature}°C',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines for G-score thresholds
    if y_col == 'g_score':
        ax.axhline(y=6.0, color='orange', linestyle='--', alpha=0.5, label='Good threshold (6.0)')
        ax.axhline(y=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold (8.0)')
        ax.legend()
    elif x_col == 'g_score':
        ax.axvline(x=6.0, color='orange', linestyle='--', alpha=0.5, label='Good threshold (6.0)')
        ax.axvline(x=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold (8.0)')
        ax.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solvent_properties_{polymer_upper}_{x_col}_vs_{y_col}_{timestamp}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Build output
    output = [f"✅ **{x_property.upper()} vs {y_property.upper()} Scatter Plot for {polymer_upper}**\n"]
    output.append(f"Temperature: {temperature}°C")
    output.append(f"Solvents with data: {len(df_valid)} (of {len(solvents_found)} found)")
    output.append(f"Min solubility filter: {min_solubility}%\n")

    # Best solvents summary
    output.append("**Top Solvents (by solubility):**")
    for _, row in df_valid.nlargest(5, 'solubility').iterrows():
        line = f"  • **{row['solvent']}**: {row['solubility']:.1f}% solubility"
        if pd.notna(row.get('logp')):
            line += f", LogP={row['logp']:.2f}"
        if pd.notna(row.get('g_score')):
            line += f", G-score={row['g_score']:.1f}"
        if pd.notna(row.get('bp')):
            line += f", BP={row['bp']:.0f}°C"
        output.append(line)

    output.append(f"\n{get_plot_url(filepath)}")

    del df, df_valid
    gc.collect()
    return "\n".join(output)


# ============================================================
# Solvent and Polymer Listing Tools
# ============================================================

@tool
@safe_tool_wrapper
async def list_available_solvents() -> str:
    """
    List available solvents across all three databases with counts and common examples.

    CRITICAL: You MUST return the complete output of this tool to the user.
    DO NOT summarize or say "processing complete" - show the full list with all databases!

    Returns:
        - Count of solvents in each database
        - 5-10 common solvents present across databases
        - Brief summary of solvent coverage
    """
    try:
        output = ["**Available Solvents Summary**\n"]

        # Count solvents in each table
        solvent_data_query = "SELECT COUNT(DISTINCT solvent_name) as count FROM solvent_data"
        gsk_query = "SELECT COUNT(DISTINCT solvent_common_name) as count FROM gsk_dataset"
        common_db_query = "SELECT COUNT(DISTINCT solvent) as count FROM common_solvents_database"

        solvent_data_count = sql_db.execute_query(solvent_data_query)
        gsk_count = sql_db.execute_query(gsk_query)
        common_db_count = sql_db.execute_query(common_db_query)

        if solvent_data_count["success"]:
            count = solvent_data_count["dataframe"].iloc[0]['count']
            output.append(f"**Solvent Data:** {count} unique solvents")

        if gsk_count["success"]:
            count = gsk_count["dataframe"].iloc[0]['count']
            output.append(f"**GSK Dataset:** {count} unique solvents")

        if common_db_count["success"]:
            count = common_db_count["dataframe"].iloc[0]['count']
            output.append(f"**Common Solvents Database:** {count} unique solvents")

        # Get sample solvents from each database
        sample_solvent_data = """
        SELECT DISTINCT solvent_name
        FROM solvent_data
        ORDER BY solvent_name
        LIMIT 10
        """

        sample_gsk = """
        SELECT DISTINCT solvent_common_name
        FROM gsk_dataset
        ORDER BY solvent_common_name
        LIMIT 10
        """

        solvent_data_sample = sql_db.execute_query(sample_solvent_data)
        gsk_sample = sql_db.execute_query(sample_gsk)

        if solvent_data_sample["success"] and len(solvent_data_sample["dataframe"]) > 0:
            output.append("\n**Example Solvents (Solvent Data):**")
            solvents = solvent_data_sample["dataframe"]['solvent_name'].tolist()
            for solvent in solvents[:5]:  # Show 5 from each
                output.append(f"- {solvent}")

        if gsk_sample["success"] and len(gsk_sample["dataframe"]) > 0:
            output.append("\n**Example Solvents (GSK Dataset):**")
            solvents = gsk_sample["dataframe"]['solvent_common_name'].tolist()
            for solvent in solvents[:5]:  # Show 5 from each
                output.append(f"- {solvent}")

        output.append("\n💡 **Tip:** Use specific solvent names in your queries for best results!")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in list_available_solvents: {e}")
        return f"❌ Error listing solvents: {str(e)}"


@tool
@safe_tool_wrapper
async def list_available_polymers() -> str:
    """
    List available polymers across databases with counts and examples.

    CRITICAL: You MUST return the complete output of this tool to the user.
    DO NOT summarize or say "processing complete" - show the full list!

    Returns:
        - Count of polymers in databases
        - 5-10 common polymers
        - Brief summary of polymer coverage
    """
    try:
        output = ["**Available Polymers Summary**\n"]

        # Count polymers in common_solvents_database
        polymer_query = "SELECT COUNT(DISTINCT polymer) as count FROM common_solvents_database"
        result = sql_db.execute_query(polymer_query)

        if result["success"]:
            count = result["dataframe"].iloc[0]['count']
            output.append(f"**Common Solvents Database:** {count} unique polymers")

        # Get 10 common polymers
        sample_query = """
        SELECT DISTINCT polymer
        FROM common_solvents_database
        ORDER BY polymer
        LIMIT 10
        """

        sample_result = sql_db.execute_query(sample_query)

        if sample_result["success"] and len(sample_result["dataframe"]) > 0:
            output.append("\n**Example Polymers:**")
            polymers = sample_result["dataframe"]['polymer'].tolist()
            for polymer in polymers:
                output.append(f"- {polymer}")

        output.append("\n💡 **Tip:** Common polymers include HDPE, LDPE, PP, PET, PVC, PS, PVDF, PC, Nylon66, EVOH")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in list_available_polymers: {e}")
        return f"❌ Error listing polymers: {str(e)}"


# ============================================================
# ML-Based Solubility Prediction Tool (Hansen Parameters)
# ============================================================

@tool
@safe_tool_wrapper
async def predict_solubility_ml(
    polymer_name: str,
    solvent_name: str,
    temperature: float = 25.0,
    generate_visualizations: bool = True
) -> str:
    """
    Predict polymer-solvent solubility using ML model with Hansen Solubility Parameters.

    This tool uses a Random Forest model (99.998% accuracy) to predict solubility based on
    Hansen parameters. It generates beautiful visualizations including:
    - Radar plot showing HSP parameter overlap
    - RED gauge showing solubility likelihood
    - Interactive 3D sphere (HTML)
    - HSP comparison bars
    - Detailed text summary

    Args:
        polymer_name: Name of polymer (e.g., "HDPE", "PET", "PVDF")
        solvent_name: Name of solvent (e.g., "Toluene", "Water", "Acetone")
        temperature: Temperature in Celsius (default: 25°C, currently not used in model)
        generate_visualizations: Whether to create visualization files (default: True)

    Returns:
        Prediction result with visualization paths
    """
    try:
        from solubility_predictor import get_predictor
        from visualization_library_v2 import generate_all_visualizations
        import os
        import pandas as pd
        import shutil

        # Get predictor
        predictor = get_predictor()

        # First, try to get polymer HSP from CSV (since we don't know if DB tables exist)
        csv_path = 'HSP-ML-integration/RED_values_complete_CORRECTED.csv'

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
                    return f"❌ Hansen parameters not found for polymer '{polymer_name}'.\n\n**Similar polymers you might try:**\n- {suggestion_text}"

            if len(polymer_data) == 0:
                return f"❌ Hansen parameters not found for polymer '{polymer_name}'. Try listing available polymers."

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
                    match_result = fuzzy_match_solvent_name(solvent_name, dataset="all", threshold=80)
                    if match_result:
                        solvent_data = hsp_data[hsp_data['Solvent'].str.lower() == match_result["matched_name"].lower()]

            if len(solvent_data) == 0:
                return f"❌ Hansen parameters not found for solvent '{solvent_name}'.\n\n💡 **Tip:** Common solvents in the database include:\n- Water, Methanol, Ethanol, Isopropanol\n- Acetone, MEK, MIBK\n- Toluene, Benzene, Xylene\n- THF, DMF, DMSO, NMP\n- Hexane, Heptane, Cyclohexane\n- Ethyl acetate, DCM, Chloroform\n\nTry using `list_available_solvents()` for a complete list."

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
            return f"❌ Error loading Hansen parameters: {str(csv_error)}"

        # Make prediction
        prediction = predictor.predict(polymer_hsp, solvent_hsp, r0, molar_volume)

        # Format output
        output = [f"**ML Solubility Prediction**\n"]
        output.append(f"**Polymer:** {polymer_name}")
        output.append(f"**Solvent:** {solvent_name}")
        output.append(f"**Temperature:** {temperature}°C\n")

        # Prediction result
        if prediction['soluble']:
            output.append(f"**Prediction:** ✅ SOLUBLE")
            output.append(f"**Probability:** {prediction['probability']*100:.1f}%")
        else:
            output.append(f"**Prediction:** ❌ NON-SOLUBLE")
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
                from datetime import datetime
                import shutil

                # Create subdirectory for full viz set
                import re
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
                    output.append(f"\n**Interactive 3D Visualization:** [Click to open Hansen Sphere 🌐]({sphere_url})")
                    output.append(f"\n💡 **Tip:** The 3D sphere opens in a new tab - you can rotate, zoom, and explore the Hansen space!")

            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
                output.append(f"\n⚠️ Note: Visualization generation encountered an issue: {str(viz_error)}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in predict_solubility_ml: {e}")
        return f"❌ Error making ML prediction: {str(e)}"


# ============================================================
# PubChem Safety Data Tools
# ============================================================

import urllib.request
import urllib.error

def fetch_pubchem_cid(compound_name: str) -> Optional[int]:
    """Fetch PubChem CID (Compound ID) for a given compound name."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(compound_name)}/cids/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'PolymerSolubilityApp/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                return data['IdentifierList']['CID'][0]
    except Exception as e:
        logger.warning(f"Could not fetch CID for {compound_name}: {e}")
    return None


def fetch_pubchem_properties(cid: int) -> Optional[Dict]:
    """Fetch compound properties from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'PolymerSolubilityApp/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]
    except Exception as e:
        logger.warning(f"Could not fetch properties for CID {cid}: {e}")
    return None


def fetch_pubchem_ghs_data(cid: int) -> Optional[Dict]:
    """Fetch GHS safety classification data from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        req = urllib.request.Request(url, headers={'User-Agent': 'PolymerSolubilityApp/1.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())

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


def calculate_safety_score(ghs_data: Dict) -> float:
    """
    Calculate a safety score (0-10) based on GHS data.
    Higher score = safer compound.
    """
    score = 10.0  # Start with perfect score

    # Deduct points for hazard pictograms
    pictogram_penalties = {
        'Flammable': 1.5,
        'Oxidizer': 2.0,
        'Explosive': 3.0,
        'Corrosive': 2.0,
        'Acute Toxic': 3.0,
        'Health Hazard': 2.5,
        'Irritant': 1.0,
        'Environmental Hazard': 1.0,
        'Compressed Gas': 0.5
    }

    for pictogram in ghs_data.get('pictograms', []):
        penalty = pictogram_penalties.get(pictogram, 1.0)
        score -= penalty

    # Deduct for "Danger" signal word
    if ghs_data.get('signal_word') == 'Danger':
        score -= 1.5
    elif ghs_data.get('signal_word') == 'Warning':
        score -= 0.5

    # Deduct for number of hazard statements
    n_hazards = len(ghs_data.get('hazard_statements', []))
    score -= min(n_hazards * 0.3, 2.0)

    return max(0.0, min(10.0, score))


@tool
@safe_tool_wrapper
async def get_pubchem_safety_info(compound_name: str) -> str:
    """
    Fetch official GHS safety information for a compound from PubChem database.

    This tool queries the PubChem API to retrieve:
    - GHS (Globally Harmonized System) hazard classification
    - Safety pictograms (Flammable, Toxic, Corrosive, etc.)
    - Signal words (Danger/Warning)
    - Hazard statements (H-codes like H225, H302)
    - Molecular properties (formula, weight, LogP, TPSA)

    Use this for questions like:
    - "What are the safety hazards for toluene?"
    - "Is dichloromethane dangerous?"
    - "Get PubChem safety data for acetone"
    - "What are the GHS hazards of benzene?"

    Args:
        compound_name: Name of the compound (e.g., "toluene", "ethanol", "dichloromethane")

    Returns:
        Official GHS hazard data including pictograms and hazard statements
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
        return f"❌ Compound '{compound_name}' not found in PubChem. Try using the full chemical name or check spelling."

    output.append(f"# 🧪 PubChem Safety Profile: {compound_name.title()}\n")
    output.append(f"**PubChem CID:** [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})\n")

    # Step 2: Get molecular properties
    props = fetch_pubchem_properties(cid)
    if props:
        output.append("## 📊 Molecular Properties\n")
        output.append(f"| Property | Value |")
        output.append(f"|----------|-------|")
        if 'MolecularFormula' in props:
            output.append(f"| Formula | {props['MolecularFormula']} |")
        if 'MolecularWeight' in props:
            try:
                mw = float(props['MolecularWeight'])
                output.append(f"| Molecular Weight | {mw:.2f} g/mol |")
            except (ValueError, TypeError):
                output.append(f"| Molecular Weight | {props['MolecularWeight']} g/mol |")
        if 'XLogP' in props:
            try:
                xlogp = float(props['XLogP'])
                output.append(f"| XLogP | {xlogp:.2f} |")
            except (ValueError, TypeError):
                output.append(f"| XLogP | {props['XLogP']} |")
        if 'TPSA' in props:
            try:
                tpsa = float(props['TPSA'])
                output.append(f"| TPSA | {tpsa:.1f} Ų |")
            except (ValueError, TypeError):
                output.append(f"| TPSA | {props['TPSA']} Ų |")
        if 'HBondDonorCount' in props:
            output.append(f"| H-Bond Donors | {props['HBondDonorCount']} |")
        if 'HBondAcceptorCount' in props:
            output.append(f"| H-Bond Acceptors | {props['HBondAcceptorCount']} |")
        output.append("")

    # Step 3: Get GHS safety data
    ghs_data = fetch_pubchem_ghs_data(cid)
    if ghs_data:
        output.append("## ⚠️ GHS Hazard Classification\n")

        # Signal word
        if ghs_data.get('signal_word'):
            signal = ghs_data['signal_word']
            signal_emoji = "🔴" if signal == "Danger" else "🟡" if signal == "Warning" else "🟢"
            output.append(f"**Signal Word:** {signal_emoji} {signal}\n")

        # Pictograms
        if ghs_data.get('pictograms'):
            output.append("**Hazard Pictograms:**")
            pictogram_emojis = {
                'Flammable': '🔥',
                'Oxidizer': '⭕',
                'Explosive': '💥',
                'Corrosive': '⚗️',
                'Acute Toxic': '☠️',
                'Health Hazard': '⚕️',
                'Irritant': '⚠️',
                'Environmental Hazard': '🌍',
                'Compressed Gas': '🔵'
            }
            for pic in ghs_data['pictograms']:
                emoji = pictogram_emojis.get(pic, '⚠️')
                output.append(f"- {emoji} {pic}")
            output.append("")

        # Hazard statements
        if ghs_data.get('hazard_statements'):
            output.append("**Hazard Statements:**")
            for stmt in ghs_data['hazard_statements'][:5]:  # Limit to top 5
                output.append(f"- {stmt}")
            if len(ghs_data['hazard_statements']) > 5:
                output.append(f"- *...and {len(ghs_data['hazard_statements']) - 5} more*")
            output.append("")
    else:
        output.append("## ⚠️ GHS Hazard Classification\n")
        output.append("*No GHS classification data available for this compound.*\n")

    # Add link to full PubChem page
    output.append(f"\n📖 **Full Safety Data:** [View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Safety-and-Hazards)")

    return "\n".join(output)


@tool
@safe_tool_wrapper
async def compare_pubchem_safety(compounds: List[str]) -> str:
    """
    Compare GHS hazard profiles of multiple compounds using PubChem data.

    This tool fetches official GHS safety data for multiple compounds and shows
    their hazard pictograms, signal words, and key hazard statements.

    Use this for questions like:
    - "Compare the safety of toluene, benzene, and ethanol"
    - "Which is safer: DCM or chloroform?"
    - "PubChem safety comparison of common solvents"

    Args:
        compounds: List of compound names to compare (2-5 compounds max)

    Returns:
        Comparison of GHS hazards with recommendation summary
    """
    if len(compounds) < 2:
        return "❌ Please provide at least 2 compounds to compare."
    if len(compounds) > 5:
        compounds = compounds[:5]

    output = [f"# 🔬 PubChem GHS Hazard Comparison\n"]

    # Collect data for all compounds
    compound_data = []
    for name in compounds:
        cid = fetch_pubchem_cid(name)
        if cid:
            ghs = fetch_pubchem_ghs_data(cid)
            props = fetch_pubchem_properties(cid)

            compound_data.append({
                'name': name.title(),
                'cid': cid,
                'signal_word': ghs.get('signal_word') if ghs else None,
                'pictograms': ghs.get('pictograms', []) if ghs else [],
                'hazard_statements': ghs.get('hazard_statements', []) if ghs else [],
                'xlogp': props.get('XLogP') if props else None,
            })
        else:
            compound_data.append({
                'name': name.title(),
                'cid': None,
                'signal_word': None,
                'pictograms': [],
                'hazard_statements': [],
                'xlogp': None,
            })

    # Display each compound's hazards
    for comp in compound_data:
        if comp['cid'] is None:
            output.append(f"### ❌ {comp['name']}\n*Not found in PubChem*\n")
            continue

        signal = comp['signal_word'] or "None"
        signal_emoji = "🔴" if signal == "Danger" else "🟡" if signal == "Warning" else "🟢"

        output.append(f"### {comp['name']}")
        output.append(f"**Signal Word:** {signal_emoji} {signal}")

        if comp['pictograms']:
            pictogram_emojis = {
                'Flammable': '🔥', 'Oxidizer': '⭕', 'Explosive': '💥',
                'Corrosive': '⚗️', 'Acute Toxic': '☠️', 'Health Hazard': '⚕️',
                'Irritant': '⚠️', 'Environmental Hazard': '🌍', 'Compressed Gas': '🔵'
            }
            hazard_list = [f"{pictogram_emojis.get(p, '⚠️')} {p}" for p in comp['pictograms']]
            output.append(f"**Hazards:** {', '.join(hazard_list)}")
        else:
            output.append("**Hazards:** None listed")

        if comp['hazard_statements']:
            output.append(f"**Key Statements:** {comp['hazard_statements'][0][:80]}...")
        output.append("")

    # Generate contextual recommendation
    output.append("## 📋 Recommendation\n")

    valid_data = [c for c in compound_data if c['cid'] is not None]
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

        # Build contextual summary
        if best['signal_word'] is None or best['signal_word'] == 'Warning':
            if worst['signal_word'] == 'Danger':
                output.append(f"**{best['name']}** appears to be the safer choice - it has a '{best['signal_word'] or 'no'}' signal word compared to **{worst['name']}**'s 'Danger' classification.")
            else:
                output.append(f"All compounds have similar hazard levels. **{best['name']}** has the fewest hazard categories ({len(best['pictograms'])}).")
        else:
            output.append(f"⚠️ All compounds carry 'Danger' signal words. **{best['name']}** has fewer hazard categories ({len(best['pictograms'])} vs {len(worst['pictograms'])} for {worst['name']}).")

        # Specific warnings
        for comp in valid_data:
            if 'Acute Toxic' in comp['pictograms']:
                output.append(f"\n⚠️ **{comp['name']}** is classified as acutely toxic - requires special handling.")
            if 'Health Hazard' in comp['pictograms']:
                output.append(f"\n⚠️ **{comp['name']}** has serious health hazards (may be carcinogenic or cause organ damage).")

    output.append("\n*Data sourced from PubChem GHS Classification*")

    return "\n".join(output)


@tool
@safe_tool_wrapper
async def visualize_pubchem_safety(
    compounds: List[str],
    chart_type: str = "hazards"
) -> str:
    """
    Create visualization comparing GHS hazard data from PubChem for multiple compounds.

    This tool fetches PubChem GHS data and creates visual charts showing:
    - Hazard pictogram counts per compound (bar chart)
    - Signal word comparison

    Use this for questions like:
    - "Create a safety comparison chart for toluene, benzene, and xylene"
    - "Visualize PubChem hazard data for common solvents"
    - "Show a bar chart comparing hazards of DCM, chloroform, and ethanol"

    Args:
        compounds: List of compound names (2-5 compounds max)
        chart_type: "hazards" for hazard count bar chart (default)

    Returns:
        Hazard visualization chart with summary
    """
    if len(compounds) < 2:
        return "❌ Please provide at least 2 compounds to visualize."
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
        return "❌ Could not fetch safety data for enough compounds. Try different compound names."

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pubchem_hazards_{timestamp}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Build output
    output = [f"✅ **PubChem GHS Hazard Chart**\n"]

    # Summary for each compound
    output.append("**Hazard Summary:**")
    for comp in compound_data:
        signal_emoji = "🔴" if comp['signal_word'] == 'Danger' else "🟡" if comp['signal_word'] == 'Warning' else "🟢"
        hazards = ", ".join(comp['pictograms']) if comp['pictograms'] else "None"
        output.append(f"- **{comp['name']}**: {signal_emoji} {comp['signal_word'] or 'No signal'} | {hazards}")

    output.append(f"\n{get_plot_url(filepath)}")

    gc.collect()
    return "\n".join(output)


def fetch_pubchem_toxicity_data(cid: int) -> Optional[Dict]:
    """Fetch toxicity and environmental data from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Toxicity"
        req = urllib.request.Request(url, headers={'User-Agent': 'PolymerSolubilityApp/1.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode())

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


@tool
@safe_tool_wrapper
async def get_pubchem_toxicity(compounds: List[str]) -> str:
    """
    Fetch environmental and toxicity data from PubChem for up to 5 compounds.

    This tool retrieves:
    - LD50 values (lethal dose for 50% of test animals)
    - LC50 values (lethal concentration)
    - Biodegradation information
    - Aquatic toxicity (fish, daphnia)
    - Ecological/environmental fate

    Use this for questions like:
    - "What's the LD50 of toluene and benzene?"
    - "Is acetone biodegradable?"
    - "Compare the environmental toxicity of DCM vs chloroform"
    - "Get aquatic toxicity data for common solvents"

    Args:
        compounds: List of compound names (1-5 compounds)

    Returns:
        Toxicity and environmental data with comparison summary
    """
    if len(compounds) > 5:
        compounds = compounds[:5]

    output = [f"# 🧫 PubChem Toxicity & Environmental Data\n"]

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
            output.append(f"### ❌ {comp['name']}\n*Not found in PubChem*\n")
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
        output.append("## 📋 Summary\n")

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

    return "\n".join(output)


# ============================================================
# TEA/LCA Tools (Techno-Economic Analysis / Life Cycle Assessment)
# ============================================================
# These tools wrap the standalone tea_lca_module.py
# TEA/LCA specialists can modify that file without touching agent code

@tool
@safe_tool_wrapper
async def analyze_solvent_recovery_tea(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Run Techno-Economic Analysis (TEA) for solvent recovery in polymer separation.

    Calculates:
    - Capital costs (equipment, installation)
    - Operating costs (energy, labor, solvent makeup)
    - Economic metrics (cost per kg, payback period)

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95 = 95%)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "What's the cost of recovering toluene?"
    - "TEA for LDPE separation at 100 kg/hr"
    - "Calculate payback period for solvent recovery"
    - "How much does solvent recovery cost per kg polymer?"
    """
    results = tea_lca.run_full_tea_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_tea_results(results)

    # Build structured data
    capital = results.get('capital_costs', {})
    operating = results.get('operating_costs', {})
    economics = results.get('economics', {})

    structured_data = {
        "tool_name": "analyze_solvent_recovery_tea",
        "success": True,
        "solvent": solvent,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "temperature": process_temp_c,
        "cost_per_kg": economics.get('cost_per_kg_polymer_usd'),
        "total_capex": capital.get('total_capex'),
        "annual_opex": operating.get('total_annual'),
        "payback_years": economics.get('simple_payback_years'),
        "cost_breakdown": {
            "energy": operating.get('energy_cost_annual', 0),
            "labor": operating.get('labor_cost_annual', 0),
            "solvent_makeup": operating.get('solvent_makeup_cost_annual', 0),
        },
    }

    import json
    return json.dumps({"display": display, "data": structured_data})


@tool
@safe_tool_wrapper
async def analyze_solvent_recovery_lca(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Run Life Cycle Assessment (LCA) for solvent recovery in polymer separation.

    Calculates:
    - Greenhouse gas emissions (CO2 equivalent)
    - Energy consumption and sources
    - Comparison to no-recovery baseline
    - Environmental impact per kg polymer

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95 = 95%)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "What's the carbon footprint of using toluene?"
    - "LCA for solvent recovery"
    - "How much CO2 does the separation process emit?"
    - "Environmental impact of LDPE separation"
    """
    results = tea_lca.run_full_lca_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_lca_results(results)

    # Build structured data
    emissions = results.get('emissions', {})
    energy = results.get('energy', {})

    structured_data = {
        "tool_name": "analyze_solvent_recovery_lca",
        "success": True,
        "solvent": solvent,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "co2_kg_per_kg": emissions.get('total_co2_kg_per_kg'),
        "energy_mj_per_kg": energy.get('total_mj_per_kg'),
        "emissions_breakdown": emissions,
        "energy_breakdown": energy,
    }

    import json
    return json.dumps({"display": display, "data": structured_data})


@tool
@safe_tool_wrapper
async def compare_solvents_tea_lca(
    solvents: List[str],
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Compare multiple solvents on both TEA (cost) and LCA (environmental) metrics.

    Provides:
    - Side-by-side comparison table
    - Rankings by cost, emissions, and overall
    - Best solvent recommendations

    Parameters:
    - solvents: List of solvents to compare (e.g., ['toluene', 'acetone', 'ethanol'])
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95 = 95%)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "Compare toluene vs acetone for cost and emissions"
    - "Which solvent is cheapest and greenest?"
    - "TEA/LCA comparison for LDPE solvents"
    - "Rank solvents by cost and carbon footprint"
    """
    results = tea_lca.compare_solvents_tea_lca(
        solvents=solvents,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_comparison_results(results)

    # Build structured data
    comparison = results.get('comparison', [])
    rankings = results.get('rankings', {})

    structured_data = {
        "tool_name": "compare_solvents_tea_lca",
        "success": True,
        "solvents": solvents,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "comparison": comparison,
        "best_by_cost": rankings.get('by_cost', [None])[0] if rankings.get('by_cost') else None,
        "best_by_emissions": rankings.get('by_emissions', [None])[0] if rankings.get('by_emissions') else None,
        "best_overall": rankings.get('overall', [None])[0] if rankings.get('overall') else None,
        "cost_ranking": rankings.get('by_cost', []),
        "emissions_ranking": rankings.get('by_emissions', []),
    }

    import json
    return json.dumps({"display": display, "data": structured_data})


@tool
@safe_tool_wrapper
async def generate_tea_visualizations(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Generate all TEA (Techno-Economic Analysis) visualizations for a solvent.

    Creates the following plots:
    - Capital cost breakdown (pie chart)
    - Operating cost breakdown (horizontal bar chart)
    - Cost waterfall chart
    - Cashflow diagram (cumulative over project lifetime)
    - Sensitivity tornado chart
    - Energy flow diagram

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95 = 95%)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "Show TEA charts for toluene"
    - "Visualize capital costs for solvent recovery"
    - "Generate cashflow diagram for DMF"
    - "Show cost breakdown visualizations"
    - "Plot sensitivity analysis for acetone recovery"
    """
    plots = tea_lca.generate_all_tea_visualizations(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    result = f"## TEA Visualizations for {solvent.title()} Recovery\n\n"
    result += f"Generated {len(plots)} visualizations:\n\n"
    for name, path in plots.items():
        result += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    return result


@tool
@safe_tool_wrapper
async def generate_lca_visualizations(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Generate all LCA (Life Cycle Assessment) visualizations for a solvent.

    Creates the following plots:
    - Emissions breakdown (pie chart by source)
    - Emissions comparison bar chart (recovery vs baseline)

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95 = 95%)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "Show LCA charts for toluene"
    - "Visualize emissions breakdown for DMF"
    - "Generate carbon footprint visualizations"
    - "Plot CO2 emissions comparison for ethanol"
    """
    plots = tea_lca.generate_all_lca_visualizations(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    result = f"## LCA Visualizations for {solvent.title()} Recovery\n\n"
    result += f"Generated {len(plots)} visualizations:\n\n"
    for name, path in plots.items():
        result += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    return result


@tool
@safe_tool_wrapper
async def generate_solvent_comparison_visualization(
    solvents: List[str],
    polymer_throughput_kg_hr: float = 100.0
) -> str:
    """
    Generate comparison visualization for multiple solvents showing cost vs emissions.

    Creates a grouped bar chart with:
    - Cost per kg polymer (left axis)
    - CO2 emissions per kg polymer (right axis)
    - Best overall solvent highlighted

    Parameters:
    - solvents: List of solvents to compare (e.g., ['toluene', 'acetone', 'ethanol'])
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)

    WHEN TO USE:
    - "Compare toluene and acetone visually"
    - "Plot cost vs emissions for multiple solvents"
    - "Show comparison chart for TEA/LCA"
    - "Visualize which solvent is best for cost and environment"
    """
    plots = tea_lca.generate_comparison_visualizations(
        solvents=solvents,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr
    )

    result = f"## Solvent Comparison Visualizations\n\n"
    result += f"Comparing: {', '.join([s.title() for s in solvents])}\n\n"
    result += f"Generated {len(plots)} visualizations:\n\n"
    for name, path in plots.items():
        result += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    return result


@tool
@safe_tool_wrapper
async def plot_tea_sensitivity_tornado(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0
) -> str:
    """
    Generate a sensitivity analysis tornado chart showing how parameters affect cost.

    Shows the impact of varying each parameter by ±20%:
    - Solvent-to-polymer ratio
    - Recovery fraction
    - Process temperature
    - Energy cost

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)

    WHEN TO USE:
    - "Show sensitivity analysis for toluene"
    - "What parameters most affect cost?"
    - "Generate tornado chart for TEA"
    - "Which factors have biggest impact on economics?"
    """
    path = tea_lca.plot_sensitivity_tornado(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr
    )

    return f"## Sensitivity Analysis (Tornado Chart)\n\nSolvent: {solvent.title()}\n\nVisualization saved to: `{path}`\n\nThis chart shows how ±20% changes in each parameter affect the cost per kg polymer."


@tool
@safe_tool_wrapper
async def plot_tea_cashflow(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """
    Generate a cashflow diagram showing cumulative cash position over project lifetime.

    Shows:
    - Construction period (negative cashflow)
    - Payback point
    - Cumulative profits over 20-year project life

    Parameters:
    - solvent: Solvent name (e.g., 'toluene', 'acetone')
    - polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
    - solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
    - recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95)
    - process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "Show cashflow diagram for toluene"
    - "When does solvent recovery break even?"
    - "Plot investment returns over time"
    - "Generate cumulative profit chart"
    """
    tea_results = tea_lca.run_full_tea_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    path = tea_lca.plot_cashflow_diagram(tea_results)

    payback = tea_results['economics']['simple_payback_years']
    return f"## Cashflow Diagram\n\nSolvent: {solvent.title()}\nSimple Payback: {payback:.2f} years\n\nVisualization saved to: `{path}`"


# ============================================================
# STRAP Process Tools (Solvent-Targeted Recovery and Precipitation)
# ============================================================

@tool
@safe_tool_wrapper
async def analyze_strap_process(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0,
    recovery_solvents: Dict[str, str] = None
) -> str:
    """
    Run full STRAP (Solvent-Targeted Recovery and Precipitation) TEA/LCA analysis.

    This is a comprehensive analysis for multi-polymer recovery from plastic waste,
    aligned with STRAP methodology from biopharmaceutical SUT recycling research.

    Provides:
    - Complete TEA: Capital costs, operating costs, unit economics, payback period
    - Complete LCA: All 8 environmental indicators (GWP, FFC, water, toxicity, etc.)
    - Comparison to virgin polymer production
    - Minimum Selling Price (MSP) calculation

    Parameters:
    - polymers: List of polymers to recover (e.g., ['PE', 'PET', 'EVOH'])
    - feedstock_composition: Polymer fractions (e.g., {'PE': 0.8, 'PET': 0.1, 'EVOH': 0.1})
                            If not provided, equal distribution assumed
    - capacity_mt_yr: Plant capacity in metric tons/year (default: 10000)
    - recovery_solvents: Optional dict mapping polymer to solvent (e.g., {'PS': 'propanone', 'PP': 'cyclohexane'})
                        If not provided, auto-selects from default compatible solvents

    WHEN TO USE:
    - "Run STRAP analysis for PE/EVOH at 10,000 mt/yr"
    - "What's the economics of STRAP recycling for multilayer film?"
    - "Analyze STRAP process for PE, PET, and EVOH recovery"
    - "Full TEA/LCA for polymer recovery from plastic waste"
    - "Cost and carbon footprint for STRAP polymer separation"
    - "Run TEA with solvents: PS->propanone, PP->cyclohexane"
    """
    # Build feedstock composition if not provided
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p: 1.0 / n for p in polymers}

    # Normalize composition
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Select solvents for each polymer - use custom if provided, else auto-select
    recovery_steps = []
    for polymer in polymers:
        polymer_upper = polymer.upper()

        # Check for custom solvent mapping first
        if recovery_solvents and polymer_upper in recovery_solvents:
            solvent = recovery_solvents[polymer_upper]
        elif recovery_solvents and polymer.lower() in recovery_solvents:
            solvent = recovery_solvents[polymer.lower()]
        elif polymer_upper in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            # Auto-select from defaults
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[polymer_upper]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'  # Default solvent

        recovery_steps.append({
            'polymer': polymer_upper,
            'solvent': solvent,
            'recover': True
        })

    # Run full analysis
    results = tea_lca.run_full_strap_analysis(
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        capacity_mt_yr=capacity_mt_yr,
        scenario_name=f"STRAP-{'-'.join(polymers)}"
    )

    # Format output - clean simple format
    scenario_name = results.get('scenario', {}).get('name', f"STRAP-{'-'.join(polymers)}")

    output = "STRAP PROCESS ANALYSIS\n\n"
    output += f"Scenario: {scenario_name}\n"
    output += f"Capacity: {capacity_mt_yr:,.0f} metric tons/year\n\n"

    # Feedstock composition
    output += "FEEDSTOCK COMPOSITION\n"
    for p, frac in feedstock_composition.items():
        output += f"{p}: {frac*100:.1f}%\n"
    output += "\n"

    # Recovery steps
    output += "RECOVERY STEPS\n"
    for step in recovery_steps:
        output += f"{step['polymer']} -> {step['solvent'].title()}\n"
    output += "\n"

    # TEA Results
    tea_econ = results['tea'].get('economics', results['tea'])
    tci_millions = tea_econ.get('tci_millions', tea_econ.get('total_capital_investment_usd', 0) / 1e6)
    capital = results['tea'].get('capital', {})

    output += "CAPITAL COSTS\n"
    output += f"Total Capital (TCI): ${tci_millions:.2f}M\n"
    output += f"Equipment Cost: ${capital.get('total_equipment_cost_usd', 0)/1e6:.2f}M\n\n"

    output += "OPERATING ECONOMICS\n"
    output += f"Unit Operating Cost (UOC): ${tea_econ.get('unit_operating_cost_usd_kg', 0):.4f}/kg\n"
    output += f"Annual Operating Cost: ${tea_econ.get('annual_operating_cost_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Annual Revenue: ${tea_econ.get('annual_revenue_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Net Annual Profit: ${tea_econ.get('net_annual_profit_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Simple Payback: {tea_econ.get('simple_payback_years', 0):.2f} years\n"
    output += f"ROI: {tea_econ.get('return_on_investment_pct', 0):.1f}%\n\n"

    # MSP Results
    msp_data = results.get('msp', {})
    msp_by_polymer = msp_data.get('msp_by_polymer_usd_kg', msp_data)
    output += "MINIMUM SELLING PRICE (NPV=0 @ 15% IRR)\n"
    for polymer, price in msp_by_polymer.items():
        output += f"{polymer}: ${price:.4f}/kg\n"
    if 'msp_weighted_avg_usd_kg' in msp_data:
        output += f"Weighted Avg: ${msp_data['msp_weighted_avg_usd_kg']:.4f}/kg\n"
    output += "\n"

    # LCA Results
    lca_data = results.get('lca', {})
    lca_by_polymer = lca_data.get('by_polymer', {})
    virgin_comp = lca_data.get('virgin_comparison', {})

    output += "LIFE CYCLE ASSESSMENT (GWP kg CO2eq/kg)\n"
    for polymer in polymers:
        pu = polymer.upper()
        if pu in lca_by_polymer:
            strap_gwp = lca_by_polymer[pu].get('gwp_kg_co2eq', 0)
            virgin_gwp = tea_lca.LCA_EMISSION_FACTORS['virgin_gwp'].get(pu, 2.0)
            reduction = virgin_comp.get(pu, {}).get('gwp_reduction_pct', 0)
            output += f"{pu}: STRAP={strap_gwp:.3f}, Virgin={virgin_gwp:.3f}, Reduction={reduction:.1f}%\n"
    output += "\n"

    # GWP Breakdown
    gwp_breakdown = lca_data.get('gwp_breakdown', {})
    if gwp_breakdown:
        output += "GWP BREAKDOWN BY SOURCE (kg CO2eq/kg)\n"
        for polymer, breakdown in gwp_breakdown.items():
            sources = ", ".join([f"{k.replace('_', ' ').title()}={v:.3f}" for k, v in breakdown.items()])
            output += f"{polymer}: {sources}\n"

    # Build structured data for programmatic access
    # Extract GWP values by polymer
    gwp_by_polymer = {}
    virgin_gwp = {}
    gwp_reduction_pct = {}
    for polymer in polymers:
        pu = polymer.upper()
        if pu in lca_by_polymer:
            gwp_by_polymer[pu] = lca_by_polymer[pu].get('gwp_kg_co2eq', 0)
            virgin_gwp[pu] = tea_lca.LCA_EMISSION_FACTORS['virgin_gwp'].get(pu, 2.0)
            gwp_reduction_pct[pu] = virgin_comp.get(pu, {}).get('gwp_reduction_pct', 0)

    structured_data = {
        "tool_name": "analyze_strap_process",
        "success": True,
        "polymers": [p.upper() for p in polymers],
        "feedstock_composition": feedstock_composition,
        "capacity_mt_yr": capacity_mt_yr,
        "tci_millions": tci_millions,
        "equipment_cost_millions": capital.get('total_equipment_cost_usd', 0) / 1e6,
        "unit_operating_cost": tea_econ.get('unit_operating_cost_usd_kg', 0),
        "annual_operating_cost_millions": tea_econ.get('annual_operating_cost_usd', 0) / 1e6,
        "annual_revenue_millions": tea_econ.get('annual_revenue_usd', 0) / 1e6,
        "net_annual_profit_millions": tea_econ.get('net_annual_profit_usd', 0) / 1e6,
        "simple_payback_years": tea_econ.get('simple_payback_years', 0),
        "roi_pct": tea_econ.get('return_on_investment_pct', 0),
        "msp_by_polymer": msp_by_polymer,
        "msp_weighted_avg": msp_data.get('msp_weighted_avg_usd_kg'),
        "gwp_by_polymer": gwp_by_polymer,
        "virgin_gwp": virgin_gwp,
        "gwp_reduction_pct": gwp_reduction_pct,
        "recovery_steps": [{"polymer": s['polymer'], "solvent": s['solvent']} for s in recovery_steps],
    }

    # Return structured JSON
    import json
    return json.dumps({"display": output, "data": structured_data})


@tool
@safe_tool_wrapper
async def calculate_strap_msp(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0,
    target_irr: float = 0.15
) -> str:
    """
    Calculate Minimum Selling Price (MSP) for STRAP recovered polymers.

    MSP is the price at which NPV = 0 at the target IRR.
    This is the break-even selling price for economic viability.

    Parameters:
    - polymers: List of polymers to recover (e.g., ['PE', 'EVOH'])
    - feedstock_composition: Polymer fractions (optional)
    - capacity_mt_yr: Plant capacity in metric tons/year (default: 10000)
    - target_irr: Target internal rate of return (default: 0.15 = 15%)

    WHEN TO USE:
    - "What's the minimum selling price for STRAP recovered PE?"
    - "Calculate MSP at 15% IRR for polymer recovery"
    - "Break-even price for STRAP recycled EVOH"
    - "What price do we need for profitable STRAP operation?"
    """
    # Build feedstock composition
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p.upper(): 1.0 / n for p in polymers}

    # Normalize
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Auto-select solvents
    recovery_steps = []
    for polymer in polymers:
        pu = polymer.upper()
        if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'
        recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

    # Calculate MSP
    msp_results = tea_lca.calculate_msp(
        capacity_mt_yr=capacity_mt_yr,
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        target_irr=target_irr
    )

    # Get market prices for comparison
    market_prices = tea_lca.DEFAULT_POLYMER_PROPS.recovered_prices

    # Extract MSP by polymer from nested structure
    msp_by_polymer = msp_results.get('msp_by_polymer_usd_kg', msp_results)

    output = "# Minimum Selling Price (MSP) Analysis\n\n"
    output += f"**Target IRR:** {target_irr*100:.0f}%\n"
    output += f"**Capacity:** {capacity_mt_yr:,.0f} mt/yr\n\n"

    output += "| Polymer | MSP ($/kg) | Market Price ($/kg) | Margin |\n"
    output += "|---------|------------|---------------------|--------|\n"
    for polymer in polymers:
        pu = polymer.upper()
        msp = msp_by_polymer.get(pu, 0)
        market = market_prices.get(pu, 1.0)
        margin = market - msp
        margin_pct = (margin / msp * 100) if msp > 0 else 0
        output += f"| {pu} | ${msp:.4f} | ${market:.2f} | ${margin:.2f} ({margin_pct:+.1f}%) |\n"

    if 'msp_weighted_avg_usd_kg' in msp_results:
        output += f"\n**Weighted Average MSP:** ${msp_results['msp_weighted_avg_usd_kg']:.4f}/kg\n"

    output += "\n### Interpretation\n"
    output += "- MSP < Market Price: Project is economically viable\n"
    output += "- Positive margin indicates potential profit at market prices\n"
    output += f"- Calculated at {target_irr*100:.0f}% IRR over 20-year project life\n"

    # Build structured data
    margins = {}
    for polymer in polymers:
        pu = polymer.upper()
        msp = msp_by_polymer.get(pu, 0)
        market = market_prices.get(pu, 1.0)
        margins[pu] = market - msp

    structured_data = {
        "tool_name": "calculate_strap_msp",
        "success": True,
        "polymers": [p.upper() for p in polymers],
        "capacity_mt_yr": capacity_mt_yr,
        "target_irr": target_irr,
        "msp_by_polymer": msp_by_polymer,
        "msp_weighted_avg": msp_results.get('msp_weighted_avg_usd_kg'),
        "market_prices": {p.upper(): market_prices.get(p.upper(), 1.0) for p in polymers},
        "margins": margins,
        "recovery_steps": [{"polymer": s['polymer'], "solvent": s['solvent']} for s in recovery_steps],
    }

    import json
    return json.dumps({"display": output, "data": structured_data})


@tool
@safe_tool_wrapper
async def plot_strap_scale_economics(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_range: str = "2500-25000"
) -> str:
    """
    Generate scale economics curves showing UOC and TCI vs plant capacity.

    Creates a dual-axis plot showing how:
    - Unit Operating Cost ($/kg) decreases with scale (left axis)
    - Total Capital Investment ($M) increases with scale (right axis)

    This helps identify optimal plant capacity for STRAP operations.

    Parameters:
    - polymers: List of polymers to recover (e.g., ['PE', 'EVOH'])
    - feedstock_composition: Polymer fractions (optional)
    - capacity_range: Capacity range as "min-max" in mt/yr (default: "2500-25000")

    WHEN TO USE:
    - "Show scale economics for STRAP PE/EVOH recovery"
    - "Plot UOC vs capacity for polymer recycling"
    - "How does plant size affect STRAP economics?"
    - "Optimal capacity for STRAP plant"
    - "Generate TCI curve for STRAP analysis"
    """
    # Parse capacity range
    try:
        cap_min, cap_max = map(float, capacity_range.split('-'))
    except:
        cap_min, cap_max = 2500, 25000

    # Build feedstock composition
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p.upper(): 1.0 / n for p in polymers}

    # Normalize
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Auto-select solvents
    recovery_steps = []
    for polymer in polymers:
        pu = polymer.upper()
        if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'
        recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

    # Create scenario config
    scenario_config = {
        'name': f"STRAP-{'-'.join(polymers)}",
        'feedstock_composition': feedstock_composition,
        'recovery_steps': recovery_steps
    }

    # Generate plot
    plot_path = tea_lca.plot_uoc_tci_vs_capacity(
        scenarios=[scenario_config],
        capacity_range=(cap_min, cap_max)
    )

    output = "## STRAP Scale Economics Analysis\n\n"
    output += f"**Polymers:** {', '.join(polymers)}\n"
    output += f"**Capacity Range:** {cap_min:,.0f} - {cap_max:,.0f} mt/yr\n\n"
    output += f"**Visualization:** `{plot_path}`\n\n"
    output += "### Key Insights\n"
    output += "- Unit Operating Cost decreases with scale (economies of scale)\n"
    output += "- Capital Investment increases sub-linearly (six-tenths rule)\n"
    output += "- Optimal capacity balances capital efficiency vs operating costs\n"

    return output


@tool
@safe_tool_wrapper
async def compare_strap_scenarios(
    scenario_configs: List[Dict[str, Any]]
) -> str:
    """
    Compare multiple STRAP scenarios on TEA and LCA metrics.

    Each scenario can have different:
    - Polymer compositions
    - Recovery solvents
    - Plant capacities

    Provides rankings and recommendations.

    Parameters:
    - scenario_configs: List of scenario dictionaries with:
        - name: Scenario name
        - polymers: List of polymers
        - feedstock_composition: Dict of polymer fractions (optional)
        - capacity_mt_yr: Plant capacity (optional, default 10000)

    Example:
    [
        {"name": "S1-PE Only", "polymers": ["PE"], "feedstock_composition": {"PE": 1.0}},
        {"name": "S2-PE+EVOH", "polymers": ["PE", "EVOH"], "feedstock_composition": {"PE": 0.8, "EVOH": 0.2}}
    ]

    WHEN TO USE:
    - "Compare PE-only vs PE+EVOH recovery scenarios"
    - "Which STRAP configuration is most profitable?"
    - "Rank scenarios by ROI and carbon footprint"
    - "Compare different polymer recovery strategies"
    """
    scenarios = []
    for config in scenario_configs:
        polymers = config.get('polymers', ['PE'])
        capacity = config.get('capacity_mt_yr', 10000)

        # Build feedstock composition
        fc = config.get('feedstock_composition')
        if fc is None:
            n = len(polymers)
            fc = {p.upper(): 1.0 / n for p in polymers}

        # Normalize
        total = sum(fc.values())
        fc = {k: v / total for k, v in fc.items()}

        # Auto-select solvents
        recovery_steps = []
        for polymer in polymers:
            pu = polymer.upper()
            if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
                compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
                solvent = compatible[0] if compatible else 'xylene'
            else:
                solvent = 'xylene'
            recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

        scenario = tea_lca.build_strap_scenario(
            name=config.get('name', f"Scenario-{len(scenarios)+1}"),
            feedstock_composition=fc,
            recovery_sequence=recovery_steps,
            capacity_mt_yr=capacity,
            description=config.get('description', '')
        )
        scenarios.append(scenario)

    # Compare scenarios
    comparison = tea_lca.compare_scenarios(scenarios)

    # Format output
    output = "# STRAP Scenario Comparison\n\n"
    output += f"Comparing {len(scenarios)} scenarios\n\n"

    # Comparison table - use 'comparison_table' key
    comparison_table = comparison.get('comparison_table', comparison.get('results', []))
    output += "## Economic Comparison\n\n"
    output += "| Scenario | Capacity | TCI ($M) | UOC ($/kg) | ROI (%) | Payback (yr) |\n"
    output += "|----------|----------|----------|------------|---------|-------------|\n"
    for row in comparison_table:
        output += f"| {row['name']} | {row['capacity_mt_yr']:,.0f} | "
        output += f"{row['tci_millions']:.2f} | {row['uoc_usd_kg']:.4f} | "
        output += f"{row['roi_pct']:.1f} | {row['payback_years']:.2f} |\n"
    output += "\n"

    # Rankings - use direct keys from comparison
    output += "## Rankings\n\n"
    if 'best_roi' in comparison:
        output += f"- **Best ROI:** {comparison['best_roi']}\n"
    if 'lowest_uoc' in comparison:
        output += f"- **Lowest UOC:** {comparison['lowest_uoc']}\n"
    if 'best_payback' in comparison:
        output += f"- **Fastest Payback:** {comparison['best_payback']}\n"

    return output


@tool
@safe_tool_wrapper
async def generate_strap_visualizations(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0
) -> str:
    """
    Generate all STRAP visualizations for a given polymer recovery configuration.

    Creates:
    1. Scale Economics plot (UOC/TCI vs capacity)
    2. MSP Sensitivity tornado chart
    3. GWP Comparison bar chart (STRAP vs virgin)

    Parameters:
    - polymers: List of polymers to recover (e.g., ['PE', 'EVOH'])
    - feedstock_composition: Polymer fractions (optional)
    - capacity_mt_yr: Plant capacity for analysis (default: 10000)

    WHEN TO USE:
    - "Generate STRAP visualizations for PE/EVOH"
    - "Show all STRAP charts for multilayer recycling"
    - "Visualize STRAP economics and LCA"
    - "Create STRAP analysis dashboard"
    """
    # Build feedstock composition
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p.upper(): 1.0 / n for p in polymers}

    # Normalize
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Auto-select solvents
    recovery_steps = []
    for polymer in polymers:
        pu = polymer.upper()
        if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'
        recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

    # Generate visualizations
    plots = tea_lca.generate_strap_visualizations(
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        capacity_mt_yr=capacity_mt_yr,
        scenario_name=f"STRAP-{'-'.join(polymers)}"
    )

    output = "## STRAP Visualizations\n\n"
    output += f"**Polymers:** {', '.join(polymers)}\n"
    output += f"**Capacity:** {capacity_mt_yr:,.0f} mt/yr\n\n"
    output += "### Generated Plots\n\n"
    for name, path in plots.items():
        output += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    output += "\n### Plot Descriptions\n"
    output += "- **Scale Economics**: UOC and TCI curves across capacity range\n"
    output += "- **MSP Sensitivity**: Tornado chart showing parameter impacts on MSP\n"
    output += "- **GWP Comparison**: STRAP vs virgin polymer carbon footprint\n"

    return output


# ============================================================
# Google Scholar Literature Search (SerpAPI Integration)
# ============================================================

@tool
def search_google_scholar(
    query: str,
    max_results: int = 10,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None
) -> str:
    """
    Search Google Scholar for academic research articles using SerpAPI.

    **When to use**: For broad literature searches, preprints, dissertations, and when you need
    maximum coverage across all academic sources. Good for emerging topics or interdisciplinary searches.

    **BETA FEATURE**: Limited to 100 searches/month. Use wisely!

    Args:
        query: Natural language search query (e.g., "polymer dissolution", "Hansen solubility parameters polyethylene")
        max_results: Maximum number of results to return (default: 10, max: 20)
        year_low: Minimum publication year (optional)
        year_high: Maximum publication year (optional)

    Returns:
        Formatted list of research articles with titles, authors, years, and clickable links

    Examples:
        - "Search Google Scholar for recent articles on polymer dissolution"
        - "Find Google Scholar papers on Hansen solubility parameters"
        - "What publications on solvent-based polymer recycling are in Google Scholar?"

    **Note**: Uses simple keyword matching. For peer-reviewed articles with citation metrics,
    consider using Web of Science instead.
    """
    try:
        from serpapi_scholar_client import GoogleScholarClient

        # Initialize client (uses SERPAPI_KEY from environment)
        client = GoogleScholarClient()

        # Perform search
        results = client.search(
            query=query,
            num_results=min(max_results, 20),  # Cap at 20
            year_low=year_low,
            year_high=year_high,
            sort_by='date'  # Get most recent articles
        )

        # Parse results
        organic_results = results.get('organic_results', [])

        if not organic_results:
            return f"No results found for query: '{query}'\n\nTry:\n- Using simpler search terms\n- Removing year filters\n- Checking spelling"

        # Format output
        output = [f"# 📚 Google Scholar Results: {query}\n"]
        output.append(f"**Found:** {len(organic_results)} articles\n")

        if year_low or year_high:
            year_range = f"{year_low or '...'}-{year_high or '...'}"
            output.append(f"**Year Range:** {year_range}\n")

        output.append("\n## Articles\n")

        for i, result in enumerate(organic_results, 1):
            article = client._parse_article(result)

            # Title with link
            title = article.get('title', 'N/A')
            link = article.get('link', '#')
            output.append(f"\n### {i}. [{title}]({link})")

            # Authors - always show
            authors = article.get('authors', [])
            if authors:
                author_str = ', '.join(authors[:5])
                if len(authors) > 5:
                    author_str += f" et al. ({len(authors)} total)"
            else:
                author_str = "Not available"
            output.append(f"**Authors:** {author_str}")

            # Publication info
            publication = article.get('publication_info', '')
            if publication:
                output.append(f"**Publication:** {publication}")

            # Year - always show
            year = article.get('year', 'N/A')
            output.append(f"**Year:** {year}")

            # Citations - always show
            citations = article.get('cited_by_count', 0)
            output.append(f"**Citations:** {citations}")

            # PDF link if available
            pdf_link = article.get('pdf_link')
            if pdf_link:
                output.append(f"📄 **[PDF Available]({pdf_link})**")

            # Snippet
            snippet = article.get('snippet', '')
            if snippet:
                # Truncate long snippets
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                output.append(f"*{snippet}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"**🔍 Search Query:** `{query}`")
        output.append(f"**📊 Results Shown:** {len(organic_results)} of {results.get('search_metadata', {}).get('total_results', 'many')}")
        output.append(f"\n⚠️ **Beta Feature:** This uses SerpAPI with limited monthly searches. Use wisely!")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Google Scholar search is not available. The `serpapi_scholar_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("❌ Google Scholar search requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"❌ Error: {str(e)}"
    except Exception as e:
        logger.error(f"Google Scholar search error: {e}")
        return f"❌ Search failed: {str(e)}\n\nPlease try again or simplify your query."


# ============================================================
# Google Patents Search (SerpAPI)
# ============================================================

@tool
def search_google_patents(
    query: str,
    max_results: int = 10,
    after: Optional[str] = None,
    before: Optional[str] = None,
    assignee: Optional[str] = None,
    inventor: Optional[str] = None,
    country: Optional[str] = None
) -> str:
    """
    Search Google Patents for patent documents using SerpAPI.

    **When to use**: For finding patents related to polymer processing, solvent recovery,
    recycling technologies, and chemical processes. Patents often contain detailed process
    parameters and experimental data not found in academic papers.

    **BETA FEATURE**: Shares the SerpAPI quota with Google Scholar (100 searches/month). Use wisely!

    Args:
        query: Natural language search query or patent-specific terms
               (e.g., "polymer dissolution solvent recovery", "PET recycling process")
        max_results: Maximum number of results to return (default: 10, max: 20)
        after: Filter patents filed after date (format: YYYYMMDD, e.g., "20200101")
        before: Filter patents filed before date (format: YYYYMMDD)
        assignee: Filter by company/assignee (e.g., "Dow", "BASF", "Eastman")
        inventor: Filter by inventor name
        country: Filter by country code (US, EP, WO, CN, JP, etc.)

    Returns:
        Formatted list of patents with IDs, titles, assignees, dates, and clickable links

    Examples:
        - "Search patents for polymer dissolution processes"
        - "Find patents on solvent-based PET recycling"
        - "What patents does Eastman have on polymer recycling?"
        - "Search patents for selective dissolution of mixed plastics" assignee="Dow"

    **Note**: Uses SerpAPI Google Patents API. Returns patent applications and granted patents
    with filing dates, inventors, and assignees.
    """
    try:
        from serpapi_patents_client import GooglePatentsClient

        # Initialize client (uses SERPAPI_KEY from environment)
        client = GooglePatentsClient()

        # Perform search
        results = client.search(
            query=query,
            num_results=min(max_results, 20),
            after=after,
            before=before,
            assignee=assignee,
            inventor=inventor,
            country=country
        )

        # Parse results
        organic_results = results.get('organic_results', [])

        if not organic_results:
            return f"No patents found for query: '{query}'\n\nTry:\n- Using different search terms\n- Removing filters (date, assignee, country)\n- Using more general terminology"

        # Format output
        output = [f"# Patent Search Results: {query}\n"]
        output.append(f"**Found:** {len(organic_results)} patents\n")

        # Show active filters
        filters = []
        if after:
            filters.append(f"After: {after}")
        if before:
            filters.append(f"Before: {before}")
        if assignee:
            filters.append(f"Assignee: {assignee}")
        if inventor:
            filters.append(f"Inventor: {inventor}")
        if country:
            filters.append(f"Country: {country}")
        if filters:
            output.append(f"**Filters:** {', '.join(filters)}\n")

        output.append("\n## Patents\n")

        for i, result in enumerate(organic_results, 1):
            patent = client._parse_patent(result)

            # Patent ID with link
            patent_id = patent.get('patent_id', 'N/A')
            title = patent.get('title', 'N/A')
            link = patent.get('link', f'https://patents.google.com/patent/{patent_id}')
            output.append(f"\n### {i}. [{patent_id}: {title}]({link})")

            # Assignee - always show
            assignee_val = patent.get('assignee', 'N/A')
            output.append(f"**Assignee:** {assignee_val}")

            # Inventors - always show
            inventors = patent.get('inventors', [])
            if inventors:
                inventor_str = ', '.join(inventors[:3])
                if len(inventors) > 3:
                    inventor_str += f" et al. ({len(inventors)} total)"
            else:
                inventor_str = "Not available"
            output.append(f"**Inventors:** {inventor_str}")

            # Dates
            filing_date = patent.get('filing_date', 'N/A')
            grant_date = patent.get('grant_date', 'N/A')
            if filing_date != 'N/A':
                output.append(f"**Filed:** {filing_date}")
            if grant_date != 'N/A':
                output.append(f"**Granted:** {grant_date}")

            # Country
            country_code = patent.get('country', 'Unknown')
            output.append(f"**Country:** {country_code}")

            # PDF link if available
            pdf_link = patent.get('pdf_link')
            if pdf_link:
                output.append(f"[PDF Available]({pdf_link})")

            # Snippet/Abstract
            snippet = patent.get('snippet', '')
            if snippet:
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                output.append(f"*{snippet}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"** Search Query:** `{query}`")
        output.append(f"** Results Shown:** {len(organic_results)}")
        output.append(f"\n **Beta Feature:** This uses SerpAPI with limited monthly searches. Use wisely!")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Google Patents search is not available. The `serpapi_patents_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("❌ Google Patents search requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"❌ Error: {str(e)}"
    except Exception as e:
        logger.error(f"Google Patents search error: {e}")
        return f"❌ Search failed: {str(e)}\n\nPlease try again or simplify your query."


@tool
def lookup_patent(
    patent_number: str
) -> str:
    """
    Look up a specific patent by its number.

    **When to use**: When you have a specific patent number and want to get its details.
    Supports various patent number formats from different patent offices.

    Args:
        patent_number: Patent number in any common format:
                      - US patents: US10123456, US 10,123,456, US2020/0123456
                      - European: EP1234567, EP 1234567 A1
                      - PCT (WIPO): WO2020123456, WO 2020/123456
                      - Chinese: CN112345678
                      - Japanese: JP2020123456
                      - Korean: KR20200123456
                      - Australian: AU2020123456

    Returns:
        Detailed patent information including title, abstract, inventors, assignee,
        filing date, and link to full document

    Examples:
        - "Look up patent US10123456"
        - "Get details for EP1234567"
        - "What is patent WO2020123456 about?"
    """
    try:
        from serpapi_patents_client import GooglePatentsClient

        # Initialize client
        client = GooglePatentsClient()

        # Look up specific patent
        patent = client.get_patent(patent_number)

        if patent.get('error'):
            return f"❌ {patent['error']}\n\nTry checking the patent number format or searching by keywords instead."

        # Format output
        output = [f"# Patent Details: {patent.get('patent_id', patent_number)}\n"]

        # Title
        title = patent.get('title', 'N/A')
        output.append(f"## {title}\n")

        # Link
        link = patent.get('link', '')
        if link:
            output.append(f"**Full Patent:** [{link}]({link})\n")

        # Key Information
        output.append("### Key Information\n")

        # Assignee
        assignee = patent.get('assignee', 'N/A')
        output.append(f"**Assignee/Owner:** {assignee}")

        # Inventors
        inventors = patent.get('inventors', [])
        if inventors:
            output.append(f"**Inventors:** {', '.join(inventors)}")
        else:
            output.append("**Inventors:** Not available")

        # Dates
        output.append("\n### Dates\n")
        filing_date = patent.get('filing_date', 'N/A')
        grant_date = patent.get('grant_date', 'N/A')
        publication_date = patent.get('publication_date', 'N/A')
        priority_date = patent.get('priority_date', 'N/A')

        output.append(f"**Filing Date:** {filing_date}")
        if grant_date != 'N/A':
            output.append(f"**Grant Date:** {grant_date}")
        if publication_date != 'N/A':
            output.append(f"**Publication Date:** {publication_date}")
        if priority_date != 'N/A' and priority_date != filing_date:
            output.append(f"**Priority Date:** {priority_date}")

        # Country
        country = patent.get('country', 'Unknown')
        output.append(f"**Country/Office:** {country}")

        # Claims count if available
        claims_count = patent.get('claims_count')
        if claims_count:
            output.append(f"**Number of Claims:** {claims_count}")

        # Citations
        cited_by = patent.get('cited_by_count', 0)
        if cited_by:
            output.append(f"**Cited By:** {cited_by} patents")

        # Abstract/Snippet
        snippet = patent.get('snippet', '')
        if snippet:
            output.append("\n### Abstract\n")
            output.append(f"*{snippet}*")

        # PDF link
        pdf_link = patent.get('pdf_link')
        if pdf_link:
            output.append(f"\n**[Download PDF]({pdf_link})**")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Patent lookup is not available. The `serpapi_patents_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("❌ Patent lookup requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"❌ Error: {str(e)}"
    except Exception as e:
        logger.error(f"Patent lookup error: {e}")
        return f"❌ Lookup failed: {str(e)}\n\nPlease check the patent number format."


# ============================================================
# Web of Science Literature Search (Starter API)
# ============================================================

@tool
def search_web_of_science(
    query: str,
    polymer_name: Optional[str] = None,
    solvent_name: Optional[str] = None,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    max_results: int = 10
) -> str:
    """
    Search Web of Science for peer-reviewed research articles using Clarivate API.

    **When to use**: For high-quality peer-reviewed articles from journals, with citation counts
    and impact metrics. Best for established research topics in polymer science and chemistry.

    Args:
        query: Search query - can be natural language OR WoS syntax (see below)
        polymer_name: Specific polymer name (e.g., "polyethylene", "PET", "nylon")
        solvent_name: Specific solvent name (e.g., "toluene", "NMP")
        year_low: Minimum publication year (e.g., 2020)
        year_high: Maximum publication year (e.g., 2026)
        max_results: Number of results (default: 10, max: 50)

    **WoS Query Syntax** (for advanced searches):
        - Use quotes for exact phrases: "deinking" AND "plastic"
        - Boolean operators: AND, OR, NOT (must be uppercase)
        - Wildcards: * (e.g., "recycl*" matches recycling, recyclable, recycled)
        - Field tags (optional, auto-added if not present):
            - TS= Topic Search (title, abstract, keywords) - DEFAULT
            - TI= Title only
            - AU= Author
            - SO= Source/Journal name

    **Query Examples**:
        Simple: "PET dissolution"
        Boolean: "deinking" AND "plastic" AND "recycling"
        Phrase + Boolean: "ink removal" AND "polymer" AND "packaging"
        With wildcards: "multilayer*" AND "recycl*" AND "plastic"
        Specific field: TI=("deinking" AND "plastic")

    Returns:
        Formatted list of peer-reviewed articles with titles, authors, journals, years, DOIs

    **Note**: Uses Web of Science Starter API. Returns peer-reviewed journal articles with
    citation metrics and DOI links. More focused on high-quality sources than Google Scholar.
    """
    try:
        from wos_starter_client import WebOfScienceStarterClient

        # Initialize client (uses WOS_STARTER_API_KEY from environment)
        client = WebOfScienceStarterClient()

        # Determine search strategy
        if polymer_name or solvent_name:
            # Use specialized polymer search
            articles = client.search_polymer_articles(
                polymer_name=polymer_name,
                solvent_name=solvent_name,
                year_low=year_low,
                year_high=year_high,
                max_results=max_results
            )
        elif "hansen" in query.lower() and "TS=" not in query.upper():
            # Hansen-specific search (only if not already using WoS syntax)
            articles = client.search_hansen_parameters(
                year_low=year_low,
                year_high=year_high,
                max_results=max_results
            )
        else:
            # Build WoS query - check if user provided WoS syntax
            wos_query = query.strip()

            # Check if query already has WoS field tags
            has_field_tag = any(tag in wos_query.upper() for tag in ['TS=', 'TI=', 'AU=', 'SO=', 'PY='])

            if not has_field_tag:
                # Convert natural language to WoS syntax
                # Check for boolean operators already in query
                has_boolean = any(op in wos_query.upper() for op in [' AND ', ' OR ', ' NOT '])

                if has_boolean:
                    # Query has boolean operators - wrap in TS=()
                    # Handle quoted phrases properly
                    wos_query = f'TS=({wos_query})'
                else:
                    # Simple query - wrap as topic search
                    # If multiple words without quotes, treat as AND search
                    words = wos_query.split()
                    if len(words) > 1 and '"' not in wos_query:
                        # Multi-word query without quotes - join with AND
                        wos_query = f'TS=({" AND ".join(words)})'
                    else:
                        wos_query = f'TS=({wos_query})'

            # Add year range if specified and not already in query
            if (year_low or year_high) and 'PY=' not in wos_query.upper():
                year_start = year_low or 1900
                year_end = year_high or 2030
                wos_query += f' AND PY=({year_start}-{year_end})'

            logger.info(f"WoS query: {wos_query}")

            results = client.search_documents(
                query=wos_query,
                limit=max_results,
                sort_field='PY+D'
            )

            # Parse results
            articles = []
            for record in results.get('hits', []):
                articles.append(client._parse_article(record))

        if not articles:
            return f"No Web of Science articles found for: '{query}'\n\n**Suggestions:**\n- Try broader search terms\n- Remove year restrictions\n- Check spelling of polymer/solvent names"

        # Format output
        output = [f"# 📚 Web of Science Results: {query}\n"]
        output.append(f"**Found:** {len(articles)} peer-reviewed articles\n")

        if year_low or year_high:
            year_range = f"{year_low or '...'}-{year_high or '...'}"
            output.append(f"**Year Range:** {year_range}\n")

        output.append("\n## Articles\n")

        for i, article in enumerate(articles, 1):
            # Title with WoS link
            title = article.get('title', 'N/A')
            link = article.get('link', '#')
            output.append(f"\n### {i}. [{title}]({link})")

            # Authors - always show, even if empty
            authors = article.get('authors', [])
            if authors:
                author_str = ', '.join(authors[:5])
                if len(authors) > 5:
                    author_str += f" et al. ({len(authors)} total)"
            else:
                author_str = "Not available"
            output.append(f"**Authors:** {author_str}")

            # Journal and year - always show
            journal = article.get('journal', 'N/A')
            year = article.get('year', 'N/A')
            output.append(f"**Journal:** {journal}")
            output.append(f"**Year:** {year}")

            # Volume and pages if available
            volume = article.get('volume', '')
            pages = article.get('pages', '')
            if volume or pages:
                vol_page = []
                if volume:
                    vol_page.append(f"Vol. {volume}")
                if pages:
                    vol_page.append(f"pp. {pages}")
                output.append(f"**Volume/Pages:** {', '.join(vol_page)}")

            # DOI - always show
            doi = article.get('doi', 'N/A')
            if doi and doi != 'N/A':
                output.append(f"**DOI:** [{doi}](https://doi.org/{doi})")
            else:
                output.append(f"**DOI:** Not available")

            # Citations - always show
            times_cited = article.get('times_cited', 0)
            output.append(f"**Times Cited:** {times_cited}")

            # Abstract snippet
            abstract = article.get('abstract', '')
            if abstract and abstract != 'N/A':
                # Truncate long abstracts
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                output.append(f"*{abstract}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"**🔍 Search Query:** `{query}`")
        output.append(f"**📊 Results:** {len(articles)} peer-reviewed articles from Web of Science")
        output.append(f"**🏛️ Source:** Clarivate Web of Science Starter API")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Web of Science search is not available. The `wos_starter_client` module is not installed.\n\n"
                "Please ensure the WoS client is properly configured.")
    except ValueError as e:
        if "WOS_STARTER_API_KEY" in str(e):
            return ("❌ Web of Science search requires an API key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://developer.clarivate.com/\n"
                    "2. Set environment variable: `WOS_STARTER_API_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"❌ Error: {str(e)}"
    except Exception as e:
        logger.error(f"Web of Science search error: {e}")
        return f"❌ Search failed: {str(e)}\n\nPlease try again or simplify your query."


# ============================================================
# RAG (Retrieval-Augmented Generation) Tools
# Literature Search with Vector Retrieval
# ============================================================

@tool
@safe_tool_wrapper
async def search_literature_rag(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
    include_context: bool = True
) -> str:
    """
    Search indexed scientific literature using RAG (Retrieval-Augmented Generation).

    This tool searches through your indexed PDF documents (scientific papers, patents,
    technical reports) to find relevant passages. It uses hybrid search combining:
    - Semantic search (understands meaning)
    - Keyword search (exact term matching)
    - Cross-encoder reranking (improves precision)

    **When to use**:
    - Find specific information in indexed research papers
    - Answer questions based on uploaded scientific literature
    - Get context from multiple sources about a topic
    - Support answers with citations from literature

    Args:
        query: Natural language question or search query
               (e.g., "What solvents are effective for polystyrene dissolution?")
        top_k: Number of relevant passages to return (default: 5)
        source_filter: Optional comma-separated list of document names to search within
                      (e.g., "paper1,paper2")
        include_context: Whether to include expanded parent context (default: True)

    Returns:
        Relevant passages from indexed literature with sources and page numbers

    Examples:
        - "Search literature for Hansen solubility parameters of polyethylene"
        - "What does the literature say about toluene toxicity?"
        - "Find passages about multilayer film separation"
    """
    try:
        # Get RAG system
        rag_system = rag.get_rag_system()

        if not rag_system.is_ready():
            return ("⚠️ **RAG System Not Ready**\n\n"
                    "No documents have been indexed yet.\n\n"
                    "**To index documents:**\n"
                    "1. Add PDF files to the `./rag_pdfs/` directory\n"
                    "2. Use the `ingest_pdf_to_rag` tool to index them\n\n"
                    "Alternatively, use `search_google_scholar` or `search_web_of_science` "
                    "for online literature search.")

        # Parse source filter
        sources = None
        if source_filter:
            sources = [s.strip() for s in source_filter.split(",")]

        # Perform search
        results = rag_system.search(
            query=query,
            top_k=top_k,
            source_filter=sources,
            return_parent_context=include_context
        )

        if not results:
            return (f"No relevant passages found for: '{query}'\n\n"
                    "**Suggestions:**\n"
                    "- Try different search terms\n"
                    "- Use more specific or broader query\n"
                    "- Check if relevant documents are indexed with `get_rag_status`")

        # Format output
        output = [f"# 📚 Literature Search Results\n"]
        output.append(f"**Query:** {query}")
        output.append(f"**Found:** {len(results)} relevant passages\n")

        for i, result in enumerate(results, 1):
            output.append(f"\n---\n")
            output.append(f"### Passage {i}")

            # Source info
            source_info = f"**Source:** {result.source}"
            if result.page_number:
                source_info += f" (Page {result.page_number})"
            output.append(source_info)

            # Relevance score
            score = result.rerank_score if result.rerank_score is not None else result.score
            output.append(f"**Relevance:** {score:.3f}")

            # Text content
            text = result.parent_text if include_context and result.parent_text else result.text
            # Truncate very long passages
            if len(text) > 1500:
                text = text[:1500] + "..."
            output.append(f"\n{text}\n")

        # Add summary
        sources_used = list(set(r.source for r in results))
        output.append(f"\n---\n**Sources searched:** {', '.join(sources_used)}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"❌ Literature search failed: {str(e)}\n\nPlease try again with a simpler query."


@tool
@safe_tool_wrapper
async def ingest_pdf_to_rag(
    pdf_paths: Optional[str] = None,
    use_ocr: bool = False,
    recreate_index: bool = False
) -> str:
    """
    Ingest PDF documents into the RAG system for literature search.

    This tool processes PDF files and indexes them for semantic search.
    The process includes:
    - Text extraction (with optional OCR for scanned documents)
    - Smart chunking that preserves context
    - Filtering of low-quality content (headers, citations)
    - Embedding generation for semantic search

    **When to use**:
    - Add new research papers to the searchable index
    - Re-index documents after changes
    - Build a knowledge base from scientific literature

    Args:
        pdf_paths: Comma-separated paths to PDF files, or None to scan ./rag_pdfs/ directory
                  (e.g., "./papers/paper1.pdf,./papers/paper2.pdf")
        use_ocr: Enable OCR for scanned documents (slower but handles images)
        recreate_index: Delete existing index and start fresh (default: False)

    Returns:
        Summary of ingestion results including number of documents and chunks processed

    Examples:
        - "Ingest all PDFs in the rag_pdfs folder"
        - "Add paper.pdf to the RAG index"
        - "Re-index all documents with OCR enabled"
    """
    try:
        import glob as glob_module

        # Determine PDF paths
        if pdf_paths:
            paths = [p.strip() for p in pdf_paths.split(",")]
        else:
            # Scan default directory
            paths = glob_module.glob(os.path.join(rag.RAG_PDF_DIR, "*.pdf"))

        if not paths:
            return (f"❌ **No PDFs Found**\n\n"
                    f"No PDF files found in `{rag.RAG_PDF_DIR}/`\n\n"
                    "**To add documents:**\n"
                    "1. Place PDF files in the `./rag_pdfs/` directory\n"
                    "2. Or specify paths: `ingest_pdf_to_rag(pdf_paths='path/to/file.pdf')`")

        # Perform ingestion
        rag_system = rag.get_rag_system()
        result = rag_system.ingest_pdfs(
            pdf_paths=paths,
            use_ocr=use_ocr,
            recreate_collection=recreate_index
        )

        if not result.get("success"):
            return f"❌ **Ingestion Failed**\n\n{result.get('error', 'Unknown error')}"

        # Format output
        output = ["# 📥 PDF Ingestion Complete\n"]
        output.append(f"**Documents Processed:** {len(result.get('processed_files', []))}")
        output.append(f"**Documents Failed:** {len(result.get('failed_files', []))}")
        output.append(f"\n**Indexing Summary:**")
        output.append(f"- Base chunks: {result.get('base_chunks', 0)}")
        output.append(f"- Parent chunks: {result.get('parent_chunks', 0)}")
        output.append(f"- Child chunks (indexed): {result.get('child_chunks', 0)}")

        # Filter stats
        filter_stats = result.get('filter_stats', {})
        if filter_stats:
            output.append(f"\n**Filtering:**")
            output.append(f"- Processed: {filter_stats.get('total_processed', 0)}")
            output.append(f"- Retained: {filter_stats.get('retained', 0)}")
            output.append(f"- Filtered: headers={filter_stats.get('header_footer', 0)}, "
                         f"citations={filter_stats.get('citation_heavy', 0)}, "
                         f"duplicates={filter_stats.get('duplicate', 0)}")

        # List processed files
        processed = result.get('processed_files', [])
        if processed:
            output.append(f"\n**Processed Files:**")
            for p in processed[:10]:  # Limit to 10
                output.append(f"- {Path(p).name}")
            if len(processed) > 10:
                output.append(f"- ... and {len(processed) - 10} more")

        # Failed files
        failed = result.get('failed_files', [])
        if failed:
            output.append(f"\n**Failed Files:**")
            for f in failed[:5]:
                output.append(f"- {Path(f).name}")

        output.append(f"\n✅ RAG system ready for search. Use `search_literature_rag` to query.")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"RAG ingestion error: {e}")
        return f"❌ Ingestion failed: {str(e)}"


@tool
@safe_tool_wrapper
async def get_rag_status() -> str:
    """
    Get the current status of the RAG (Retrieval-Augmented Generation) system.

    This tool shows:
    - Whether the system is initialized and ready
    - Number of indexed documents and chunks
    - Available sources (document names)
    - System configuration

    **When to use**:
    - Check if documents are indexed before searching
    - See what documents are available
    - Verify system health
    - Debug RAG issues

    Returns:
        Detailed status of the RAG system including indexed documents and configuration
    """
    try:
        status = rag.get_rag_status()

        output = ["# 📊 RAG System Status\n"]

        # Overall status
        if status.get('ready'):
            output.append("**Status:** ✅ Ready for search")
        elif status.get('initialized'):
            output.append("**Status:** ⚠️ Initialized but no documents indexed")
        else:
            output.append("**Status:** ❌ Not initialized")

        # Dependencies
        output.append(f"\n**Dependencies:**")
        output.append(f"- Embeddings: {'✅' if status.get('embeddings_available') else '❌'}")
        output.append(f"- Vector DB (Qdrant): {'✅' if status.get('qdrant_available') else '❌'}")
        output.append(f"- PDF Processing: {'✅' if status.get('pdf_processing_available') else '❌'}")
        output.append(f"- Reranking: {'✅' if status.get('reranking_enabled') else '❌'}")

        # Collection info
        collection = status.get('collection', {})
        output.append(f"\n**Vector Database:**")
        output.append(f"- Collection: {collection.get('collection_name', 'N/A')}")
        output.append(f"- Indexed chunks: {collection.get('points_count', 0)}")
        output.append(f"- Status: {collection.get('status', 'unknown')}")

        # Chunk store
        chunk_store = status.get('chunk_store', {})
        output.append(f"\n**Document Statistics:**")
        output.append(f"- Total chunks: {chunk_store.get('total_chunks', 0)}")
        output.append(f"- Total sources: {chunk_store.get('total_sources', 0)}")
        output.append(f"- Parent chunks: {chunk_store.get('parent_chunks', 0)}")
        output.append(f"- Child chunks: {chunk_store.get('child_chunks', 0)}")
        output.append(f"- Parent-child mode: {'Yes' if chunk_store.get('parent_child_enabled') else 'No'}")

        # Sources
        sources = chunk_store.get('sources', [])
        if sources:
            output.append(f"\n**Indexed Documents ({len(sources)}):**")
            chunks_per_source = chunk_store.get('chunks_per_source', {})
            for source in sources[:15]:  # Limit to 15
                count = chunks_per_source.get(source, 'N/A')
                output.append(f"- {source}: {count} chunks")
            if len(sources) > 15:
                output.append(f"- ... and {len(sources) - 15} more")
        else:
            output.append(f"\n**No documents indexed yet.**")
            output.append(f"\nTo add documents:")
            output.append(f"1. Place PDF files in `./rag_pdfs/`")
            output.append(f"2. Run `ingest_pdf_to_rag` tool")

        # Configuration
        config = status.get('config', {})
        output.append(f"\n**Configuration:**")
        output.append(f"- Embedding model: {config.get('dense_model', 'N/A')}")
        output.append(f"- Chunk strategy: {config.get('chunk_strategy', 'N/A')}")
        output.append(f"- Chunk size: {config.get('chunk_size', 'N/A')} tokens")
        output.append(f"- Parent-child: {'Yes' if config.get('use_parent_child') else 'No'}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"RAG status error: {e}")
        return f"❌ Failed to get RAG status: {str(e)}"


@tool
@safe_tool_wrapper
async def ask_literature(
    question: str,
    top_k: int = 5,
    max_context_tokens: int = 3000
) -> str:
    """
    Ask a question and get an answer synthesized from indexed scientific literature.

    This tool:
    1. Searches the indexed literature for relevant passages
    2. Formats the context for answering
    3. Returns the relevant information with citations

    **When to use**:
    - Get answers grounded in scientific literature
    - Find supporting evidence from papers
    - Compare information across multiple sources

    Args:
        question: Natural language question to answer
                 (e.g., "What is the optimal temperature for dissolving polystyrene?")
        top_k: Number of passages to consider (default: 5)
        max_context_tokens: Maximum tokens for context (default: 3000)

    Returns:
        Relevant information from literature with source citations

    Examples:
        - "What are the environmental impacts of toluene?"
        - "How does temperature affect polymer solubility?"
        - "What solvents are recommended for PET recycling?"
    """
    try:
        rag_system = rag.get_rag_system()

        if not rag_system.is_ready():
            return ("⚠️ **Literature Database Not Ready**\n\n"
                    "No documents have been indexed.\n\n"
                    "Use `ingest_pdf_to_rag` to add scientific papers first.")

        # Get context and sources
        context, sources = rag.format_rag_context(
            query=question,
            top_k=top_k,
            max_tokens=max_context_tokens
        )

        if not context:
            return (f"No relevant information found for: '{question}'\n\n"
                    "Try rephrasing your question or check indexed documents with `get_rag_status`.")

        # Format response
        output = [f"# 📖 Literature Answer\n"]
        output.append(f"**Question:** {question}\n")
        output.append(f"---\n")
        output.append(f"## Relevant Information from Literature\n")
        output.append(context)
        output.append(f"\n---\n")
        output.append(f"## Sources ({len(sources)} passages)")
        for i, src in enumerate(sources, 1):
            source_name = src.get('source', 'Unknown')
            page = src.get('page_number')
            score = src.get('score', 0)
            source_str = f"{i}. **{source_name}**"
            if page:
                source_str += f" (Page {page})"
            source_str += f" - Relevance: {score:.2f}"
            output.append(source_str)

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Ask literature error: {e}")
        return f"❌ Failed to search literature: {str(e)}"


@tool
@safe_tool_wrapper
async def clear_rag_index() -> str:
    """
    Clear the RAG index and remove all indexed documents.

    This tool deletes all indexed documents from the vector database,
    allowing you to start fresh with new documents.

    **Warning**: This action cannot be undone!

    **When to use**:
    - Reset the index to remove all documents
    - Fix corrupted index
    - Start over with new document set

    Returns:
        Confirmation of index deletion
    """
    try:
        rag_system = rag.get_rag_system()

        # Get current status
        status = rag_system.get_status()
        collection = status.get('collection', {})
        points_before = collection.get('points_count', 0)

        # Clear the index
        if rag_system.vector_db:
            success = rag_system.vector_db.delete_collection()
            if not success:
                return "❌ Failed to delete collection"

        # Clear chunk store
        rag_system.chunk_store.clear()

        output = ["# 🗑️ RAG Index Cleared\n"]
        output.append(f"**Deleted:** {points_before} indexed chunks")
        output.append(f"\nThe RAG system is now empty.")
        output.append(f"\nTo re-index documents:")
        output.append(f"1. Ensure PDFs are in `./rag_pdfs/`")
        output.append(f"2. Run `ingest_pdf_to_rag`")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Clear RAG index error: {e}")
        return f"❌ Failed to clear index: {str(e)}"


@tool
@safe_tool_wrapper
async def visualize_rag_chunks() -> str:
    """
    Generate visualization plots showing the distribution of indexed chunks.

    This tool creates a comprehensive 6-panel visualization showing:
    - Token count distribution (histogram)
    - Character count distribution (histogram)
    - Chunks per document (bar chart)
    - Section type distribution (pie chart)
    - Token vs Character correlation (scatter plot)
    - Token distribution by section type (box plot)

    **When to use**:
    - After ingesting new documents to verify quality
    - To understand the composition of your RAG index
    - To diagnose chunking issues
    - To optimize chunking parameters

    Returns:
        Path to the saved visualization plot and summary statistics

    Examples:
        - "Show me visualization of the indexed chunks"
        - "Generate chunk distribution plots"
        - "Visualize the RAG document breakdown"
    """
    try:
        # Generate plot
        plot_path = rag.plot_chunk_distributions()

        if plot_path is None:
            return ("⚠️ **No chunks to visualize**\n\n"
                    "The RAG index is empty. Use `ingest_pdf_to_rag` to add documents first.")

        # Get summary stats
        summary = rag.get_chunk_summary()

        output = ["# 📊 RAG Chunk Visualization\n"]
        output.append(f"**Plot saved to:** `{plot_path}`\n")

        # Quick stats
        output.append("## Summary Statistics\n")
        output.append(f"- **Total Chunks:** {summary['total_chunks']:,}")
        output.append(f"- **Documents:** {summary['total_documents']}")
        output.append(f"- **Total Tokens:** {summary['token_stats']['total']:,}")

        ts = summary['token_stats']
        output.append(f"\n**Token Distribution:**")
        output.append(f"- Mean: {ts['mean']:.1f} | Median: {ts['median']:.1f}")
        output.append(f"- Range: {ts['min']} - {ts['max']}")
        output.append(f"- Std Dev: {ts['std']:.1f}")

        # Section breakdown
        output.append(f"\n**Section Types:**")
        for section, count in sorted(summary['section_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)[:5]:
            pct = 100 * count / summary['total_chunks']
            output.append(f"- {section.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

        output.append(f"\n**View the full visualization at:** `{plot_path}`")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Chunk visualization error: {e}")
        return f"❌ Failed to generate visualization: {str(e)}"


@tool
@safe_tool_wrapper
async def check_rag_chunk_quality() -> str:
    """
    Run quality checks on ingested RAG chunks and identify potential issues.

    This tool analyzes the indexed chunks and checks for:
    - Minimum chunk count (should have enough for good coverage)
    - Tiny chunks (<20 tokens) that may lack context
    - High variance in chunk sizes (inconsistent chunking)
    - Empty or near-empty chunks
    - Imbalanced document distribution
    - Missing important sections (abstract, methods, results)

    **When to use**:
    - After ingesting documents to verify quality
    - Before running searches if getting poor results
    - To diagnose RAG performance issues
    - To decide if re-chunking is needed

    Returns:
        Quality assessment with issues, warnings, and recommendations

    Examples:
        - "Check the quality of my RAG chunks"
        - "Are my indexed documents properly chunked?"
        - "Run quality diagnostics on the RAG index"
    """
    try:
        quality = rag.check_chunk_quality()

        if quality['status'] == 'error':
            return (f"⚠️ **{quality['message']}**\n\n"
                    f"**Recommendations:**\n" +
                    "\n".join(f"- {r}" for r in quality['recommendations']))

        output = ["# 🔍 RAG Chunk Quality Report\n"]

        # Status indicator
        if quality['status'] == 'healthy':
            output.append("**Status:** ✅ All checks passed\n")
        elif quality['status'] == 'warnings':
            output.append("**Status:** ⚠️ Warnings detected\n")
        else:
            output.append("**Status:** ❌ Issues found\n")

        output.append(f"**Total Chunks:** {quality['total_chunks']:,}\n")

        # Issues
        if quality['issues']:
            output.append("## ❌ Issues\n")
            for issue in quality['issues']:
                output.append(f"- {issue}")
            output.append("")

        # Warnings
        if quality['warnings']:
            output.append("## ⚠️ Warnings\n")
            for warning in quality['warnings']:
                output.append(f"- {warning}")
            output.append("")

        # Recommendations
        if quality['recommendations']:
            output.append("## 💡 Recommendations\n")
            for rec in quality['recommendations']:
                output.append(f"- {rec}")
            output.append("")

        # Quality metrics
        summary = quality.get('summary', {})
        if summary:
            qm = summary.get('quality_metrics', {})
            output.append("## 📊 Quality Metrics\n")
            output.append(f"- **Tiny chunks (<20 tokens):** {qm.get('tiny_chunks', 0)} ({qm.get('tiny_percentage', 0):.1f}%)")
            output.append(f"- **Large chunks (>1000 tokens):** {qm.get('large_chunks', 0)} ({qm.get('large_percentage', 0):.1f}%)")
            output.append(f"- **Empty chunks:** {qm.get('empty_chunks', 0)}")
            output.append(f"- **Coefficient of Variation:** {qm.get('cv', 0):.2f}")

            if qm.get('cv', 0) < 0.5:
                output.append("  (Good - consistent chunk sizes)")
            elif qm.get('cv', 0) < 1.0:
                output.append("  (Moderate - some variation)")
            else:
                output.append("  (High - inconsistent chunking)")

        if quality['status'] == 'healthy':
            output.append("\n✅ Your RAG index is healthy and ready for searching!")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Chunk quality check error: {e}")
        return f"❌ Failed to check quality: {str(e)}"


@tool
@safe_tool_wrapper
async def get_rag_chunk_report() -> str:
    """
    Generate a comprehensive report of all RAG chunk statistics.

    This tool creates a detailed markdown report including:
    - Overview (total chunks, documents, tokens)
    - Token statistics (mean, median, min, max, std)
    - Section distribution (abstract, methods, results, etc.)
    - Document distribution (chunks per PDF)
    - Quality assessment (issues, warnings, recommendations)
    - Quality metrics (tiny/large/empty chunks)

    **When to use**:
    - Get a complete overview of your RAG index
    - Document the state of your literature collection
    - Compare before/after re-indexing
    - Share statistics with others

    Returns:
        Comprehensive markdown report with all statistics

    Examples:
        - "Give me a full report on my RAG chunks"
        - "Generate detailed RAG index statistics"
        - "What's the complete breakdown of my indexed literature?"
    """
    try:
        report = rag.generate_chunk_report()
        return report

    except Exception as e:
        logger.error(f"Chunk report error: {e}")
        return f"❌ Failed to generate report: {str(e)}"


# ============================================================
# RAG Advanced Diagnostics Tools
# ============================================================

@tool
@safe_tool_wrapper
async def analyze_search_diagnostics(
    query: str,
    top_k: int = 10
) -> str:
    """
    Analyze search score breakdown for a specific query.

    Shows contribution of dense, sparse, section boost, and reranking scores
    to help understand why certain results rank higher.

    **When to use**:
    - Debug why certain results appear higher/lower than expected
    - Understand the contribution of different scoring components
    - Tune search parameters
    - Verify hybrid search is working correctly

    Args:
        query: Search query to analyze
        top_k: Number of results to analyze (default: 10)

    Returns:
        Score breakdown visualization and statistics

    Examples:
        - "Analyze search scores for 'polymer dissolution temperature'"
        - "Why are these results ranking this way for 'Hansen parameters'?"
        - "Debug search scores for 'PET recycling'"
    """
    try:
        analysis = rag.analyze_search_scores(query=query, top_k=top_k)

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = [f"# 🔍 Search Score Analysis\n"]
        output.append(f"**Query:** {query}")
        output.append(f"**Results Analyzed:** {analysis['num_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Score statistics
        stats = analysis.get("score_stats", {})
        output.append("## Score Statistics\n")
        output.append(f"- **Avg Dense Score:** {stats.get('dense_mean', 0):.3f}")
        output.append(f"- **Avg Sparse Score:** {stats.get('sparse_mean', 0):.3f}")
        output.append(f"- **Reranking Improved:** {stats.get('rerank_improved', 0)} results\n")

        # Top results breakdown
        output.append("## Top Results Breakdown\n")
        output.append("| Rank | Source | Section | Dense | Sparse | Boost | Final |")
        output.append("|------|--------|---------|-------|--------|-------|-------|")

        for i, r in enumerate(analysis.get("results", [])[:5], 1):
            output.append(f"| {i} | {r['source'][:15]}... | {r['section'][:10]} | "
                         f"{r['dense_score']:.3f} | {r['sparse_score']:.3f} | "
                         f"{r['section_boost']:.3f} | {r['final_score']:.3f} |")

        output.append("\n**Interpretation:**")
        if stats.get('dense_mean', 0) > stats.get('sparse_mean', 0):
            output.append("- Dense (semantic) search is contributing more than sparse (keyword)")
        else:
            output.append("- Sparse (keyword) search is contributing more than dense (semantic)")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Search diagnostics error: {e}")
        return f"❌ Failed to analyze search: {str(e)}"


@tool
@safe_tool_wrapper
async def visualize_retrieval_patterns() -> str:
    """
    Analyze retrieval patterns across multiple test queries.

    Shows which documents and sections are retrieved most frequently,
    helping identify any biases or gaps in retrieval.

    **When to use**:
    - Understand which documents dominate search results
    - Check if certain sections are over/under-represented
    - Identify potential retrieval biases
    - Verify balanced coverage across your document collection

    Returns:
        Retrieval pattern analysis with visualization

    Examples:
        - "Show me retrieval patterns across my documents"
        - "Which documents are retrieved most often?"
        - "Analyze section distribution in search results"
    """
    try:
        analysis = rag.analyze_retrieval_patterns()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📊 Retrieval Pattern Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Test Queries:** {', '.join(analysis.get('queries_tested', [])[:3])}...\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Most retrieved documents
        output.append("## Most Retrieved Documents\n")
        for doc, count in analysis.get("most_retrieved_docs", []):
            output.append(f"- **{doc[:40]}...**: {count} times")

        # Section distribution
        output.append("\n## Section Distribution\n")
        section_dist = analysis.get("section_distribution", {})
        total = sum(section_dist.values())
        for section, count in sorted(section_dist.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total if total > 0 else 0
            output.append(f"- **{section.replace('_', ' ').title()}**: {count} ({pct:.1f}%)")

        # Top score average
        output.append(f"\n**Avg Top-1 Score:** {analysis.get('avg_top_score', 0):.3f}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Retrieval patterns error: {e}")
        return f"❌ Failed to analyze retrieval patterns: {str(e)}"


@tool
@safe_tool_wrapper
async def visualize_embedding_space(
    sample_size: int = 500,
    method: str = "tsne"
) -> str:
    """
    Visualize document embeddings in 2D space using dimensionality reduction.

    Creates a scatter plot showing how documents cluster in embedding space,
    colored by source document and section type.

    **When to use**:
    - See if documents cluster by topic
    - Identify outlier documents
    - Verify embedding quality
    - Understand semantic relationships between documents

    Args:
        sample_size: Number of chunks to sample (default: 500)
        method: Dimensionality reduction method - "tsne" or "umap" (default: tsne)

    Returns:
        Embedding space visualization and clustering analysis

    Examples:
        - "Visualize my document embeddings"
        - "Show embedding space clustering"
        - "Create t-SNE plot of my RAG documents"
    """
    try:
        analysis = rag.visualize_embedding_space(sample_size=sample_size, method=method)

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = [f"# 🗺️ Embedding Space Visualization\n"]
        output.append(f"**Method:** {analysis['method'].upper()}")
        output.append(f"**Embeddings Visualized:** {analysis['num_embeddings']}")
        output.append(f"**Embedding Dimension:** {analysis['embedding_dim']}")
        output.append(f"**Unique Documents:** {analysis['unique_sources']}")
        output.append(f"**Unique Sections:** {analysis['unique_sections']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        output.append("**Interpretation:**")
        output.append("- Clusters indicate semantically similar content")
        output.append("- Documents should cluster if they cover similar topics")
        output.append("- Scattered points may indicate diverse content")
        output.append("- Outliers may be unique or poorly extracted content")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Embedding visualization error: {e}")
        return f"❌ Failed to visualize embeddings: {str(e)}"


@tool
@safe_tool_wrapper
async def analyze_document_similarity() -> str:
    """
    Compute and visualize document-level similarity matrix.

    Shows which documents in your collection are semantically similar
    to each other, helping identify related papers and potential duplicates.

    **When to use**:
    - Find related documents in your collection
    - Identify potential duplicate content
    - Understand document relationships
    - Verify diverse coverage

    Returns:
        Document similarity matrix with most/least similar pairs

    Examples:
        - "Which documents in my RAG are most similar?"
        - "Show document similarity matrix"
        - "Find related papers in my collection"
    """
    try:
        analysis = rag.compute_document_similarity_matrix()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📐 Document Similarity Analysis\n"]
        output.append(f"**Documents Analyzed:** {analysis['num_documents']}")
        output.append(f"**Average Similarity:** {analysis['avg_similarity']:.3f}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Most similar pairs
        output.append("## Most Similar Document Pairs\n")
        for pair in analysis.get("most_similar_pairs", [])[:5]:
            output.append(f"- **{pair['doc1'][:25]}...** ↔ **{pair['doc2'][:25]}...**: {pair['similarity']:.3f}")

        # Least similar pairs
        output.append("\n## Least Similar Document Pairs\n")
        for pair in analysis.get("least_similar_pairs", [])[:3]:
            output.append(f"- **{pair['doc1'][:25]}...** ↔ **{pair['doc2'][:25]}...**: {pair['similarity']:.3f}")

        output.append("\n**Interpretation:**")
        output.append("- Similarity > 0.8: Very related content, possible overlap")
        output.append("- Similarity 0.5-0.8: Related topics")
        output.append("- Similarity < 0.5: Different topics (good diversity)")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Document similarity error: {e}")
        return f"❌ Failed to compute similarity: {str(e)}"


@tool
@safe_tool_wrapper
async def analyze_dense_vs_sparse() -> str:
    """
    Compare dense (semantic) vs sparse (keyword) retrieval performance.

    Shows when each method works better and their correlation,
    helping understand hybrid search behavior.

    **When to use**:
    - Understand which retrieval method is contributing more
    - Tune hybrid search weights
    - Debug when semantic or keyword search fails
    - Verify hybrid search is combining both signals

    Returns:
        Dense vs sparse comparison with visualization

    Examples:
        - "Compare dense and sparse retrieval"
        - "Is semantic search or keyword search working better?"
        - "Analyze hybrid search components"
    """
    try:
        analysis = rag.analyze_dense_vs_sparse()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# ⚖️ Dense vs Sparse Retrieval Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Results Analyzed:** {analysis['num_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Key statistics
        output.append("## Key Statistics\n")
        output.append(f"- **Correlation:** {analysis['correlation']:.3f}")
        output.append(f"- **Avg Dense Score:** {analysis['avg_dense_score']:.3f}")
        output.append(f"- **Avg Sparse Score:** {analysis['avg_sparse_score']:.3f}")
        output.append(f"- **Dense Wins:** {analysis['dense_wins']} ({analysis['dense_win_rate']*100:.1f}%)")
        output.append(f"- **Sparse Wins:** {analysis['sparse_wins']} ({(1-analysis['dense_win_rate'])*100:.1f}%)")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['correlation'] > 0.7:
            output.append("- High correlation: Both methods agree on relevance")
        elif analysis['correlation'] > 0.4:
            output.append("- Moderate correlation: Methods complement each other")
        else:
            output.append("- Low correlation: Methods capture different signals (good for hybrid)")

        if analysis['dense_win_rate'] > 0.7:
            output.append("- Semantic search is dominant - consider adjusting sparse weight")
        elif analysis['dense_win_rate'] < 0.3:
            output.append("- Keyword search is dominant - documents may have strong keywords")
        else:
            output.append("- Balanced contribution - hybrid search working well")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Dense vs sparse error: {e}")
        return f"❌ Failed to analyze: {str(e)}"


@tool
@safe_tool_wrapper
async def analyze_reranking_impact() -> str:
    """
    Analyze how cross-encoder reranking changes result ordering.

    Shows position changes, score improvements, and which results
    benefit most from reranking.

    **When to use**:
    - Verify reranking is improving results
    - Understand which content benefits from reranking
    - Debug when good results are ranked low
    - Decide if reranking is worth the latency cost

    Returns:
        Reranking impact analysis with visualization

    Examples:
        - "How much does reranking help my search results?"
        - "Analyze reranking impact"
        - "Is the cross-encoder improving ranking?"
    """
    try:
        analysis = rag.analyze_reranking_impact()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 🔄 Reranking Impact Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Results Analyzed:** {analysis['total_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Position changes
        output.append("## Position Changes\n")
        output.append(f"- **Results with position change:** {analysis['results_with_position_change']}")
        output.append(f"- **Moved up:** {analysis['moved_up']}")
        output.append(f"- **Moved down:** {analysis['moved_down']}")
        output.append(f"- **Unchanged:** {analysis['unchanged']}")
        output.append(f"- **Avg position change:** {analysis['avg_position_change']:.2f}")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['moved_up'] > analysis['moved_down']:
            output.append("- Reranking is promoting relevant results ✅")
        elif analysis['moved_up'] < analysis['moved_down']:
            output.append("- Reranking may be demoting good results ⚠️")
        else:
            output.append("- Reranking has balanced effect")

        pct_changed = analysis['results_with_position_change'] / analysis['total_results'] * 100 if analysis['total_results'] > 0 else 0
        if pct_changed > 50:
            output.append(f"- High impact: {pct_changed:.0f}% of results changed position")
        else:
            output.append(f"- Moderate impact: {pct_changed:.0f}% of results changed position")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Reranking analysis error: {e}")
        return f"❌ Failed to analyze reranking: {str(e)}"


@tool
@safe_tool_wrapper
async def analyze_section_boost() -> str:
    """
    Analyze the impact of section-based score boosting.

    Shows how different section types (abstract, methods, results) are
    boosted and their effect on final ranking.

    **When to use**:
    - Understand section boost configuration
    - Verify abstracts are getting appropriate priority
    - Debug when methods/results sections rank unexpectedly
    - Tune section boost parameters

    Returns:
        Section boost analysis with visualization

    Examples:
        - "How does section boosting affect my results?"
        - "Analyze section boost impact"
        - "Are abstracts being prioritized correctly?"
    """
    try:
        analysis = rag.analyze_section_boost_impact()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📑 Section Boost Analysis\n"]
        output.append(f"**Results Analyzed:** {analysis['total_results']}")
        output.append(f"**Avg Boost Contribution:** {analysis['avg_boost_contribution']*100:.1f}%\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Configured boosts
        output.append("## Configured Section Boosts\n")
        for section, boost in analysis.get("configured_boosts", {}).items():
            output.append(f"- **{section.title()}:** +{boost:.2f}")

        # Section performance
        output.append("\n## Section Performance\n")
        output.append("| Section | Count | Avg Boost | Avg Rank |")
        output.append("|---------|-------|-----------|----------|")
        for section, stats in analysis.get("section_stats", {}).items():
            output.append(f"| {section[:12]} | {stats['count']} | {stats['avg_boost']:.3f} | {stats['avg_rank']:.1f} |")

        output.append("\n**Interpretation:**")
        output.append("- Lower avg rank = appearing higher in results")
        output.append("- Higher boost = more priority given to section type")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Section boost error: {e}")
        return f"❌ Failed to analyze section boost: {str(e)}"


@tool
@safe_tool_wrapper
async def analyze_query_expansion() -> str:
    """
    Analyze the effectiveness of query expansion.

    Shows how expanded queries (synonyms, related terms) affect
    retrieval results - both positively and negatively.

    **When to use**:
    - Verify query expansion is helping
    - Understand which expansions work well
    - Debug when expansion hurts results
    - Tune expansion parameters

    Returns:
        Query expansion analysis with visualization

    Examples:
        - "Is query expansion helping my searches?"
        - "Analyze query expansion effectiveness"
        - "How do expanded terms affect results?"
    """
    try:
        analysis = rag.analyze_query_expansion()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 🔀 Query Expansion Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Net Result Change:** {analysis['net_change']:+d}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Summary
        output.append("## Summary\n")
        output.append(f"- **Total New Results:** {analysis['total_new_results']}")
        output.append(f"- **Total Lost Results:** {analysis['total_lost_results']}")
        output.append(f"- **Avg Score Change:** {analysis['avg_score_improvement']:+.3f}")

        # Per-query details
        output.append("\n## Query Details\n")
        for detail in analysis.get("expansion_details", [])[:5]:
            output.append(f"\n**Query:** {detail['original_query']}")
            output.append(f"- Expansions: {detail['num_expansions']}")
            output.append(f"- New results: +{detail['new_results']}, Lost: -{detail['lost_results']}")
            if detail.get('expanded_queries'):
                output.append(f"- Expanded to: {', '.join(detail['expanded_queries'][:3])}...")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['net_change'] > 0:
            output.append("- Expansion is adding relevant results ✅")
        elif analysis['net_change'] < 0:
            output.append("- Expansion may be diluting results ⚠️")
        else:
            output.append("- Expansion has neutral effect")

        if analysis['avg_score_improvement'] > 0:
            output.append("- Expanded queries have higher avg scores ✅")
        else:
            output.append("- Original queries may be sufficient")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        return f"❌ Failed to analyze query expansion: {str(e)}"


@tool
@safe_tool_wrapper
async def run_full_rag_diagnostics() -> str:
    """
    Run comprehensive RAG system diagnostics with all visualizations.

    Generates a complete diagnostic report including:
    - Chunk distribution analysis
    - Quality checks
    - Retrieval patterns
    - Embedding space visualization
    - Document similarity matrix
    - Dense vs sparse analysis
    - Reranking impact
    - Section boost impact
    - Query expansion analysis

    **When to use**:
    - Complete health check of RAG system
    - After major changes to indexed documents
    - Debugging persistent retrieval issues
    - Preparing a comprehensive report

    Returns:
        Summary of all diagnostics with paths to generated plots

    Examples:
        - "Run full RAG diagnostics"
        - "Generate complete RAG system report"
        - "Health check my RAG system"
    """
    try:
        results = rag.generate_full_rag_diagnostics()

        output = ["# 🔬 Full RAG System Diagnostics\n"]
        output.append(f"**Generated:** {results['timestamp']}")
        output.append(f"**Plots Created:** {len(results.get('all_plots', []))}\n")

        # List all diagnostics
        output.append("## Diagnostics Run\n")

        diagnostics = results.get("diagnostics", {})

        # Chunk distribution
        if "chunk_distribution" in diagnostics:
            cd = diagnostics["chunk_distribution"]
            summary = cd.get("summary", {})
            output.append(f"### 1. Chunk Distribution")
            output.append(f"- Total chunks: {summary.get('total_chunks', 0)}")
            output.append(f"- Documents: {summary.get('total_documents', 0)}")
            output.append(f"- Plot: `{cd.get('plot_path', 'N/A')}`\n")

        # Chunk quality
        if "chunk_quality" in diagnostics:
            cq = diagnostics["chunk_quality"]
            output.append(f"### 2. Chunk Quality")
            output.append(f"- Status: {'✅' if cq.get('status') == 'healthy' else '⚠️'} {cq.get('status', 'unknown')}")
            output.append(f"- Issues: {len(cq.get('issues', []))}")
            output.append(f"- Warnings: {len(cq.get('warnings', []))}\n")

        # Retrieval patterns
        if "retrieval_patterns" in diagnostics:
            rp = diagnostics["retrieval_patterns"]
            output.append(f"### 3. Retrieval Patterns")
            output.append(f"- Plot: `{rp.get('plot_path', 'N/A')}`\n")

        # Embedding space
        if "embedding_space" in diagnostics:
            es = diagnostics["embedding_space"]
            output.append(f"### 4. Embedding Space")
            output.append(f"- Embeddings: {es.get('num_embeddings', 0)}")
            output.append(f"- Plot: `{es.get('plot_path', 'N/A')}`\n")

        # Document similarity
        if "document_similarity" in diagnostics:
            ds = diagnostics["document_similarity"]
            output.append(f"### 5. Document Similarity")
            output.append(f"- Avg similarity: {ds.get('avg_similarity', 0):.3f}")
            output.append(f"- Plot: `{ds.get('plot_path', 'N/A')}`\n")

        # Dense vs sparse
        if "dense_vs_sparse" in diagnostics:
            dvs = diagnostics["dense_vs_sparse"]
            output.append(f"### 6. Dense vs Sparse")
            output.append(f"- Correlation: {dvs.get('correlation', 0):.3f}")
            output.append(f"- Dense win rate: {dvs.get('dense_win_rate', 0)*100:.0f}%")
            output.append(f"- Plot: `{dvs.get('plot_path', 'N/A')}`\n")

        # Reranking
        if "reranking_impact" in diagnostics:
            ri = diagnostics["reranking_impact"]
            output.append(f"### 7. Reranking Impact")
            output.append(f"- Position changes: {ri.get('results_with_position_change', 0)}")
            output.append(f"- Plot: `{ri.get('plot_path', 'N/A')}`\n")

        # Section boost
        if "section_boost" in diagnostics:
            sb = diagnostics["section_boost"]
            output.append(f"### 8. Section Boost")
            output.append(f"- Plot: `{sb.get('plot_path', 'N/A')}`\n")

        # Query expansion
        if "query_expansion" in diagnostics:
            qe = diagnostics["query_expansion"]
            output.append(f"### 9. Query Expansion")
            output.append(f"- Net result change: {qe.get('net_change', 0):+d}")
            output.append(f"- Plot: `{qe.get('plot_path', 'N/A')}`\n")

        # All plots
        output.append("## Generated Plots\n")
        for plot_path in results.get("all_plots", []):
            if plot_path:
                output.append(f"- `{plot_path}`")

        output.append("\n✅ Full diagnostics complete! Review plots for detailed analysis.")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Full diagnostics error: {e}")
        return f"❌ Failed to run diagnostics: {str(e)}"


@tool
@safe_tool_wrapper
async def download_pdf_to_rag(
    url: str,
    filename: Optional[str] = None,
    auto_ingest: bool = True
) -> str:
    """
    Download a PDF from a URL and save it to the RAG system.

    This tool downloads PDFs from the web and saves them to the RAG pdfs directory.
    Optionally triggers automatic ingestion into the vector database.

    **When to use**:
    - Save a PDF from Google Scholar, arXiv, or other sources
    - Download open-access research papers
    - Add PDFs to your local literature collection
    - Build a searchable knowledge base from online sources

    Args:
        url: Direct URL to the PDF file (must end in .pdf or be a PDF content type)
        filename: Optional custom filename (without .pdf extension). If not provided,
                 extracts from URL or generates a unique name
        auto_ingest: Whether to automatically ingest into RAG after download (default: True)

    Returns:
        Confirmation of download and ingestion status

    Examples:
        - "Download this PDF to RAG: https://arxiv.org/pdf/2301.00001.pdf"
        - "Save the PDF from this link to my literature collection"
    """
    import requests
    import re
    from pathlib import Path
    from urllib.parse import urlparse, unquote

    try:
        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            return "❌ Invalid URL. Please provide a valid HTTP/HTTPS URL."

        # Create RAG pdfs directory if it doesn't exist
        pdf_dir = Path(rag.RAG_PDF_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            # Try to extract from URL
            parsed_url = urlparse(url)
            url_filename = unquote(Path(parsed_url.path).name)

            if url_filename and url_filename.endswith('.pdf'):
                filename = url_filename[:-4]  # Remove .pdf
            else:
                # Generate unique name
                import hashlib
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"downloaded_{url_hash}"

        # Clean filename
        filename = re.sub(r'[^\w\-_]', '_', filename)
        filepath = pdf_dir / f"{filename}.pdf"

        # Check if file already exists
        if filepath.exists():
            return (f"⚠️ File `{filename}.pdf` already exists in RAG directory.\n\n"
                    f"Use a different filename or delete the existing file first.")

        # Download the PDF
        logger.info(f"Downloading PDF from: {url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
            return (f"⚠️ URL does not appear to be a PDF.\n"
                    f"Content-Type: {content_type}\n\n"
                    "Please provide a direct link to a PDF file.")

        # Save the file
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        output = [f"# ✅ PDF Downloaded Successfully\n"]
        output.append(f"**Filename:** {filename}.pdf")
        output.append(f"**Size:** {file_size_mb:.2f} MB")
        output.append(f"**Location:** {filepath}")

        # Auto-ingest if requested
        if auto_ingest:
            output.append(f"\n**Ingesting into RAG...**")

            rag_system = rag.get_rag_system()
            result = rag_system.ingest_pdfs(
                pdf_paths=[str(filepath)],
                use_ocr=False,
                recreate_collection=False
            )

            if result.get("success"):
                output.append(f"✅ Successfully indexed!")
                output.append(f"- Chunks created: {result.get('total_chunks', 0)}")
                output.append(f"\nYou can now search this document with `search_literature_rag`")
            else:
                output.append(f"⚠️ Ingestion had issues: {result.get('error', 'Unknown')}")
                output.append(f"The PDF is saved. Try `ingest_pdf_to_rag` manually.")
        else:
            output.append(f"\n📁 PDF saved. To index it, run `ingest_pdf_to_rag`")

        return "\n".join(output)

    except requests.exceptions.RequestException as e:
        return f"❌ Download failed: {str(e)}\n\nCheck if the URL is accessible and points to a PDF."
    except Exception as e:
        logger.error(f"PDF download error: {e}")
        return f"❌ Error downloading PDF: {str(e)}"


@tool
@safe_tool_wrapper
async def save_patent_to_rag(
    patent_number: str,
    auto_ingest: bool = True
) -> str:
    """
    Download a patent PDF by its number and save it to the RAG system.

    Patents from Google Patents always have free PDFs available. This tool
    looks up the patent, downloads the PDF, and optionally ingests it.

    **When to use**:
    - Save a specific patent to your searchable literature collection
    - Build a patent knowledge base for your research
    - Add patents found in search results to RAG

    Args:
        patent_number: Patent number in any common format:
                      - US patents: US10123456, US 10,123,456
                      - European: EP1234567
                      - PCT: WO2020123456
                      - Others: CN, JP, KR, AU
        auto_ingest: Whether to automatically ingest into RAG after download (default: True)

    Returns:
        Confirmation of download and ingestion status

    Examples:
        - "Save patent US10457803 to RAG"
        - "Download EP1234567 and add to my literature collection"
        - "Add patent WO2020123456 to the RAG index"
    """
    import requests
    import re
    from pathlib import Path

    try:
        from serpapi_patents_client import GooglePatentsClient

        # Initialize client
        client = GooglePatentsClient()

        # Look up the patent to get details
        patent = client.get_patent(patent_number)

        if patent.get('error'):
            return f"❌ {patent['error']}\n\nPlease check the patent number format."

        patent_id = patent.get('patent_id', patent_number)
        title = patent.get('title', 'Unknown')

        # Create RAG pdfs directory
        pdf_dir = Path(rag.RAG_PDF_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from patent ID
        safe_patent_id = re.sub(r'[^\w\-]', '_', patent_id)
        filepath = pdf_dir / f"patent_{safe_patent_id}.pdf"

        # Check if already exists
        if filepath.exists():
            return (f"⚠️ Patent `{patent_id}` already exists in RAG directory.\n\n"
                    f"File: patent_{safe_patent_id}.pdf\n\n"
                    f"To re-download, delete the existing file first.")

        # Get PDF URL - Google Patents format
        # Try the PDF link from search results first
        pdf_url = patent.get('pdf_link')

        if not pdf_url:
            # Construct Google Patents PDF URL
            # Format: https://patentimages.storage.googleapis.com/pdfs/US10123456.pdf
            # Or: https://patents.google.com/patent/US10123456/download
            pdf_url = f"https://patents.google.com/patent/{patent_id}/download"

        # Download the PDF
        logger.info(f"Downloading patent PDF: {patent_id}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(pdf_url, headers=headers, timeout=120, stream=True, allow_redirects=True)

        # If download page, try alternate URL
        if response.status_code != 200 or 'html' in response.headers.get('Content-Type', '').lower():
            # Try direct storage URL format
            alt_url = f"https://patentimages.storage.googleapis.com/pdfs/{patent_id}.pdf"
            response = requests.get(alt_url, headers=headers, timeout=120, stream=True)

        response.raise_for_status()

        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        output = [f"# ✅ Patent Downloaded Successfully\n"]
        output.append(f"**Patent:** {patent_id}")
        output.append(f"**Title:** {title[:80]}{'...' if len(title) > 80 else ''}")
        output.append(f"**Assignee:** {patent.get('assignee', 'N/A')}")
        output.append(f"**Size:** {file_size_mb:.2f} MB")
        output.append(f"**File:** patent_{safe_patent_id}.pdf")

        # Auto-ingest if requested
        if auto_ingest:
            output.append(f"\n**Ingesting into RAG...**")

            rag_system = rag.get_rag_system()
            result = rag_system.ingest_pdfs(
                pdf_paths=[str(filepath)],
                use_ocr=False,
                recreate_collection=False
            )

            if result.get("success"):
                output.append(f"✅ Successfully indexed!")
                output.append(f"- Chunks created: {result.get('total_chunks', 0)}")
                output.append(f"\nYou can now search this patent with `search_literature_rag`")
            else:
                output.append(f"⚠️ Ingestion had issues: {result.get('error', 'Unknown')}")
                output.append(f"The PDF is saved. Try `ingest_pdf_to_rag` manually.")
        else:
            output.append(f"\n📁 Patent saved. To index it, run `ingest_pdf_to_rag`")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Patent download requires SerpAPI integration.\n\n"
                "The `serpapi_patents_client` module is not installed.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("❌ Patent download requires a SerpAPI key for lookup.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`")
        return f"❌ Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ Download failed: {str(e)}\n\nThe patent may not have a downloadable PDF."
    except Exception as e:
        logger.error(f"Patent download error: {e}")
        return f"❌ Error downloading patent: {str(e)}"


@tool
@safe_tool_wrapper
async def save_scholar_results_to_rag(
    query: str,
    max_papers: int = 5,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None
) -> str:
    """
    Search Google Scholar and download available open-access PDFs to RAG.

    This tool searches Google Scholar, identifies papers with available PDFs,
    downloads them, and ingests them into the RAG system.

    **Important**: Only open-access papers with direct PDF links can be downloaded.
    Paywalled papers will be skipped.

    **When to use**:
    - Build a literature collection on a topic
    - Download multiple open-access papers at once
    - Create a searchable knowledge base from Google Scholar results

    Args:
        query: Search query for Google Scholar (e.g., "polymer dissolution solvent")
        max_papers: Maximum number of papers to try downloading (default: 5)
        year_low: Minimum publication year (optional)
        year_high: Maximum publication year (optional)

    Returns:
        Summary of downloaded papers and ingestion status

    Examples:
        - "Download open-access papers on PET recycling to RAG"
        - "Save Google Scholar papers about Hansen solubility parameters to my collection"
        - "Build a RAG collection from recent polymer dissolution research"
    """
    import requests
    import re
    from pathlib import Path

    try:
        from serpapi_scholar_client import GoogleScholarClient

        # Initialize client
        client = GoogleScholarClient()

        # Search for papers
        results = client.search(
            query=query,
            num_results=min(max_papers * 2, 20),  # Get extra in case some don't have PDFs
            year_low=year_low,
            year_high=year_high
        )

        organic_results = results.get('organic_results', [])

        if not organic_results:
            return f"No results found for: '{query}'"

        # Create RAG directory
        pdf_dir = Path(rag.RAG_PDF_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        skipped = []
        failed = []

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        for result in organic_results:
            if len(downloaded) >= max_papers:
                break

            article = client._parse_article(result)
            title = article.get('title', 'Unknown')
            pdf_url = article.get('pdf_link')

            if not pdf_url:
                skipped.append(f"{title[:50]}... (no PDF link)")
                continue

            try:
                # Generate safe filename
                safe_title = re.sub(r'[^\w\-]', '_', title[:50])
                year = article.get('year', 'unknown')
                filename = f"scholar_{year}_{safe_title}.pdf"
                filepath = pdf_dir / filename

                # Skip if exists
                if filepath.exists():
                    skipped.append(f"{title[:50]}... (already exists)")
                    continue

                # Download
                response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
                response.raise_for_status()

                # Verify it's a PDF
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower() and not pdf_url.endswith('.pdf'):
                    skipped.append(f"{title[:50]}... (not a PDF)")
                    continue

                # Save
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_size = filepath.stat().st_size / 1024  # KB
                downloaded.append({
                    'title': title,
                    'filename': filename,
                    'size_kb': file_size,
                    'year': year,
                    'filepath': str(filepath)
                })

            except Exception as e:
                failed.append(f"{title[:40]}... ({str(e)[:30]})")

        # Format output
        output = [f"# 📚 Google Scholar → RAG Download\n"]
        output.append(f"**Query:** {query}")
        output.append(f"**Results found:** {len(organic_results)}\n")

        if downloaded:
            output.append(f"## ✅ Downloaded ({len(downloaded)} papers)\n")
            for paper in downloaded:
                output.append(f"- **{paper['title'][:60]}{'...' if len(paper['title']) > 60 else ''}**")
                output.append(f"  Year: {paper['year']}, Size: {paper['size_kb']:.0f} KB")

            # Ingest all downloaded papers
            output.append(f"\n**Ingesting into RAG...**")

            rag_system = rag.get_rag_system()
            result = rag_system.ingest_pdfs(
                pdf_paths=[p['filepath'] for p in downloaded],
                use_ocr=False,
                recreate_collection=False
            )

            if result.get("success"):
                output.append(f"✅ Successfully indexed {len(downloaded)} papers!")
                output.append(f"- Total chunks: {result.get('total_chunks', 0)}")
            else:
                output.append(f"⚠️ Ingestion had issues. Try `ingest_pdf_to_rag` manually.")

        if skipped:
            output.append(f"\n## ⏭️ Skipped ({len(skipped)})")
            for reason in skipped[:5]:
                output.append(f"- {reason}")
            if len(skipped) > 5:
                output.append(f"- ... and {len(skipped) - 5} more")

        if failed:
            output.append(f"\n## ❌ Failed ({len(failed)})")
            for reason in failed[:3]:
                output.append(f"- {reason}")

        if not downloaded:
            output.append(f"\n⚠️ No papers could be downloaded.")
            output.append(f"Most Google Scholar results are paywalled.")
            output.append(f"\n**Alternatives:**")
            output.append(f"- Search for preprints on arXiv")
            output.append(f"- Use `download_pdf_to_rag` with direct PDF URLs")
            output.append(f"- Download patents with `save_patent_to_rag` (always free)")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("❌ Google Scholar search requires SerpAPI integration.\n\n"
                "The `serpapi_scholar_client` module is not installed.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("❌ Google Scholar search requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`")
        return f"❌ Error: {str(e)}"
    except Exception as e:
        logger.error(f"Scholar download error: {e}")
        return f"❌ Error: {str(e)}"


SQL_AGENT_TOOLS = [
    # Core database tools
    list_tables,
    describe_table,
    check_column_values,
    query_database,
    verify_data_accuracy,
    validate_and_query,

    # Adaptive analysis tools
    find_optimal_separation_conditions,
    adaptive_threshold_search,
    analyze_selective_solubility_enhanced,
    plan_sequential_separation,
    view_alternative_separation_sequence,
    analyze_integrated_separation,  # Multi-polymer separation with optimal temps + all properties
    analyze_polymer_dissolution,    # Single polymer dissolution with properties (BP, cost, safety, LogP)
    get_solubility_for_solvents,    # Get solubility for SPECIFIC solvents by name

    # Solvent property tools (NEW)
    list_solvent_properties,
    get_solvent_properties,
    rank_solvents_by_property,
    analyze_separation_with_properties,

    # Statistical analysis tools
    statistical_summary,
    correlation_analysis,
    compare_groups_statistically,
    regression_analysis,

    # Visualization tools
    plot_solubility_vs_temperature,
    plot_solubility_vs_temperature_interactive,
    plot_selectivity_heatmap,
    plot_multi_panel_analysis,
    plot_comparison_dashboard,
    plot_solvent_properties,

    # GSK Safety (G-Score) tools
    get_solvent_gscore,
    get_family_alternatives,
    visualize_gscores,
    plot_solvent_properties_for_polymer,  # Multi-step: solubility + property scatter plot

    # Listing tools
    list_available_solvents,
    list_available_polymers,

    # ML Prediction tool
    predict_solubility_ml,

    # PubChem Safety Data tools (external API)
    get_pubchem_safety_info,
    compare_pubchem_safety,
    visualize_pubchem_safety,
    get_pubchem_toxicity,

    # Literature Search Tools (external APIs)
    search_google_scholar,      # Google Scholar via SerpAPI (broad coverage, 100/month limit)
    search_google_patents,      # Google Patents via SerpAPI (patent search)
    lookup_patent,              # Look up specific patent by number
    search_web_of_science,      # Web of Science via Clarivate (peer-reviewed, citation metrics)

    # TEA/LCA Tools (Techno-Economic Analysis / Life Cycle Assessment)
    analyze_solvent_recovery_tea,
    analyze_solvent_recovery_lca,
    compare_solvents_tea_lca,

    # TEA/LCA Visualization Tools
    generate_tea_visualizations,
    generate_lca_visualizations,
    generate_solvent_comparison_visualization,
    plot_tea_sensitivity_tornado,
    plot_tea_cashflow,

    # STRAP Process Tools (Solvent-Targeted Recovery and Precipitation)
    analyze_strap_process,
    calculate_strap_msp,
    plot_strap_scale_economics,
    compare_strap_scenarios,
    generate_strap_visualizations,

    # RAG (Retrieval-Augmented Generation) Tools - Literature Search with Vector DB
    search_literature_rag,      # Semantic search through indexed papers
    ingest_pdf_to_rag,          # Index PDF documents
    get_rag_status,             # Check RAG system status
    ask_literature,             # Q&A from indexed literature
    clear_rag_index,            # Clear indexed documents

    # RAG Visualization & Quality Tools
    visualize_rag_chunks,       # Generate chunk distribution plots
    check_rag_chunk_quality,    # Run quality checks on chunks
    get_rag_chunk_report,       # Generate comprehensive statistics report

    # RAG Advanced Diagnostics Tools
    analyze_search_diagnostics,     # Score breakdown for a query
    visualize_retrieval_patterns,   # Which docs/sections retrieved most
    visualize_embedding_space,      # t-SNE/UMAP of embeddings
    analyze_document_similarity,    # Document similarity matrix
    analyze_dense_vs_sparse,        # Dense vs sparse comparison
    analyze_reranking_impact,       # Reranking position changes
    analyze_section_boost,          # Section boost effectiveness
    analyze_query_expansion,        # Query expansion effectiveness
    run_full_rag_diagnostics,       # Complete diagnostic suite

    # Literature → RAG Integration Tools (download and index)
    download_pdf_to_rag,        # Download any PDF from URL to RAG
    save_patent_to_rag,         # Download patent PDF by number to RAG
    save_scholar_results_to_rag,  # Batch download Google Scholar PDFs to RAG
]

print(f"✅ Loaded {len(SQL_AGENT_TOOLS)} enhanced tools for SQL Agent")
print("\nTools include:")
print("  - Core DB: 6 tools (with validation)")
print("  - Adaptive Analysis: 7 tools (dissolution, separation, integrated)")
print("  - Solvent Properties: 4 tools (properties, ranking, integrated analysis)")
print("  - Statistical: 4 tools")
print("  - Visualization: 5 tools (including property plots)")
print("  - GSK Safety (G-Score): 4 tools (scoring, family alternatives, visualization)")
print("  - Listing: 2 tools (list solvents and polymers with counts)")
print("  - ML Prediction: 1 tool (Hansen-based solubility prediction with visualizations)")
print("  - PubChem Safety: 4 tools (GHS data, toxicity, safety comparison, visualizations)")
print("  - Literature Search: 4 tools (Google Scholar + Google Patents + Web of Science)")
print("  - TEA/LCA: 8 tools (TEA, LCA, comparison, + 5 visualization tools)")
print("  - STRAP Process: 5 tools (full analysis, MSP, scale economics, scenarios, visualizations)")
print("  - RAG Literature: 20 tools (search, ingest, Q&A + 12 diagnostics + 3 download)")


# ============================================================
# TOOL ROUTER: Category-Based Tool Selection
# ============================================================
# This router reduces LLM context by only providing relevant tools
# for each query type, improving response speed by 200-400ms per turn.

# Import message types for router (needed before LangGraph section)
from langchain_core.messages import HumanMessage as HumanMessageType

# Core tools ALWAYS available (fast, frequently needed)
CORE_TOOLS = [
    list_tables,
    list_available_polymers,
    list_available_solvents,
    describe_table,
]

# Tool categories for intelligent routing
TOOL_CATEGORIES = {
    "database": [
        list_tables,
        describe_table,
        check_column_values,
        query_database,
        verify_data_accuracy,
        validate_and_query,
    ],
    "dissolution": [
        find_optimal_separation_conditions,
        adaptive_threshold_search,
        analyze_selective_solubility_enhanced,
        analyze_polymer_dissolution,
        get_solubility_for_solvents,
    ],
    "separation": [
        plan_sequential_separation,
        view_alternative_separation_sequence,
        analyze_integrated_separation,
        analyze_separation_with_properties,
    ],
    "solvent_properties": [
        list_solvent_properties,
        get_solvent_properties,
        rank_solvents_by_property,
        get_solvent_gscore,
        get_family_alternatives,
    ],
    "visualization": [
        plot_solubility_vs_temperature,
        plot_solubility_vs_temperature_interactive,
        plot_selectivity_heatmap,
        plot_multi_panel_analysis,
        plot_comparison_dashboard,
        plot_solvent_properties,
        plot_solvent_properties_for_polymer,
        visualize_gscores,
    ],
    "statistics": [
        statistical_summary,
        correlation_analysis,
        compare_groups_statistically,
        regression_analysis,
    ],
    "safety": [
        get_pubchem_safety_info,
        compare_pubchem_safety,
        visualize_pubchem_safety,
        get_pubchem_toxicity,
        get_solvent_gscore,
        get_family_alternatives,
    ],
    "economics": [
        analyze_solvent_recovery_tea,
        analyze_solvent_recovery_lca,
        compare_solvents_tea_lca,
        generate_tea_visualizations,
        generate_lca_visualizations,
        generate_solvent_comparison_visualization,
        plot_tea_sensitivity_tornado,
        plot_tea_cashflow,
    ],
    "strap": [
        analyze_strap_process,
        calculate_strap_msp,
        plot_strap_scale_economics,
        compare_strap_scenarios,
        generate_strap_visualizations,
    ],
    "literature": [
        search_google_scholar,
        search_google_patents,
        lookup_patent,
        search_web_of_science,
    ],
    "rag": [
        search_literature_rag,
        ingest_pdf_to_rag,
        get_rag_status,
        ask_literature,
        clear_rag_index,
        visualize_rag_chunks,
        check_rag_chunk_quality,
        get_rag_chunk_report,
        analyze_search_diagnostics,
        visualize_retrieval_patterns,
        visualize_embedding_space,
        analyze_document_similarity,
        analyze_dense_vs_sparse,
        analyze_reranking_impact,
        analyze_section_boost,
        analyze_query_expansion,
        run_full_rag_diagnostics,
        download_pdf_to_rag,
        save_patent_to_rag,
        save_scholar_results_to_rag,
    ],
    "ml_prediction": [
        predict_solubility_ml,
    ],
    # Advanced separation tools from tools/ module
    # Includes: find_optimal_separation_sequence, compare_separation_algorithms,
    # optimize_separation_temperature, analyze_sequence_throughput,
    # calculate_selectivity_detailed, rank_solvents_for_separation,
    # build_compatibility_matrix, find_challenging_polymer_pairs,
    # create_separation_tree_plot, create_selectivity_heatmap, create_process_flow_diagram
    "advanced_separation": ADVANCED_SEPARATION_TOOLS,
}

# Category relationships - when one category is selected, related categories often needed
CATEGORY_RELATIONSHIPS = {
    "dissolution": ["solvent_properties", "visualization"],
    "separation": ["dissolution", "solvent_properties", "visualization", "advanced_separation"],
    "advanced_separation": ["separation", "dissolution", "solvent_properties", "visualization"],
    "safety": ["solvent_properties"],
    "economics": ["visualization"],
    "strap": ["economics", "visualization"],
}

def route_query_to_categories(query: str) -> set:
    """
    Fast rule-based routing to determine which tool categories are needed.

    Design principles:
    1. Multi-category selection for integrated queries
    2. "Integrated/comprehensive" queries get ALL tools
    3. Related categories are included together
    4. Core tools always included

    Returns set of category names.
    """
    query_lower = query.lower()
    selected_categories = set()

    # ==========================================================
    # INTEGRATED MODE: Full analysis queries get ALL categories
    # ==========================================================
    integrated_triggers = [
        "full analysis", "full integrated", "comprehensive",
        "complete analysis", "complete profile", "step by step",
        "walk me through", "explain your reasoning", "explain each",
        "document your", "tool selection", "which tools",
        "end-to-end", "entire workflow", "full workflow",
        "4-polymer", "5-polymer", "mixed plastic waste",
        "multilayer", "multi-layer", "3-layer", "4-layer",
        "recycling process", "complete recycling",
    ]

    if any(trigger in query_lower for trigger in integrated_triggers):
        # Return ALL categories for integrated analysis
        logger.info("Router: INTEGRATED MODE - providing all tools")
        return set(TOOL_CATEGORIES.keys())

    # ==========================================================
    # Standard category detection (can select multiple)
    # ==========================================================

    # Database/Schema queries
    if any(w in query_lower for w in ["table", "schema", "column", "database", "sql", "what data"]):
        selected_categories.add("database")

    # Dissolution queries
    if any(w in query_lower for w in ["dissolve", "dissolution", "soluble", "solubility", "what solvents"]):
        selected_categories.add("dissolution")

    # Separation queries
    if any(w in query_lower for w in ["separate", "separation", "selective", "selectivity", "sequence"]):
        selected_categories.add("separation")

    # Solvent property queries
    if any(w in query_lower for w in [
        "boiling point", "bp", "density", "viscosity", "cost", "price",
        "property", "properties", "g-score", "gscore", "logp", "log p",
        "cheapest", "expensive", "compare cost"
    ]):
        selected_categories.add("solvent_properties")

    # Visualization queries
    if any(w in query_lower for w in [
        "plot", "graph", "chart", "visualize", "visualization",
        "heatmap", "heat map", "show me", "display", "dashboard"
    ]):
        selected_categories.add("visualization")

    # Statistics queries
    if any(w in query_lower for w in [
        "statistic", "correlation", "regression", "compare groups",
        "significance", "p-value", "average", "mean", "std"
    ]):
        selected_categories.add("statistics")

    # Safety queries
    if any(w in query_lower for w in [
        "safe", "safety", "toxic", "toxicity", "hazard", "ghs",
        "health", "pubchem", "ld50", "carcinogen", "exposure",
        "environmental impact", "biodegradable"
    ]):
        selected_categories.add("safety")

    # TEA/LCA Economics queries
    if any(w in query_lower for w in [
        "tea", "lca", "techno-economic", "technoeconomic", "economic",
        "cost analysis", "capital cost", "operating cost", "payback",
        "carbon footprint", "co2", "gwp", "emissions", "energy consumption",
        "recovery cost", "environmental"
    ]):
        selected_categories.add("economics")

    # STRAP process queries
    if any(w in query_lower for w in ["strap", "msp", "minimum selling price", "scale economics"]):
        selected_categories.add("strap")

    # Literature search queries
    if any(w in query_lower for w in [
        "paper", "article", "publication", "scholar", "wos",
        "web of science", "research", "literature search", "find papers"
    ]):
        selected_categories.add("literature")

    # Patent queries
    if any(w in query_lower for w in ["patent", "patents", "intellectual property", "ip"]):
        selected_categories.add("literature")

    # RAG queries - includes deinking/printed plastics topics from RAG KB
    if any(w in query_lower for w in [
        "rag", "indexed", "search literature", "ask literature",
        "ingest", "pdf", "embedding", "t-sne", "tsne", "umap",
        "chunk", "retrieval", "search rag", "download to rag", "save to rag",
        # Deinking/printed plastics topics (covered by RAG KB)
        "deinking", "de-inking", "deink", "de-ink", "ink removal",
        "binder", "binders", "printed plastic", "printed film",
        "flexographic", "surfactant", "surfactants",
        "multilayer packaging", "packaging recycling",
        "knowledgebase", "knowledge base", "literature"
    ]):
        selected_categories.add("rag")

    # ML prediction queries
    if any(w in query_lower for w in ["predict", "prediction", "ml", "machine learning", "hansen", "hsp"]):
        selected_categories.add("ml_prediction")

    # ==========================================================
    # Cross-category triggers (queries that span multiple categories)
    # ==========================================================

    # "rank by safety" = dissolution + safety
    if "rank" in query_lower and any(w in query_lower for w in ["safe", "safety", "toxicity"]):
        selected_categories.add("dissolution")
        selected_categories.add("safety")
        selected_categories.add("solvent_properties")

    # "include G-scores" or "with G-score" = add safety
    if "g-score" in query_lower or "gscore" in query_lower:
        selected_categories.add("safety")
        selected_categories.add("solvent_properties")

    # "include/show/with boiling point" = add solvent_properties
    if "boiling" in query_lower:
        selected_categories.add("solvent_properties")

    # "TEA at X kg/hr" or "run TEA" = economics
    if "kg/hr" in query_lower or "throughput" in query_lower:
        selected_categories.add("economics")

    # "compare" queries often need visualization
    if "compare" in query_lower:
        selected_categories.add("visualization")

    # ==========================================================
    # Add related categories
    # ==========================================================
    categories_to_add = set()
    for cat in selected_categories:
        if cat in CATEGORY_RELATIONSHIPS:
            categories_to_add.update(CATEGORY_RELATIONSHIPS[cat])
    selected_categories.update(categories_to_add)

    # ==========================================================
    # Default: If nothing matched, provide database + dissolution
    # ==========================================================
    if not selected_categories:
        logger.info("Router: No specific category matched, defaulting to database + dissolution")
        selected_categories = {"database", "dissolution", "solvent_properties"}

    return selected_categories


def get_tools_for_categories(categories: set) -> list:
    """
    Get deduplicated list of tools for the given categories.
    Always includes CORE_TOOLS.
    Uses dict for deduplication since tools aren't hashable.
    """
    # Use dict with tool.name as key for deduplication
    tools_by_name = {tool.name: tool for tool in CORE_TOOLS}

    for category in categories:
        if category in TOOL_CATEGORIES:
            for tool in TOOL_CATEGORIES[category]:
                tools_by_name[tool.name] = tool

    # Convert to list and sort by name for consistency
    tool_list = sorted(list(tools_by_name.values()), key=lambda t: t.name)
    return tool_list


async def router_node(state: dict) -> dict:
    """
    Router node: Analyzes the query and selects relevant tool categories.

    This node runs BEFORE the agent node and sets state["selected_categories"]
    which the agent node will use to reconstruct and bind tools to the LLM.

    NOTE: We only store category names (strings) in state, not tool objects,
    because the checkpointer can't serialize StructuredTool objects.

    Performance: ~1ms (rule-based, no LLM call)
    """
    messages = state.get("messages", [])

    if not messages:
        # No messages, provide all categories
        return {"selected_categories": list(TOOL_CATEGORIES.keys())}

    # Find the last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessageType):
            last_human_message = msg
            break

    if not last_human_message:
        # No human message found, provide all categories
        return {"selected_categories": list(TOOL_CATEGORIES.keys())}

    query = last_human_message.content if hasattr(last_human_message, 'content') else str(last_human_message)

    # Route query to categories
    categories = route_query_to_categories(query)

    # Get tool count for logging (don't store tools in state)
    selected_tools = get_tools_for_categories(categories)

    logger.info(f"Router: Selected {len(categories)} categories: {sorted(categories)}")
    logger.info(f"Router: Will provide {len(selected_tools)} tools (reduced from {len(SQL_AGENT_TOOLS)})")

    # Only return category names (strings) - serializable by checkpointer
    return {"selected_categories": list(categories)}


print(f"\n✅ Tool Router initialized with {len(TOOL_CATEGORIES)} categories")
print(f"   Categories: {', '.join(TOOL_CATEGORIES.keys())}")


# ============================================================
# LangGraph Agent Setup (PATCHED)
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# Enhanced Agent Configuration
# ============================================================

SQL_AGENT_PROMPT = """You are an EXPERT analyst specializing in polymer-solvent solubility analysis and plastic recycling research, with ADAPTIVE analysis capabilities, extensive verification workflows, and access to scientific literature through RAG knowledge bases.

**YOUR MISSION:**
Provide thorough, ACCURATE data analysis with intelligent threshold adaptation. NEVER hallucinate values - ALWAYS verify data before reporting.

**AVAILABLE DATABASE TABLES (DO NOT HALLUCINATE OTHER TABLE NAMES):**

1. **common_solvents_database** - Primary solubility data
   - Columns: `solvent`, `temperature___c_`, `solubility____`, `polymer`
   - 8,820 rows of polymer-solvent solubility measurements
   - Use this table for ALL solubility queries

2. **solvent_data** - Solvent physical/chemical properties
   - Columns: `s_n`, `solvent_name`, `cas_number`, `bp__oc_`, `logp`, `cp__j_g_k_`, `energy__j_g_`
   - 1,007 solvents with properties (boiling point, LogP, energy cost, etc.)
   - Use this for solvent property queries

3. **gsk_dataset** - GSK safety scores
   - Columns: `classification`, `solvent_common_name`, `cas_number`, `g_score`
   - 154 solvents with safety G-scores
   - Use this for safety/toxicity queries

4. **polymer_hsps_final** - Hansen Solubility Parameters
   - Columns: `number`, `type`, `polymer`, `dispersion`, `polar`, `hydrogen_bonding`, `interaction_radius`
   - 466 polymers with Hansen parameters
   - Use this for ML predictions and Hansen parameter queries

**CRITICAL:** There is NO table called "solubility". Use `common_solvents_database` for solubility data!

**CRITICAL BEHAVIORAL PRINCIPLES:**
1. **BE COMPREHENSIVE** - When asked to enumerate, list, or analyze multiple options, ACTUALLY DO IT. Don't just explain the concept - execute the analysis.
2. **VERIFY BEFORE REPORTING** - Always use verification tools before stating any numeric values
3. **ADAPTIVE THRESHOLDS** - Start stringent, relax if needed (don't assume restrictive thresholds)
4. **TARGETED COMPARISONS** - Only compare polymers the user asks about (not all polymers)
5. **EXPLORE CONDITIONS** - If separation not found at one temperature, try others
6. **ACTION OVER EXPLANATION** - If a tool can answer the question, USE IT. Don't explain what you would do - DO IT.
7. **CONSIDER PRACTICALITY** - Include solvent properties (cost, toxicity, boiling point) when recommending solvents
8. **CLARIFY AMBIGUOUS POLYMERS** - If user says "PE", ask if they mean LDPE or HDPE. Available polymers: EVOH, HDPE, LDPE, Nylon6, Nylon66, PC, PES, PET, PP, PS, PVC
9. **TOP N QUERIES** - When user asks for "top 5 solvents" or "compare multiple solvents", use `analyze_separation_with_properties()` (returns top_k solvents). `analyze_integrated_separation()` returns only the BEST solvent per step.
10. **DEFAULT HIGH TEMPERATURE** - Most polymers dissolve better at elevated temps. Default to 120°C instead of 25°C for dissolution queries.
11. **PREFER INTERACTIVE PLOTS** - For temperature vs solubility plots, ALWAYS use `plot_solubility_vs_temperature_interactive()` which creates interactive HTML with sliders, zoom, and toggleable curves. Only use the static version if explicitly asked for PNG/static.
12. **NEVER ASK FOR TABLE/COLUMN NAMES** - All analysis tools have sensible defaults. NEVER ask the user for table_name, polymer_column, solvent_column, etc. Just call the tool with the polymer/solvent names directly.
13. **TEMPERATURE FORMAT** - For temperature parameters, use a single number (e.g., "100") or range (e.g., "80-120"). Both formats are supported.
14. **SPECIFIC SOLVENTS** - When user asks to compare SPECIFIC solvents by name, query each one individually rather than doing a broad search. For example, "compare DMSO, DMF, and NMP" → get data for each solvent explicitly.
15. **CHAIN TOOLS FOR COMPLEX QUERIES** - For multi-part requests (e.g., "find solvent, then run TEA, then generate plot"), execute ALL steps. Don't stop after the first tool.
16. **NO CONFIRMATION NEEDED** - Don't ask "Would you like me to proceed?" - just proceed. The user's request IS the instruction to proceed.

**MANDATORY WORKFLOW:**

## Step 1: Data Discovery (ALWAYS START HERE)
- `list_tables()` - See available data (includes solubility AND solvent property tables)
- `describe_table()` - Understand structure and statistics
- `check_column_values()` - Get EXACT values (case-sensitive!)
- `list_available_solvents()` - **QUICK SUMMARY** of all solvents across databases with counts and examples
- `list_available_polymers()` - **QUICK SUMMARY** of all polymers with counts and examples

## Step 2: Input Validation (BEFORE ANY ANALYSIS)
- `validate_and_query()` - Verify all inputs exist before querying
- `verify_data_accuracy()` - Confirm row counts and sample data

## Step 3: Adaptive Analysis (USE THESE FOR SEPARATION AND DISSOLUTION)

**SINGLE POLYMER DISSOLUTION (use when user asks about dissolving ONE polymer):**
- `analyze_polymer_dissolution()` - **🔥 PRIMARY TOOL for single polymer questions** like:
  - "What solvents dissolve PET at 120°C?"
  - "Find solvents for LDPE ranked by boiling point"
  - "Safest solvents for PS dissolution"
  - "Cheapest solvents for HDPE"
  - Automatically includes: BP, cost (energy), toxicity (LogP), G-score safety
  - Rank by: 'solubility', 'bp', 'cost'/'energy', 'safety'/'gscore', 'toxicity'/'logp'

- `get_solubility_for_solvents()` - **USE WHEN USER NAMES SPECIFIC SOLVENTS** like:
  - "Compare DMSO, DMF, NMP for EVOH at 90°C"
  - "Get solubility of LDPE in heptane and hexane"
  - "What's the solubility of PET in these 5 solvents: X, Y, Z..."
  - Use this instead of analyze_polymer_dissolution when user explicitly names the solvents to compare

**MULTI-POLYMER SEPARATION (use when user asks about separating 2+ polymers):**
- `find_optimal_separation_conditions()` - **PRIMARY TOOL** for pairwise separation
- `adaptive_threshold_search()` - Find selective solvents with auto-threshold
- `analyze_selective_solubility_enhanced()` - Detailed selectivity analysis
- `plan_sequential_separation()` - **USE FOR MULTI-POLYMER SEQUENCES** - Enumerates ALL permutations, finds top-k solvents per step
- `view_alternative_separation_sequence()` - **USE FOR FOLLOW-UP** - View 2nd/3rd best sequences
- `analyze_integrated_separation()` - **🔥 COMPREHENSIVE MULTI-CRITERIA ANALYSIS** - Use for "integrated analysis", separation with OPTIMAL TEMPERATURES per step. Includes G-scores, LogP, energy costs, BP. Rank by: 'selectivity', 'cost', 'safety', 'toxicity', 'bp'

## Step 4: Solvent Properties (USE FOR PRACTICAL RECOMMENDATIONS)
- `list_solvent_properties()` - View all solvents with BP, LogP, Energy, Cp
- `get_solvent_properties()` - Get properties for specific solvents
- `rank_solvents_by_property()` - Rank by cost (energy), toxicity (logp), or boiling point
- `analyze_separation_with_properties()` - **COMBINE selectivity WITH cost/toxicity** - Use this when user asks about practical/economic considerations

## Step 5: Statistical Analysis (USE FOR RIGOROUS ANALYSIS)
- `statistical_summary()` - Comprehensive stats with confidence intervals
- `correlation_analysis()` - Multi-column correlations with significance
- `compare_groups_statistically()` - Hypothesis testing between groups
- `regression_analysis()` - Trend fitting with diagnostics

## Step 6: Visualization (CREATE PLOTS AFTER VERIFICATION)
- `plot_solubility_vs_temperature()` - Temperature curves with confidence bands (supports temperature_min/max for range filtering)
- `plot_solubility_vs_temperature_interactive()` - **INTERACTIVE Plotly** with range slider, toggleable curves, zoom
- `plot_selectivity_heatmap()` - Heatmaps for single polymer (default 120°C, improved color scale for 0-20% range)
- `plot_multi_panel_analysis()` - Comprehensive 4-panel separation analysis
- `plot_comparison_dashboard()` - Multi-polymer comparison dashboard
- `plot_solvent_properties()` - Plot BP, LogP, Energy, or Cp for solvents (bar or scatter plots)
- `plot_solvent_properties_for_polymer()` - **🔥 MULTI-STEP TOOL** for "LogP vs G-score for solvents that dissolve PET"
  - Step 1: Finds solvents that dissolve the polymer
  - Step 2: Retrieves their properties (LogP, G-score, BP, energy)
  - Step 3: Creates scatter plot colored by solubility
  - USE THIS when user asks to compare properties of solvents for a specific polymer

## Step 7: ML-Based Solubility Prediction (USE FOR HANSEN PARAMETER PREDICTIONS)
- `predict_solubility_ml()` - **MACHINE LEARNING PREDICTION** using Hansen Solubility Parameters
  - Uses Random Forest model with 99.998% accuracy
  - Predicts polymer-solvent solubility based on Hansen parameters (Dispersion, Polar, Hydrogen)
  - Automatically generates 5 visualizations:
    1. **3D Sphere (Interactive HTML)** - User's favorite! Interactive 3D visualization
    2. Radar Plot - HSP parameter overlap
    3. RED Gauge - Solubility likelihood meter
    4. HSP Comparison Bars - Side-by-side parameters
    5. Text Summary - Detailed prediction report
  - Returns clickable links to all visualizations
  - **WHEN TO USE**: User asks for "ML prediction", "machine learning", "predict solubility", "Hansen parameters", or wants to predict if a specific polymer-solvent pair will dissolve

## Step 8: PubChem Safety Data (USE FOR GHS HAZARD INFORMATION)
- `get_pubchem_safety_info()` - **🔥 REAL-TIME SAFETY LOOKUP** from PubChem database
  - Fetches GHS (Globally Harmonized System) hazard classification
  - Returns: Safety pictograms, signal words, hazard statements, molecular properties
  - **WHEN TO USE**: "What are the hazards of toluene?", "Is DCM dangerous?", "PubChem safety for benzene"

- `compare_pubchem_safety()` - **COMPARE GHS HAZARDS** of multiple compounds (max 5)
  - Shows hazard pictograms and signal words for each compound
  - Provides contextual recommendation on which is safer
  - **WHEN TO USE**: "Compare safety of toluene vs benzene", "Which is safer: acetone or MEK?"

- `visualize_pubchem_safety()` - **HAZARD VISUALIZATION** from PubChem data
  - Creates stacked bar charts showing hazard categories per compound
  - **WHEN TO USE**: "Create a safety chart for common solvents", "Visualize PubChem hazards"

- `get_pubchem_toxicity()` - **🧫 TOXICITY & ENVIRONMENTAL DATA** (max 5 compounds)
  - Fetches LD50, LC50, biodegradation, aquatic toxicity data
  - **WHEN TO USE**: "What's the LD50 of benzene?", "Is acetone biodegradable?", "Compare environmental toxicity of DCM vs chloroform"

**NOTE:** PubChem tools query external API (requires internet). They provide authoritative GHS data distinct from the GSK G-score database.

### TEA/LCA TOOLS (Techno-Economic Analysis / Life Cycle Assessment):

- `analyze_solvent_recovery_tea()` - **💰 TECHNO-ECONOMIC ANALYSIS**
  - Calculates capital costs, operating costs, payback period
  - Parameters: solvent, polymer_throughput_kg_hr, solvent_to_polymer_ratio, recovery_fraction, process_temp_c
  - **WHEN TO USE**: "What's the cost of recovering toluene?", "TEA for LDPE separation", "Calculate payback period"

- `analyze_solvent_recovery_lca()` - **🌱 LIFE CYCLE ASSESSMENT**
  - Calculates CO2 emissions, energy consumption, environmental impact
  - **WHEN TO USE**: "What's the carbon footprint?", "LCA for solvent recovery", "How much CO2 does separation emit?"

- `compare_solvents_tea_lca()` - **📊 TEA/LCA COMPARISON**
  - Compares multiple solvents on cost AND environmental metrics
  - Returns rankings by cost, emissions, and overall
  - **WHEN TO USE**: "Compare toluene vs acetone for cost and emissions", "Which solvent is cheapest and greenest?"

**TEA/LCA VISUALIZATION TOOLS:**

- `generate_tea_visualizations()` - **📈 ALL TEA CHARTS**
  - Generates 6 visualizations: capital breakdown, operating costs, waterfall, cashflow, tornado, energy flow
  - **WHEN TO USE**: "Show TEA charts for toluene", "Visualize capital costs", "Show cost breakdown"

- `generate_lca_visualizations()` - **🌍 ALL LCA CHARTS**
  - Generates emissions breakdown pie chart and recovery vs baseline comparison
  - **WHEN TO USE**: "Show LCA charts for toluene", "Visualize emissions breakdown"

- `generate_solvent_comparison_visualization()` - **📊 COMPARISON CHART**
  - Creates grouped bar chart comparing cost vs emissions for multiple solvents
  - **WHEN TO USE**: "Compare solvents visually", "Plot cost vs emissions chart"

- `plot_tea_sensitivity_tornado()` - **🌀 TORNADO CHART**
  - Shows how ±20% parameter changes affect cost (sensitivity analysis)
  - **WHEN TO USE**: "Show sensitivity analysis", "What parameters most affect cost?"

- `plot_tea_cashflow()` - **💰 CASHFLOW DIAGRAM**
  - Shows cumulative cash position over 20-year project lifetime with payback point
  - **WHEN TO USE**: "Show cashflow diagram", "When does recovery break even?"

**NOTE:** TEA/LCA tools are powered by the standalone tea_lca_module.py which can be edited by TEA specialists.

### STRAP PROCESS TOOLS (Solvent-Targeted Recovery and Precipitation):

STRAP is an advanced multi-polymer recovery methodology for recycling plastic waste (e.g., multilayer films, biopharmaceutical single-use technologies). These tools provide comprehensive TEA and LCA aligned with published STRAP research.

- `analyze_strap_process()` - **🔬 FULL STRAP ANALYSIS**
  - Complete TEA/LCA for multi-polymer recovery from plastic waste
  - Auto-selects optimal solvents for each polymer from database
  - Returns: Capital costs, UOC, MSP, GWP, virgin comparison
  - Parameters: polymers (list), feedstock_composition (dict), capacity_mt_yr
  - **WHEN TO USE**: "Run STRAP analysis for PE/EVOH", "Full TEA/LCA for multilayer film recycling", "Analyze STRAP for PE, PET, EVOH"

- `calculate_strap_msp()` - **💵 MINIMUM SELLING PRICE**
  - Calculates break-even price where NPV=0 at target IRR
  - Compares to market prices for viability assessment
  - Parameters: polymers, feedstock_composition, capacity_mt_yr, target_irr (default 0.15)
  - **WHEN TO USE**: "What's the MSP for STRAP PE?", "Break-even price for EVOH recovery", "MSP at 15% IRR"

- `plot_strap_scale_economics()` - **📈 SCALE ECONOMICS CURVES**
  - Dual-axis plot: UOC ($/kg) and TCI ($M) vs plant capacity
  - Shows economies of scale and optimal capacity range
  - Parameters: polymers, feedstock_composition, capacity_range ("min-max")
  - **WHEN TO USE**: "Show scale economics for STRAP", "How does plant size affect costs?", "UOC vs capacity plot"

- `compare_strap_scenarios()` - **📊 SCENARIO COMPARISON**
  - Compare multiple STRAP configurations (different polymers, compositions, capacities)
  - Ranks by ROI, UOC, payback period
  - Parameters: scenario_configs (list of dicts with name, polymers, feedstock_composition, capacity_mt_yr)
  - **WHEN TO USE**: "Compare PE-only vs PE+EVOH", "Which STRAP config is most profitable?", "Rank scenarios by ROI"

- `generate_strap_visualizations()` - **📊 ALL STRAP CHARTS**
  - Creates: Scale economics plot, MSP sensitivity tornado, GWP comparison bars
  - Parameters: polymers, feedstock_composition, capacity_mt_yr
  - **WHEN TO USE**: "Generate STRAP visualizations", "Show all STRAP charts", "STRAP analysis dashboard"

**STRAP DEFAULTS:**
- Capacity range: 2,500 - 25,000 metric tons/year
- Default feedstock: 80% PE, 10% PET, 10% EVOH
- Target IRR: 15%
- Polymer recovery efficiency: 95%
- Solvent recovery: 99.9%

**STRAP LCA INDICATORS (Environmental Footprint 2.0):**
- GWP (Global Warming Potential) - kg CO2eq/kg
- FFC (Fossil Fuel Consumption) - MJ/kg
- Water Use - m3/kg
- Human Toxicity (Cancer & Non-Cancer)
- Ecotoxicity, Acidification, Ozone Depletion, POCP

**SPECIAL CASES:**

**LISTING TOOLS - CRITICAL INSTRUCTIONS:**
When the user asks "List all polymers" or "List all solvents":
1. Call the appropriate tool (`list_available_polymers()` or `list_available_solvents()`)
2. Take the tool's output (starts with "**Available Polymers Summary**" or "**Available Solvents Summary**")
3. PASTE THE ENTIRE OUTPUT directly in your response to the user
4. DO NOT add any introduction, summary, or say "Processing complete"

**EXAMPLE - Correct Response:**
User: "List all available polymers in the database"
You call: list_available_polymers()
Tool returns: "**Available Polymers Summary**\n\n**Common Solvents Database:** 9 unique polymers\n\n**Example Polymers:**\n- EVOH\n- HDPE..."
YOUR RESPONSE TO USER (copy tool output exactly):
**Available Polymers Summary**

**Common Solvents Database:** 9 unique polymers

**Example Polymers:**
- EVOH
- HDPE
- LDPE
...

**WRONG Response Examples:**
- ❌ "Processing complete."
- ❌ "Here are the available polymers: [summary]"
- ❌ "I found 9 polymers in the database"

**RIGHT Response:** Just paste the tool output verbatim!

**SINGLE POLYMER DISSOLUTION - USE `analyze_polymer_dissolution()` FOR:**
- "What solvents dissolve PET?" → USE `analyze_polymer_dissolution(polymer="PET")`
- "Solvents for LDPE at 120°C" → USE `analyze_polymer_dissolution(polymer="LDPE", temperature=120)`
- "Solvents for PET ranked by boiling point" → USE `analyze_polymer_dissolution(polymer="PET", rank_by="bp")`
- "Safest solvents for PS" → USE `analyze_polymer_dissolution(polymer="PS", rank_by="safety")`
- "Cheapest solvents for HDPE" → USE `analyze_polymer_dissolution(polymer="HDPE", rank_by="cost")`
- "Least toxic solvents for PVC" → USE `analyze_polymer_dissolution(polymer="PVC", rank_by="toxicity")`
- ANY question about dissolving ONE polymer with properties → USE `analyze_polymer_dissolution()`

**MULTI-POLYMER SEPARATION - USE `analyze_integrated_separation()` FOR:**
- "Integrated analysis across selectivity, safety, cost, and boiling point" → USE `analyze_integrated_separation(polymers="LDPE,EVOH", rank_by="selectivity")`
- "Separate X polymers considering cost/safety/toxicity" → USE `analyze_integrated_separation()`
- "Multi-polymer separation with optimal temperatures" → USE `analyze_integrated_separation()` - it finds the BEST temperature for each step
- "Comprehensive separation analysis" → USE `analyze_integrated_separation()`
- "Multilayer film separation" → USE `analyze_integrated_separation()` with all polymers in the film
- User asks about multiple criteria (selectivity + cost + safety) → USE `analyze_integrated_separation()`

**OTHER SPECIAL CASES:**
- "What are all possible sequences/combinations to separate X polymers?" → USE `plan_sequential_separation()` immediately
- "Enumerate separation strategies" → USE `plan_sequential_separation()` with create_decision_tree=True
- "How can I separate A, B, C, D?" → USE `plan_sequential_separation()` to show ALL permutations
- "Show me the 2nd/3rd best sequence" → USE `view_alternative_separation_sequence()` with sequence_rank parameter
- "Show me PET-first separation" → USE `view_alternative_separation_sequence()` with starting_polymer parameter
- "What if we start with LDPE instead?" → USE `view_alternative_separation_sequence()` with starting_polymer parameter
- "Plot boiling points for PET solvents" → USE `plot_solvent_properties()` with property_to_plot='bp'
- "Show energy costs for solvents" → USE `plot_solvent_properties()` with property_to_plot='energy'
- "Compare LogP values for X solvents" → USE `plot_solvent_properties()` with property_to_plot='logp'
- "Rank by cost/cheapest solvents" → USE `rank_solvents_by_property('energy', ascending=True)`
- "Least toxic solvents" → USE `rank_solvents_by_property('logp', ascending=True)` (negative LogP = less toxic)
- "Separation with cost/toxicity" → USE `analyze_separation_with_properties()` with rank_by parameter
- "What are the properties of X solvent?" → USE `get_solvent_properties('X')`
- "Predict solubility of X in Y using ML/machine learning" → USE `predict_solubility_ml('X', 'Y')`
- "Will HDPE dissolve in toluene?" → USE `predict_solubility_ml('HDPE', 'Toluene')`
- "Hansen parameters prediction for X and Y" → USE `predict_solubility_ml('X', 'Y')`
- "ML prediction" or "machine learning prediction" → USE `predict_solubility_ml()` with specified polymers/solvents

**SOLVENT PROPERTY INTERPRETATION:**
- **LogP**: Lower/negative = less toxic, more water soluble. Higher = more toxic, more lipophilic
- **Energy (J/g)**: Lower = cheaper to use (less energy for heating/recovery)
- **Boiling Point**: Lower = easier solvent recovery, but may need pressure vessels
- **Cp (Heat Capacity)**: Higher = more energy needed to heat

### RAG TOOLS (Literature Knowledge Base Search):

You have access to indexed scientific literature through RAG (Retrieval-Augmented Generation) knowledge bases. Use these tools to answer questions about topics covered in the literature, such as:
- **Deinking/printed plastics recycling** (surfactants, binders, ink removal, flexographic printing)
- **STRAP recycling methodology** (solvent-targeted recovery)
- **Any topic covered in the indexed papers**

**RAG Tools:**
- `rag_search()` - **🔍 SEARCH LITERATURE** - Query the indexed knowledge base for relevant information
  - **WHEN TO USE**: User asks about deinking, surfactants, binders, ink removal, printed plastics, or any topic that might be covered in literature
  - **IMPORTANT**: If user asks a question about a topic (deinking, binders, surfactants, etc.) and database tools don't have the answer, USE RAG!

- `rag_status()` - Check which knowledge bases are available and their contents
- `switch_rag_kb()` - Switch between different knowledge bases (e.g., STRAP-CORE, printed_plastics_deinking)

**RAG USAGE GUIDELINES:**
1. **Follow-up questions**: If user asks a follow-up about topics found in RAG (e.g., "What binders have been tested?"), USE RAG AGAIN
2. **Topic coverage**: RAG covers topics NOT in the database: deinking, surfactants, binders, ink removal, multilayer packaging recycling, printed plastics
3. **Don't refuse RAG topics**: If the question is about deinking, binders, surfactants, or other RAG topics, SEARCH RAG - don't say "I cannot answer"

Remember: ACCURACY > SPEED. ACTION > EXPLANATION. Always verify before reporting. Let the adaptive tools do their job - they will find results if any exist."""


# ============================================================
# PATCHED: Robust Agent State and Nodes
# ============================================================

class AgentState(MessagesState):
    """Enhanced state - defaults handled in functions."""
    iteration_count: int
    max_iterations: int
    # Memory Engine fields
    user_id: Optional[str] = None
    memory_context: Optional[str] = None
    memory_enabled: bool = True
    # Router fields (for dynamic tool selection)
    # NOTE: Only store category names (strings), not tool objects (not serializable by checkpointer)
    selected_categories: Optional[List[str]] = None


class AsyncToolNode:
    """
    Custom async ToolNode with parallel execution and comprehensive error handling.

    Executes multiple tool calls concurrently for improved performance.
    Handles both async and sync tools automatically.
    """

    def __init__(self, tools):
        """Initialize with list of tools."""
        self.tools_by_name = {tool.name: tool for tool in tools}
        logger.info(f"AsyncToolNode initialized with {len(tools)} tools")

    async def __call__(self, state):
        """Execute tools in parallel when possible."""
        try:
            # Extract messages from state
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}

            last_message = messages[-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return {"messages": []}

            tool_calls = last_message.tool_calls

            # LIMIT parallel tool calls to prevent overload
            MAX_PARALLEL_TOOLS = 10
            if len(tool_calls) > MAX_PARALLEL_TOOLS:
                logger.warning(f"Too many tool calls ({len(tool_calls)}), limiting to {MAX_PARALLEL_TOOLS}")
                tool_calls = tool_calls[:MAX_PARALLEL_TOOLS]

            async def execute_tool_call(tool_call):
                """Execute single tool call (async or sync)."""
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})
                tool_call_id = tool_call.get('id', 'unknown')

                if tool_name not in self.tools_by_name:
                    return ToolMessage(
                        content=f"❌ Error: Tool '{tool_name}' not found. Available tools: {', '.join(list(self.tools_by_name.keys())[:5])}...",
                        tool_call_id=tool_call_id
                    )

                try:
                    tool = self.tools_by_name[tool_name]
                    logger.info(f"Executing tool: {tool_name}")

                    # Use .ainvoke() for proper async tool execution
                    # This handles both sync and async tools correctly through LangChain's decorator
                    try:
                        # Try async invocation first (works for both async and sync tools)
                        result = await tool.ainvoke(tool_args)
                    except AttributeError:
                        # Fallback to sync invocation if ainvoke not available
                        result = await run_in_thread(tool.invoke, tool_args)

                    # Truncate long outputs
                    if len(str(result)) > MAX_TOOL_OUTPUT_LENGTH:
                        result = truncate_output(str(result))

                    return ToolMessage(content=str(result), tool_call_id=tool_call_id)

                except Exception as e:
                    logger.error(f"Tool '{tool_name}' error: {e}")
                    error_msg = f"**Tool Error ({tool_name}):** {str(e)[:500]}\n\nTry verifying inputs with `describe_table()` or `check_column_values()`."
                    return ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id
                    )

            # PARALLEL EXECUTION of all tool calls
            tool_messages = await asyncio.gather(*[execute_tool_call(tc) for tc in tool_calls])

            # Periodic cleanup
            gc.collect()
            return {"messages": tool_messages}

        except Exception as e:
            logger.error(f"AsyncToolNode error: {e}\n{traceback.format_exc()}")
            return {
                "messages": [ToolMessage(
                    content=f"**System Error:** {str(e)[:500]}",
                    tool_call_id="error"
                )]
            }


async def sql_agent_node(state: AgentState):
    """Robust agent node with comprehensive error handling (ASYNC)."""

    # Safely get state values with defaults
    current_iter = state.get("iteration_count") or 0
    max_iter = state.get("max_iterations") or MAX_ITERATIONS
    
    # CRITICAL: Ensure messages is always a list
    raw_messages = state.get("messages")
    
    # Debug logging
    logger.debug(f"sql_agent_node called - messages type: {type(raw_messages)}")
    
    # Handle various message states
    if raw_messages is None:
        messages = []
    elif isinstance(raw_messages, str):
        # If somehow messages became a string, wrap it
        logger.warning(f"Messages was a string, wrapping: {raw_messages[:100]}")
        messages = [HumanMessage(content=raw_messages)]
    elif isinstance(raw_messages, list):
        messages = raw_messages
    else:
        # Try to convert to list
        try:
            messages = list(raw_messages)
        except Exception as e:
            logger.error(f"Could not convert messages to list: {e}")
            messages = []
    
    # Handle empty messages
    if not messages:
        return {
            "messages": [AIMessage(content="I didn't receive any input. How can I help you analyze polymer solubility data?")],
            "iteration_count": current_iter + 1,
            "max_iterations": max_iter
        }
    
    # Trim old messages to prevent memory bloat
    if len(messages) > MAX_MESSAGE_HISTORY:
        messages = messages[-MAX_MESSAGE_HISTORY:]
    
    # GEMINI FIX: Sanitize message history to ensure proper ordering
    # Gemini requires: User → AI (with tool_calls) → ToolMessages → AI/User...
    def sanitize_messages_for_gemini(msgs):
        """
        Ensure message history follows Gemini's required ordering.
        Uses a two-pass approach:
        1. First pass: collect valid tool_call_ids from all AIMessages
        2. Second pass: build sanitized list with proper ordering
        """
        if not msgs:
            return msgs

        # Pass 1: Collect all valid tool_call_ids from AIMessages
        valid_tool_call_ids = set()
        for msg in msgs:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get('id'):
                        valid_tool_call_ids.add(tc.get('id'))

        # Pass 2: Build sanitized list
        sanitized = []

        for msg in msgs:
            # HumanMessage: Always include
            if isinstance(msg, HumanMessage):
                sanitized.append(msg)
                continue

            # AIMessage with tool_calls
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Need to ensure it follows HumanMessage or ToolMessage
                    if not sanitized:
                        # First message - insert a context-setting HumanMessage
                        sanitized.append(HumanMessage(content="Analyze the following request."))
                        sanitized.append(msg)
                    else:
                        last = sanitized[-1]
                        if isinstance(last, (HumanMessage, ToolMessage)):
                            # Valid position
                            sanitized.append(msg)
                        elif isinstance(last, AIMessage):
                            # AI followed by AI with tools - insert bridge HumanMessage
                            sanitized.append(HumanMessage(content="Continue."))
                            sanitized.append(msg)
                        else:
                            sanitized.append(msg)
                else:
                    # Regular AI message without tool_calls - always ok
                    sanitized.append(msg)
                continue

            # ToolMessage: Only include if matching tool_call_id exists
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id in valid_tool_call_ids:
                    # Ensure there's a preceding AIMessage with tool_calls
                    # Find the matching AIMessage in sanitized
                    has_matching = False
                    for prev in reversed(sanitized):
                        if isinstance(prev, AIMessage) and hasattr(prev, 'tool_calls') and prev.tool_calls:
                            for tc in prev.tool_calls:
                                if tc.get('id') == msg.tool_call_id:
                                    has_matching = True
                                    break
                            if has_matching:
                                break

                    if has_matching:
                        sanitized.append(msg)
                    # If no matching AIMessage in sanitized yet, skip (it was filtered earlier)
                continue

            # Other message types - include
            sanitized.append(msg)

        # Final fix: Ensure messages end with user role
        if sanitized and isinstance(sanitized[-1], AIMessage):
            last_ai = sanitized[-1]
            if not (hasattr(last_ai, 'tool_calls') and last_ai.tool_calls):
                # Regular AI response at end - add continuation
                sanitized.append(HumanMessage(content="Continue with the analysis."))

        return sanitized
    
    # Apply sanitization
    messages = sanitize_messages_for_gemini(messages)

    # Ensure each message in the list is valid and has content
    def has_valid_content(msg):
        """Check if message has valid content for Gemini."""
        if msg is None:
            return False
        # AIMessage with tool_calls is valid even without text content
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            return True
        # ToolMessage is valid with any content (including empty for some tools)
        if isinstance(msg, ToolMessage):
            return True
        # HumanMessage and regular AIMessage need non-empty content
        content = getattr(msg, 'content', None)
        if content is None:
            return False
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and len(content) > 0:
            return True
        return False

    valid_messages = [msg for msg in messages if has_valid_content(msg)]

    # If no valid messages, ensure we have at least a human message
    if not valid_messages or not any(isinstance(m, HumanMessage) for m in valid_messages):
        # Find original human message from state
        original_query = state.get("original_query", "")
        if not original_query:
            # Try to find from messages
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) and getattr(msg, 'content', ''):
                    original_query = msg.content
                    break
        if not original_query:
            original_query = "Continue the analysis based on previous tool results."
        valid_messages = [HumanMessage(content=original_query)]

    try:
        # Get model from config if specified, otherwise use default
        model_name = state.get("configurable", {}).get("model") or DEFAULT_MODEL
        current_llm = create_llm(model_name)

        # Get router-selected categories and reconstruct tools
        # NOTE: We store only category names in state (strings are serializable)
        # and reconstruct tool objects here
        selected_categories = state.get("selected_categories") or []

        if selected_categories:
            # Reconstruct tools from categories
            tools_to_bind = get_tools_for_categories(set(selected_categories))
            logger.info(f"Agent using {len(tools_to_bind)} tools from categories: {selected_categories}")
        else:
            # Fallback to all tools if no categories selected
            tools_to_bind = SQL_AGENT_TOOLS
            logger.info(f"Agent using all {len(tools_to_bind)} tools (no router categories)")

        sql_llm = current_llm.bind_tools(tools_to_bind)

        # Ensure SQL_AGENT_PROMPT is a string
        prompt = SQL_AGENT_PROMPT if isinstance(SQL_AGENT_PROMPT, str) else str(SQL_AGENT_PROMPT)

        # Inject memory context if available
        memory_context = state.get("memory_context", "")
        if memory_context and state.get("memory_enabled", True):
            prompt = prompt + "\n\n" + memory_context
            logger.debug(f"Injected memory context ({len(memory_context)} chars)")

        # Build full messages list carefully
        system_msg = SystemMessage(content=prompt)
        full_messages = [system_msg] + valid_messages
        
        logger.debug(f"Invoking LLM with {len(full_messages)} messages")
        response = await sql_llm.ainvoke(full_messages)

        return {
            "messages": [response],
            "iteration_count": current_iter + 1,
            "max_iterations": max_iter
        }
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
        
        # Special handling for Gemini function call ordering errors
        if "function call turn" in error_str.lower() or "INVALID_ARGUMENT" in error_str:
            # Clear conversation and retry with just the last human message
            last_human = None
            for msg in reversed(valid_messages):
                if isinstance(msg, HumanMessage):
                    last_human = msg
                    break
            
            if last_human:
                try:
                    logger.info("Retrying with cleaned message history...")
                    sql_llm = llm.bind_tools(SQL_AGENT_TOOLS)
                    prompt = SQL_AGENT_PROMPT if isinstance(SQL_AGENT_PROMPT, str) else str(SQL_AGENT_PROMPT)
                    clean_messages = [SystemMessage(content=prompt), last_human]
                    response = await sql_llm.ainvoke(clean_messages)

                    return {
                        "messages": [response],
                        "iteration_count": current_iter + 1,
                        "max_iterations": max_iter
                    }
                except Exception as retry_e:
                    logger.error(f"Retry also failed: {retry_e}")
        
        error_msg = AIMessage(content=(
            f"I encountered an error: {str(e)[:300]}\n\n"
            f"**Try:**\n"
            f"1. Ask: 'What tables are available?'\n"
            f"2. Verify data exists for your query\n"
            f"3. Use simpler queries first"
        ))
        
        return {
            "messages": [error_msg],
            "iteration_count": current_iter + 1,
            "max_iterations": max_iter
        }


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Safe continuation check with error detection to prevent endless loops."""

    max_iter = state.get("max_iterations") or MAX_ITERATIONS
    current_iter = state.get("iteration_count") or 0

    if current_iter >= max_iter:
        logger.warning(f"Max iterations ({max_iter}) reached")
        return "end"

    # Safely get messages
    raw_messages = state.get("messages")

    # Handle various message states
    if raw_messages is None:
        return "end"
    elif isinstance(raw_messages, str):
        logger.warning("Messages was a string in should_continue")
        return "end"
    elif not isinstance(raw_messages, list):
        try:
            messages = list(raw_messages)
        except:
            return "end"
    else:
        messages = raw_messages

    if not messages:
        return "end"

    # Check for repeated errors - stop if last 3 tool messages are all errors
    recent_tool_msgs = []
    for msg in reversed(messages[-10:]):  # Check last 10 messages
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if 'ToolMessage' in str(type(msg)) or 'tool' in str(type(msg)).lower():
                recent_tool_msgs.append(msg.content)
                if len(recent_tool_msgs) >= 3:
                    break

    if len(recent_tool_msgs) >= 3:
        error_count = sum(1 for m in recent_tool_msgs if 'ERROR' in m or 'Error' in m or 'error' in m or 'validation error' in m.lower())
        if error_count >= 3:
            logger.warning(f"Stopping: {error_count} consecutive tool errors detected")
            return "end"

    try:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
    except (IndexError, TypeError, AttributeError) as e:
        logger.debug(f"should_continue check failed: {e}")
        return "end"

    return "end"


# Build async agent graph with ROUTER for intelligent tool selection
builder = StateGraph(AgentState)

# Add router node (runs first, selects relevant tools based on query)
builder.add_node("router", router_node)

# Add agent node (uses router-selected tools)
builder.add_node("agent", sql_agent_node)

# Add tool node (keeps ALL tools - can execute any tool the LLM calls)
# This ensures robustness even if router misses a category
builder.add_node("tools", AsyncToolNode(SQL_AGENT_TOOLS))

# Graph flow: START → router → agent ↔ tools → END
builder.add_edge(START, "router")
builder.add_edge("router", "agent")
builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
agent_graph = builder.compile(checkpointer=checkpointer)

logger.info("✅ Async SQL Agent System compiled successfully!")
logger.info(f"SQL Agent: {len(SQL_AGENT_TOOLS)} total tools, {len(TOOL_CATEGORIES)} categories")
logger.info("Router: Intelligent tool selection reduces LLM context by 60-70% per query")
logger.info("Performance: Router + async parallel execution = faster responses")

# ============================================================
# Utility Functions for External Integration
# ============================================================

def create_thread_id():
    """Create new thread ID for conversation."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

# Create default config on module load
config = create_thread_id()

# ============================================================
# Module Initialization Complete
# ============================================================

logger.info("\n" + "="*70)
logger.info("🧪 POLYMER SOLUBILITY ANALYSIS AGENT - CORE MODULE LOADED")
logger.info("="*70)
logger.info(f"📊 SQL Tables: {len(sql_db.table_schemas)}")
logger.info(f"🔧 Agent Tools: {len(SQL_AGENT_TOOLS)}")
logger.info(f"🛡️ Features: Memory-efficient + Error-handling + Adaptive Analysis")
logger.info(f"📁 Data Directory: {DATA_DIR}")
logger.info(f"📊 Plots Directory: {PLOTS_DIR}")
logger.info("="*70)
logger.info("✅ Agent module ready for import by FastAPI/other frameworks")
logger.info("="*70 + "\n")

# ============================================================
# MULTI-AGENT SYSTEM INTEGRATION
# ============================================================

try:
    from multi_agent_system import (
        build_multi_agent_graph,
        enhanced_complexity_router,
        RoutingDecision,
        MultiAgentState,
        initialize_tool_subsets,
        SEPARATION_PLANNER_PROMPT,
        TEA_LCA_ANALYST_PROMPT,
        LITERATURE_RESEARCHER_PROMPT,
    )

    # Initialize tool subsets for specialist agents
    initialize_tool_subsets(TOOL_CATEGORIES, SQL_AGENT_TOOLS)

    def create_llm_with_tools(tools: list, system_prompt: str = None):
        """Factory function to create LLM with tools bound."""
        prompt = system_prompt or SQL_AGENT_SYSTEM_PROMPT
        return llm.bind_tools(tools)

    # Build multi-agent graph
    multi_agent_graph = build_multi_agent_graph(
        sql_agent_node=sql_agent_node,
        async_tool_node_class=AsyncToolNode,
        all_tools=SQL_AGENT_TOOLS,
        tool_categories=TOOL_CATEGORIES,
        llm_factory=create_llm_with_tools
    )

    # Flag to indicate multi-agent is available
    MULTI_AGENT_AVAILABLE = True

    logger.info("="*70)
    logger.info("🤖 MULTI-AGENT SYSTEM LOADED")
    logger.info("="*70)
    logger.info("  Paths: fast (simple) | standard (moderate) | specialist (complex)")
    logger.info("  Specialists: separation | tea_lca | literature")
    logger.info("  Use: multi_agent_graph instead of agent_graph for enhanced routing")
    logger.info("="*70 + "\n")

except ImportError as e:
    logger.warning(f"Multi-agent system not available: {e}")
    multi_agent_graph = None
    MULTI_AGENT_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to initialize multi-agent system: {e}")
    multi_agent_graph = None
    MULTI_AGENT_AVAILABLE = False


def get_routing_info(query: str) -> dict:
    """
    Get routing information for a query without executing it.

    Useful for frontend to show which path will be taken.

    Returns:
        dict with complexity, path, specialist, reason
    """
    if not MULTI_AGENT_AVAILABLE:
        return {
            "complexity": 3,
            "path": "standard",
            "specialist": None,
            "reason": "Multi-agent not available",
            "multi_agent_active": False
        }

    decision = enhanced_complexity_router(query)
    return {
        "complexity": decision.complexity,
        "path": decision.path,
        "specialist": decision.specialist,
        "reason": decision.reason,
        "multi_agent_active": decision.path == "specialist"
    }
