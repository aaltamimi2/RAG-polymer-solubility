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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
)

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

from langchain_core.tools import tool

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "./data"
SQL_DB_PATH = ":memory:"
PLOTS_DIR = "./plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Memory management constants
MAX_ITERATIONS = 50
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

def cleanup_old_plots(keep_latest: int = MAX_PLOTS_TO_KEEP):
    """Remove old plots to free disk space."""
    try:
        plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
        plots += glob.glob(os.path.join(PLOTS_DIR, "*.html"))
        
        if len(plots) <= keep_latest:
            return 0
        
        plots.sort(key=os.path.getmtime)
        removed = 0
        
        for filepath in plots[:-keep_latest]:
            try:
                os.remove(filepath)
                removed += 1
            except OSError:
                pass
        
        gc.collect()
        logger.info(f"Cleaned up {removed} old plot files")
        return removed
    except Exception as e:
        logger.warning(f"Error cleaning up plots: {e}")
        return 0


def truncate_output(text: str, max_length: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
    """Truncate tool output to prevent memory issues."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 50
    return text[:half] + f"\n\n... [TRUNCATED {len(text) - max_length} chars] ...\n\n" + text[-half:]


def safe_tool_wrapper(func):
    """Decorator for safe tool execution with error handling and memory cleanup."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            if result is None:
                return "Operation completed (no output)."
            
            result_str = str(result)
            
            # Truncate if too long
            if len(result_str) > MAX_TOOL_OUTPUT_LENGTH:
                result_str = truncate_output(result_str)
            
            return result_str
            
        except Exception as e:
            logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
            return (
                f"**❌ Error in {func.__name__}:**\n"
                f"```\n{str(e)[:500]}\n```\n\n"
                f"**Suggestions:**\n"
                f"- Verify input parameters with `describe_table()`\n"
                f"- Check values with `check_column_values()`\n"
                f"- Use `verify_data_accuracy()` to confirm data exists"
            )
        finally:
            gc.collect()
    
    return wrapper


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
# Data Validator Class (PATCHED with caching)
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
# Enhanced SQLDatabase (PATCHED)
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
def query_database(sql_query: str) -> str:
    """Execute a SQL query with enhanced validation and error reporting."""
    result = sql_db.execute_query(sql_query)

    if result["success"]:
        output = f"**Query Results**\n\nQuery: `{result['query']}`\n\nRows returned: {result['rows']}\n\n"
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
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: str,
    start_temperature: float = 25.0,
    initial_selectivity: float = 30.0
) -> str:
    """Find optimal conditions to separate target polymer from comparison polymers.

    Note: Selectivity is in percentage points (0-100 scale). A selectivity of 30 means
    the target polymer has 30% higher solubility than the max competing polymer.
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

    return "\n".join(output)


@tool
@safe_tool_wrapper
def adaptive_threshold_search(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: Optional[str] = None,
    temperature: float = 25.0,
    start_threshold: float = 0.5
) -> str:
    """Search for selective solvents using adaptive thresholds."""
    
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
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: Optional[str] = None,
    temperature_range: str = "25-120",
    auto_threshold: bool = True
) -> str:
    """Enhanced selective solubility analysis with adaptive thresholds."""
    temp_min, temp_max = map(float, temperature_range.split('-'))

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

    return "\n".join(output)


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
def compare_groups_statistically(
    table_name: str,
    value_column: str,
    group_column: str,
    group1: str,
    group2: str,
    filters: Optional[str] = None
) -> str:
    """Statistical comparison between two groups with hypothesis testing."""
    where_clause = f"WHERE {filters} AND" if filters else "WHERE"

    query1 = f"SELECT {value_column} FROM {table_name} {where_clause} LOWER({group_column}) = LOWER('{group1}')"
    query2 = f"SELECT {value_column} FROM {table_name} {where_clause} LOWER({group_column}) = LOWER('{group2}')"

    result1 = sql_db.execute_query(query1, limit=100000)
    result2 = sql_db.execute_query(query2, limit=100000)

    if not result1["success"] or not result2["success"]:
        return f"❌ Query failed: {result1.get('error', result2.get('error'))}"

    data1 = result1["dataframe"][value_column].dropna()
    data2 = result2["dataframe"][value_column].dropna()

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
    include_confidence_bands: bool = True
) -> str:
    """Create temperature vs solubility curves with validation and confidence bands."""
    polymer_list = [p.strip() for p in polymers.split(',')]
    solvent_list = [s.strip() for s in solvents.split(',')]

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

    query = f"""
    SELECT {polymer_column}, {solvent_column}, {temperature_column},
           AVG({solubility_column}) as avg_sol,
           STDDEV({solubility_column}) as std_sol,
           COUNT(*) as n
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {solvent_column} IN ('{solvent_filter}')
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
    plt.tight_layout()

    filepath = save_plot(fig, "solubility_temp_curve", "matplotlib")

    output = f"✅ **Solubility vs Temperature Plot Created**\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    output += f"Data points: {result['rows']}\n"
    if include_confidence_bands:
        output += "Shaded regions show 95% confidence intervals\n"
    output += f"\n{get_plot_url(filepath)}"

    del df
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
    temperature: float = 25.0,
    temperature_tolerance: float = 5.0
) -> str:
    """Create heatmap showing solubility across all polymer-solvent combinations."""
    query = f"""
    SELECT {polymer_column}, {solvent_column},
           AVG({solubility_column}) as avg_solubility
    FROM {table_name}
    WHERE {temperature_column} BETWEEN {temperature - temperature_tolerance}
          AND {temperature + temperature_tolerance}
    GROUP BY {polymer_column}, {solvent_column}
    ORDER BY {polymer_column}, {solvent_column}
    """

    result = sql_db.execute_query(query, limit=10000)
    if not result["success"]:
        return f"❌ Query failed: {result.get('error')}"

    df = result["dataframe"]
    pivot_df = df.pivot(index=polymer_column, columns=solvent_column, values='avg_solubility')

    # Determine if we should show annotations based on size
    n_cells = pivot_df.shape[0] * pivot_df.shape[1]
    show_annot = n_cells <= 100  # Only annotate if not too many cells
    annot_fontsize = 10 if n_cells <= 30 else 8 if n_cells <= 60 else 6

    fig, axes = plt.subplots(1, 2 if target_polymer else 1,
                            figsize=(20 if target_polymer else 14, 10))

    if target_polymer:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes]

    # Main heatmap with improved formatting
    sns.heatmap(pivot_df, annot=show_annot, fmt='.1f', cmap='YlGnBu',
               cbar_kws={'label': 'Solubility (%)', 'shrink': 0.8},
               linewidths=0.5, ax=axes[0], annot_kws={'size': annot_fontsize})
    axes[0].set_title(f'Solubility Heatmap ({temperature}°C ± {temperature_tolerance}°C)',
                     fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('Solvent', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Polymer', fontsize=14, fontweight='bold')
    # Rotate x-axis labels for readability
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=11)

    if target_polymer and target_polymer in pivot_df.index:
        target_row = pivot_df.loc[target_polymer]
        selectivity_df = pd.DataFrame()

        for col in pivot_df.columns:
            selectivity_df[col] = target_row[col] - pivot_df[col]

        selectivity_df.index = [f"{p} vs {target_polymer}" for p in pivot_df.index]

        sns.heatmap(selectivity_df, annot=show_annot, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Selectivity (%)', 'shrink': 0.8},
                   linewidths=0.5, ax=axes[1], annot_kws={'size': annot_fontsize})
        axes[1].set_title(f'Selectivity for {target_polymer}',
                         fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel('Solvent', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=10)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=11)

    plt.tight_layout()
    filepath = save_plot(fig, "selectivity_heatmap", "matplotlib")

    output = f"✅ **Heatmap Created**\n\n"
    output += f"Temperature: {temperature}°C ± {temperature_tolerance}°C\n"
    output += f"Polymers: {len(pivot_df)}\n"
    output += f"Solvents: {len(pivot_df.columns)}\n"

    if target_polymer and target_polymer in pivot_df.index:
        output += f"\n**Selectivity Summary for {target_polymer}:**\n"
        for solvent in list(pivot_df.columns)[:10]:
            target_sol = pivot_df.loc[target_polymer, solvent]
            max_other = pivot_df.drop(target_polymer)[solvent].max()
            selectivity = target_sol - max_other
            symbol = "✅" if selectivity > 10 else "⚠️" if selectivity > 0 else "❌"
            output += f"  {symbol} {solvent}: selectivity = {selectivity:.4f}\n"

    output += f"\n{get_plot_url(filepath)}"

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


# ============================================================
# Collect all tools
# ============================================================

@tool
@safe_tool_wrapper
def plan_sequential_separation(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    top_k_solvents: int = 5,
    temperature: float = 25.0,
    create_decision_tree: bool = True
) -> str:
    """
    Plan all possible sequential separation sequences for multiple polymers.
    
    Enumerates all n! permutations and finds top-k solvents for each separation step.
    Once a polymer is removed, it's no longer considered in subsequent steps.
    Optionally creates a decision tree visualization.
    
    Args:
        table_name: Database table name
        polymer_column: Column containing polymer names
        solvent_column: Column containing solvent names
        temperature_column: Column containing temperature values
        solubility_column: Column containing solubility values
        polymers: Comma-separated list of polymers to separate (e.g., "PP,PET,LDPE,PVC")
        top_k_solvents: Number of top solvents to report per separation step (default: 5)
        temperature: Target temperature in °C (default: 25.0)
        create_decision_tree: Whether to create a decision tree plot (default: True)
    
    Returns:
        Comprehensive separation plan with all sequences, top solvents, and decision tree
    """
    from itertools import permutations
    
    # Parse polymers
    polymer_list = [p.strip() for p in polymers.split(',') if p.strip()]
    n_polymers = len(polymer_list)
    
    if n_polymers < 2:
        return "Error: Need at least 2 polymers for separation planning."
    
    if n_polymers > 6:
        return f"Error: Too many polymers ({n_polymers}). Maximum 6 for computational feasibility ({6}! = 720 sequences)."
    
    # Generate all permutations
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
    
    # Function to find top-k solvents for separating target from remaining polymers
    def find_top_solvents(target: str, remaining: list, k: int = 5) -> list:
        """Find top-k solvents for separating target from remaining polymers."""
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
            df = sql_db.conn.execute(query).fetchdf()
        except Exception as e:
            return [{"solvent": "Error", "selectivity": 0, "error": str(e)}]
        
        if len(df) == 0:
            return [{"solvent": "No data", "selectivity": 0}]
        
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
        
        # Add solvent properties if available
        solvent_table = get_solvent_table_name()
        if solvent_table and results:
            try:
                # Use exact matching for solvent properties
                solvent_names = [r["solvent"] for r in results[:k]]
                prop_lookup = lookup_solvent_properties(solvent_names, solvent_table)

                # Add properties to results
                for r in results:
                    if r["solvent"] in prop_lookup:
                        r.update({k: v for k, v in prop_lookup[r["solvent"]].items() if v is not None})
            except Exception as e:
                logger.debug(f"Could not fetch solvent properties: {e}")
        
        return results[:k] if results else [{"solvent": "None found", "selectivity": 0}]
    
    # Analyze each sequence
    output.append("## Detailed Analysis of Each Sequence\n")
    
    sequence_scores = []  # Track overall scores for ranking
    sequence_details = []  # Store details for decision tree
    
    for seq_idx, sequence in enumerate(all_sequences, 1):
        output.append(f"### Sequence {seq_idx}: {' → '.join(sequence)}\n")
        
        remaining = list(sequence)
        total_min_selectivity = float('inf')
        seq_steps = []
        
        for step, target in enumerate(sequence[:-1], 1):  # Last polymer doesn't need separation
            remaining.remove(target)
            
            output.append(f"**Step {step}: Separate {target} from {{{', '.join(remaining)}}}**")
            
            top_solvents = find_top_solvents(target, remaining, top_k_solvents)
            
            step_data = {
                "step": step,
                "target": target,
                "remaining": remaining.copy(),
                "solvents": top_solvents
            }
            seq_steps.append(step_data)
            
            for rank, sol_info in enumerate(top_solvents, 1):
                if "error" in sol_info:
                    output.append(f"  {rank}. Error: {sol_info['error']}")
                elif sol_info["solvent"] == "No data":
                    output.append(f"  {rank}. No data available")
                else:
                    symbol = "✅" if sol_info["selectivity"] > 10 else "⚠️" if sol_info["selectivity"] > 0 else "❌"
                    line = f"  {rank}. {symbol} **{sol_info['solvent']}**: selectivity={sol_info['selectivity']:.1f}% (target={sol_info['target_sol']:.1f}%, max_other={sol_info['max_other']:.1f}%)"
                    
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
                    
                    output.append(line)
            
            # Track minimum selectivity across best solvents
            if top_solvents and "selectivity" in top_solvents[0]:
                best_selectivity = top_solvents[0]["selectivity"]
                total_min_selectivity = min(total_min_selectivity, best_selectivity)
            
            output.append("")
        
        # Final polymer
        output.append(f"**Step {len(sequence)}: {sequence[-1]} is isolated** ✅\n")
        
        sequence_scores.append({
            "sequence": sequence,
            "min_selectivity": total_min_selectivity,
            "steps": seq_steps
        })
        sequence_details.append(seq_steps)
        
        output.append("---\n")
    
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
    
    # Create decision tree visualization - CLEAN FLOWCHART STYLE
    if create_decision_tree and sequence_scores:
        output.append("## Decision Tree Visualization\n")

        try:
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

            # Build solvent lookup from sequence_details
            solvent_lookup = {}
            for seq_data in sequence_scores:
                for step in seq_data["steps"]:
                    key = (step["target"], tuple(sorted(step["remaining"])))
                    if key not in solvent_lookup and step["solvents"]:
                        solvent_lookup[key] = step["solvents"][0]

            # Create figure - side by side for each starting polymer
            n_trees = n_polymers
            fig_width = max(6 * n_trees, 12)
            fig_height = 8
            fig, axes = plt.subplots(1, n_trees, figsize=(fig_width, fig_height))
            if n_trees == 1:
                axes = [axes]

            fig.suptitle(f'Sequential Separation Decision Trees\nPolymers: {", ".join(polymer_list)} | Temperature: {temperature}°C',
                        fontsize=14, fontweight='bold', y=0.98)

            for ax_idx, first_polymer in enumerate(polymer_list):
                ax = axes[ax_idx]
                ax.set_xlim(-1, 3)
                ax.set_ylim(-0.5, 5)
                ax.axis('off')
                ax.set_title(f'Start with {first_polymer}', fontsize=11, fontweight='bold', pad=8)

                remaining_after_first = [p for p in polymer_list if p != first_polymer]

                # Level 0: Start node
                ax.scatter(1, 4.5, s=600, c='#3498db', zorder=10, edgecolors='black', linewidth=2)
                ax.text(1, 4.5, 'Mix', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

                # Level 1: Extract first polymer
                key1 = (first_polymer, tuple(sorted(remaining_after_first)))
                sol1 = solvent_lookup.get(key1, {"solvent": "N/A", "selectivity": 0})
                sel1 = sol1.get("selectivity", 0)
                color1 = get_color(sel1)

                # Arrow from start to first extraction
                ax.annotate('', xy=(1, 3.7), xytext=(1, 4.2),
                           arrowprops=dict(arrowstyle='->', color=color1, linewidth=3))

                # Solvent label on arrow
                sol_name1 = sol1.get('solvent', 'N/A')
                ax.text(1.7, 3.95, f"{sol_name1}\n(sel: {sel1:.2f})", fontsize=9, ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color1, linewidth=1.5))

                # First polymer node
                ax.scatter(1, 3.3, s=600, c=color1, zorder=10, edgecolors='black', linewidth=2)
                ax.text(1, 3.3, first_polymer, ha='center', va='center', fontsize=10, fontweight='bold')

                # Remaining label
                ax.text(-0.3, 3.3, f'Remaining:\n{", ".join(remaining_after_first)}',
                       ha='right', va='center', fontsize=8, color='#7f8c8d', style='italic')

                # Level 2 and beyond
                if len(remaining_after_first) == 1:
                    # Only one polymer left - it's isolated
                    last_poly = remaining_after_first[0]
                    ax.annotate('', xy=(1, 1.9), xytext=(1, 2.9),
                               arrowprops=dict(arrowstyle='->', color='#2ecc71', linewidth=3))
                    ax.scatter(1, 1.5, s=600, c='#2ecc71', zorder=10, edgecolors='black', linewidth=2, marker='s')
                    ax.text(1, 1.5, last_poly, ha='center', va='center', fontsize=10, fontweight='bold')
                    ax.text(1, 0.9, '✓ Isolated', ha='center', va='top', fontsize=9, color='#27ae60', fontweight='bold')

                elif len(remaining_after_first) >= 2:
                    # Need to separate second polymer
                    second_polymer = remaining_after_first[0]
                    remaining_after_second = remaining_after_first[1:]

                    key2 = (second_polymer, tuple(sorted(remaining_after_second)))
                    sol2 = solvent_lookup.get(key2, {"solvent": "N/A", "selectivity": 0})
                    sel2 = sol2.get("selectivity", 0)
                    color2 = get_color(sel2)

                    # Arrow to second extraction
                    ax.annotate('', xy=(1, 2.0), xytext=(1, 2.9),
                               arrowprops=dict(arrowstyle='->', color=color2, linewidth=3))

                    # Solvent label
                    sol_name2 = sol2.get('solvent', 'N/A')
                    ax.text(1.7, 2.45, f"{sol_name2}\n(sel: {sel2:.2f})", fontsize=9, ha='left', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color2, linewidth=1.5))

                    # Second polymer node
                    ax.scatter(1, 1.6, s=600, c=color2, zorder=10, edgecolors='black', linewidth=2)
                    ax.text(1, 1.6, second_polymer, ha='center', va='center', fontsize=10, fontweight='bold')

                    # Show remaining
                    if len(remaining_after_second) == 1:
                        ax.text(-0.3, 1.6, f'Remaining:\n{remaining_after_second[0]}',
                               ha='right', va='center', fontsize=8, color='#7f8c8d', style='italic')
                        # Final isolated
                        ax.annotate('', xy=(1, 0.3), xytext=(1, 1.2),
                                   arrowprops=dict(arrowstyle='->', color='#2ecc71', linewidth=3))
                        ax.scatter(1, 0, s=600, c='#2ecc71', zorder=10, edgecolors='black', linewidth=2, marker='s')
                        ax.text(1, 0, remaining_after_second[0], ha='center', va='center', fontsize=10, fontweight='bold')
                        ax.text(1, -0.5, '✓ Isolated', ha='center', va='top', fontsize=9, color='#27ae60', fontweight='bold')
                    else:
                        ax.text(-0.3, 1.6, f'Remaining:\n{", ".join(remaining_after_second)}',
                               ha='right', va='center', fontsize=8, color='#7f8c8d', style='italic')
                        ax.text(1, 0.8, f'Continue with\n{len(remaining_after_second)} more steps...',
                               ha='center', va='center', fontsize=8, color='#95a5a6', style='italic')

            # Add legend with percentage-based thresholds
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
                          markersize=12, markeredgecolor='black', label='Excellent (>30%)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f',
                          markersize=12, markeredgecolor='black', label='Good (10-30%)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22',
                          markersize=12, markeredgecolor='black', label='Marginal (0-10%)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                          markersize=12, markeredgecolor='black', label='Poor (<0%)'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                          markersize=12, markeredgecolor='black', label='Final (isolated)'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
                      bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True)

            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            filepath = save_plot(fig, f"separation_trees_{n_polymers}polymers")
            output.append(f"📊 Decision trees saved: {get_plot_url(filepath)}")

        except Exception as e:
            logger.error(f"Decision tree error: {e}", exc_info=True)
            output.append(f"⚠️ Could not create decision tree: {e}")
    
    # Summary recommendations
    output.append("\n## Recommendations\n")
    if sequence_scores and sequence_scores[0]["min_selectivity"] > 10:
        best = sequence_scores[0]
        output.append(f"✅ **Best sequence:** {' → '.join(best['sequence'])}")
        output.append(f"   - Minimum selectivity: {best['min_selectivity']:.1f}%")
        output.append(f"   - All steps have positive selectivity")
    elif sequence_scores:
        output.append("⚠️ **No sequence has all high-selectivity steps.**")
        output.append("Consider:")
        output.append("  - Exploring different temperatures")
        output.append("  - Using multi-stage extraction")
        output.append("  - Combining solvents")
    
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


def lookup_solvent_properties(solvent_names: list, solvent_table: str) -> dict:
    """
    Look up solvent properties with exact matching first, then fuzzy fallback.
    Returns a dict mapping solvent names to their properties.
    """
    if not solvent_table or solvent_table not in sql_db.table_schemas:
        return {}

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

    result = {}

    for solvent in solvent_names:
        sol_lower = solvent.lower().strip()
        props = {'logp': None, 'bp': None, 'energy': None, 'cp': None}

        # Try exact match on cosmobase column first
        if cosmobase_col:
            query = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{cosmobase_col}\") = '{sol_lower}'"
            try:
                df = sql_db.conn.execute(query).fetchdf()
                if len(df) == 1:
                    row = df.iloc[0]
                    props = {
                        'logp': row[logp_col] if logp_col and logp_col in row else None,
                        'bp': row[bp_col] if bp_col and bp_col in row else None,
                        'energy': row[energy_col] if energy_col and energy_col in row else None,
                        'cp': row[cp_col] if cp_col and cp_col in row else None,
                    }
                    result[solvent] = props
                    continue
            except:
                pass

        # Try exact match on name column
        if name_col:
            query = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{name_col}\") = '{sol_lower}'"
            try:
                df = sql_db.conn.execute(query).fetchdf()
                if len(df) == 1:
                    row = df.iloc[0]
                    props = {
                        'logp': row[logp_col] if logp_col and logp_col in row else None,
                        'bp': row[bp_col] if bp_col and bp_col in row else None,
                        'energy': row[energy_col] if energy_col and energy_col in row else None,
                        'cp': row[cp_col] if cp_col and cp_col in row else None,
                    }
                    result[solvent] = props
                    continue
            except:
                pass

        # Fuzzy match - find the best (shortest name that contains the solvent)
        match_col = cosmobase_col or name_col
        if match_col:
            query = f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{sol_lower}%' ORDER BY LENGTH(\"{match_col}\")"
            try:
                df = sql_db.conn.execute(query).fetchdf()
                if len(df) > 0:
                    # Take shortest match (most specific)
                    row = df.iloc[0]
                    props = {
                        'logp': row[logp_col] if logp_col and logp_col in row else None,
                        'bp': row[bp_col] if bp_col and bp_col in row else None,
                        'energy': row[energy_col] if energy_col and energy_col in row else None,
                        'cp': row[cp_col] if cp_col and cp_col in row else None,
                    }
            except:
                pass

        result[solvent] = props

    return result


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
def analyze_separation_with_properties(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: str,
    temperature: float = 25.0,
    rank_by: str = "selectivity",
    top_k: int = 10
) -> str:
    """
    Find selective solvents AND include their physical/economic properties.
    
    Combines separation analysis with solvent property data to help choose
    solvents based on both selectivity AND practical considerations.
    
    Args:
        table_name: Solubility data table
        polymer_column: Column with polymer names
        solvent_column: Column with solvent names
        temperature_column: Column with temperature values
        solubility_column: Column with solubility values
        target_polymer: Polymer to dissolve
        comparison_polymers: Comma-separated polymers to separate from
        temperature: Target temperature (default 25°C)
        rank_by: How to rank results - 'selectivity', 'energy' (cost), 'logp' (toxicity), 'bp'
        top_k: Number of top results to return
    
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
        prop_lookup = lookup_solvent_properties(solvent_names, solvent_table)

        if prop_lookup:
            properties_available = True
            for r in results:
                if r["solvent"] in prop_lookup:
                    r.update(prop_lookup[r["solvent"]])
    
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
            
            if with_energy:
                cheapest = min(with_energy, key=lambda x: x['energy'])
                output.append(f"- Lowest cost (with positive selectivity): **{cheapest['solvent']}** (Energy: {cheapest['energy']:.1f} J/g)")
            
            if with_logp:
                least_toxic = min(with_logp, key=lambda x: x['logp'])
                output.append(f"- Least toxic (with positive selectivity): **{least_toxic['solvent']}** (LogP: {least_toxic['logp']:.2f})")
    
    return "\n".join(output)


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
    plot_selectivity_heatmap,
    plot_multi_panel_analysis,
    plot_comparison_dashboard,
]

print(f"✅ Loaded {len(SQL_AGENT_TOOLS)} enhanced tools for SQL Agent")
print("\nTools include:")
print("  - Core DB: 6 tools (with validation)")
print("  - Adaptive Analysis: 4 tools (including sequential separation)")
print("  - Solvent Properties: 4 tools (properties, ranking, integrated analysis)")
print("  - Statistical: 4 tools")
print("  - Visualization: 4 tools")


# ============================================================
# LangGraph Agent Setup (PATCHED)
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr

# ============================================================
# Enhanced Agent Configuration
# ============================================================

SQL_AGENT_PROMPT = """You are an EXPERT SQL data analyst specializing in polymer-solvent solubility analysis with ADAPTIVE analysis capabilities and extensive verification workflows.

**YOUR MISSION:**
Provide thorough, ACCURATE data analysis with intelligent threshold adaptation. NEVER hallucinate values - ALWAYS verify data before reporting.

**CRITICAL BEHAVIORAL PRINCIPLES:**
1. **BE COMPREHENSIVE** - When asked to enumerate, list, or analyze multiple options, ACTUALLY DO IT. Don't just explain the concept - execute the analysis.
2. **VERIFY BEFORE REPORTING** - Always use verification tools before stating any numeric values
3. **ADAPTIVE THRESHOLDS** - Start stringent, relax if needed (don't assume restrictive thresholds)
4. **TARGETED COMPARISONS** - Only compare polymers the user asks about (not all polymers)
5. **EXPLORE CONDITIONS** - If separation not found at one temperature, try others
6. **ACTION OVER EXPLANATION** - If a tool can answer the question, USE IT. Don't explain what you would do - DO IT.
7. **CONSIDER PRACTICALITY** - Include solvent properties (cost, toxicity, boiling point) when recommending solvents

**MANDATORY WORKFLOW:**

## Step 1: Data Discovery (ALWAYS START HERE)
- `list_tables()` - See available data (includes solubility AND solvent property tables)
- `describe_table()` - Understand structure and statistics
- `check_column_values()` - Get EXACT values (case-sensitive!)

## Step 2: Input Validation (BEFORE ANY ANALYSIS)
- `validate_and_query()` - Verify all inputs exist before querying
- `verify_data_accuracy()` - Confirm row counts and sample data

## Step 3: Adaptive Analysis (USE THESE FOR SEPARATION QUESTIONS)
- `find_optimal_separation_conditions()` - **PRIMARY TOOL** for pairwise separation
- `adaptive_threshold_search()` - Find selective solvents with auto-threshold
- `analyze_selective_solubility_enhanced()` - Detailed selectivity analysis
- `plan_sequential_separation()` - **USE FOR MULTI-POLYMER SEQUENCES** - Enumerates ALL permutations, finds top-k solvents per step, creates decision tree

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
- `plot_solubility_vs_temperature()` - Temperature curves with confidence bands
- `plot_selectivity_heatmap()` - Heatmaps with optional target highlighting
- `plot_multi_panel_analysis()` - Comprehensive 4-panel separation analysis
- `plot_comparison_dashboard()` - Multi-polymer comparison dashboard

**SPECIAL CASES:**
- "What are all possible sequences/combinations to separate X polymers?" → USE `plan_sequential_separation()` immediately
- "Enumerate separation strategies" → USE `plan_sequential_separation()` with create_decision_tree=True
- "How can I separate A, B, C, D?" → USE `plan_sequential_separation()` to show ALL permutations
- "Rank by cost/cheapest solvents" → USE `rank_solvents_by_property('energy', ascending=True)`
- "Least toxic solvents" → USE `rank_solvents_by_property('logp', ascending=True)` (negative LogP = less toxic)
- "Separation with cost/toxicity" → USE `analyze_separation_with_properties()` with rank_by parameter
- "What are the properties of X solvent?" → USE `get_solvent_properties('X')`

**SOLVENT PROPERTY INTERPRETATION:**
- **LogP**: Lower/negative = less toxic, more water soluble. Higher = more toxic, more lipophilic
- **Energy (J/g)**: Lower = cheaper to use (less energy for heating/recovery)
- **Boiling Point**: Lower = easier solvent recovery, but may need pressure vessels
- **Cp (Heat Capacity)**: Higher = more energy needed to heat

Remember: ACCURACY > SPEED. ACTION > EXPLANATION. Always verify before reporting. Let the adaptive tools do their job - they will find results if any exist."""


# ============================================================
# PATCHED: Robust Agent State and Nodes
# ============================================================

class AgentState(MessagesState):
    """Enhanced state - defaults handled in functions."""
    iteration_count: int
    max_iterations: int


class RobustToolNode(ToolNode):
    """ToolNode with comprehensive error handling."""
    
    def __call__(self, state):
        try:
            result = super().__call__(state)
            
            # Truncate long tool outputs
            if result and isinstance(result, dict) and "messages" in result:
                messages = result.get("messages", [])
                if isinstance(messages, list):
                    for i, msg in enumerate(messages):
                        if isinstance(msg, ToolMessage) and msg.content:
                            if len(str(msg.content)) > MAX_TOOL_OUTPUT_LENGTH:
                                truncated = truncate_output(str(msg.content))
                                result["messages"][i] = ToolMessage(
                                    content=truncated,
                                    tool_call_id=msg.tool_call_id
                                )
            
            # Periodic cleanup
            gc.collect()
            return result
            
        except Exception as e:
            logger.error(f"Tool error: {e}\n{traceback.format_exc()}")
            
            # Find tool_call_id from state safely
            tool_call_id = "error"
            raw_messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
            
            # Ensure messages is a list
            if raw_messages is None:
                messages = []
            elif isinstance(raw_messages, str):
                messages = []
            elif isinstance(raw_messages, list):
                messages = raw_messages
            else:
                try:
                    messages = list(raw_messages)
                except:
                    messages = []
            
            for msg in reversed(messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    try:
                        tool_call_id = msg.tool_calls[0].get('id', 'error')
                    except:
                        tool_call_id = 'error'
                    break
            
            return {
                "messages": [ToolMessage(
                    content=f"**Tool Error:** {str(e)[:500]}\n\nTry verifying inputs with `describe_table()` or `check_column_values()`.",
                    tool_call_id=tool_call_id
                )]
            }


def sql_agent_node(state: AgentState):
    """Robust agent node with comprehensive error handling."""
    
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
    # Gemini requires function calls to come after user/function response turns
    def sanitize_messages_for_gemini(msgs):
        """
        Ensure message history follows Gemini's required ordering:
        - Function calls must come immediately after user turn or function response
        - Remove orphaned tool messages without matching AI tool calls
        """
        if not msgs:
            return msgs
        
        sanitized = []
        i = 0
        
        while i < len(msgs):
            msg = msgs[i]
            
            # Always include HumanMessage
            if isinstance(msg, HumanMessage):
                sanitized.append(msg)
                i += 1
                continue
            
            # For AIMessage with tool_calls, ensure it follows user or tool response
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Check what came before
                    if sanitized:
                        last = sanitized[-1]
                        # Valid: after HumanMessage or ToolMessage
                        if isinstance(last, (HumanMessage, ToolMessage)):
                            sanitized.append(msg)
                        else:
                            # Invalid position - skip or add placeholder
                            logger.warning("Skipping AI tool call in invalid position")
                    else:
                        # First message shouldn't be a tool call
                        logger.warning("Skipping AI tool call as first message")
                else:
                    # Regular AI message without tool calls - always ok
                    sanitized.append(msg)
                i += 1
                continue
            
            # For ToolMessage, ensure matching AI message with tool_calls exists before it
            if isinstance(msg, ToolMessage):
                # Check if we have a preceding AI message with matching tool call
                has_matching_ai = False
                for prev in reversed(sanitized):
                    if isinstance(prev, AIMessage) and hasattr(prev, 'tool_calls') and prev.tool_calls:
                        # Check if tool_call_id matches
                        for tc in prev.tool_calls:
                            if tc.get('id') == msg.tool_call_id:
                                has_matching_ai = True
                                break
                        if has_matching_ai:
                            break
                    elif isinstance(prev, HumanMessage):
                        # Hit a human message without finding matching AI - orphaned tool response
                        break
                
                if has_matching_ai:
                    sanitized.append(msg)
                else:
                    logger.warning(f"Skipping orphaned ToolMessage: {msg.tool_call_id}")
                i += 1
                continue
            
            # Other message types - include
            sanitized.append(msg)
            i += 1
        
        return sanitized
    
    # Apply sanitization
    messages = sanitize_messages_for_gemini(messages)
    
    # Ensure each message in the list is valid
    valid_messages = [msg for msg in messages if msg is not None]
    
    try:
        sql_llm = llm.bind_tools(SQL_AGENT_TOOLS)
        
        # Ensure SQL_AGENT_PROMPT is a string
        prompt = SQL_AGENT_PROMPT if isinstance(SQL_AGENT_PROMPT, str) else str(SQL_AGENT_PROMPT)
        
        # Build full messages list carefully
        system_msg = SystemMessage(content=prompt)
        full_messages = [system_msg] + valid_messages
        
        logger.debug(f"Invoking LLM with {len(full_messages)} messages")
        response = sql_llm.invoke(full_messages)
        
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
                    response = sql_llm.invoke(clean_messages)
                    
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
    """Safe continuation check."""
    
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
    
    try:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
    except (IndexError, TypeError, AttributeError) as e:
        logger.debug(f"should_continue check failed: {e}")
        return "end"
    
    return "end"


# Build enhanced agent graph
builder = StateGraph(AgentState)
builder.add_node("agent", sql_agent_node)
builder.add_node("tools", RobustToolNode(SQL_AGENT_TOOLS))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
agent_graph = builder.compile(checkpointer=checkpointer)

logger.info("✅ Enhanced SQL Agent System compiled successfully!")
logger.info(f"SQL Agent: {len(SQL_AGENT_TOOLS)} tools available")


# ============================================================
# PATCHED: Enhanced Gradio Interface Functions
# ============================================================

def create_thread_id():
    """Create new thread ID for conversation."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

# Create fresh config on startup
config = create_thread_id()

def verify_system_ready():
    """Verify the system is properly initialized."""
    issues = []
    warnings = []
    
    # Check database tables
    if not sql_db.table_schemas:
        issues.append("No tables loaded in database")
    else:
        logger.info(f"Database OK: {len(sql_db.table_schemas)} tables loaded")
        table_names = list(sql_db.table_schemas.keys())
        logger.info(f"  Tables: {table_names}")
        
        # Check for required tables
        has_solubility = any('solvent' in t.lower() and 'database' in t.lower() for t in table_names)
        has_properties = any('solvent' in t.lower() and 'data' in t.lower() and 'database' not in t.lower() for t in table_names)
        
        if not has_solubility:
            warnings.append("COMMON-SOLVENTS-DATABASE.csv not loaded - solubility analysis unavailable")
        if not has_properties:
            warnings.append("Solvent_Data.csv not loaded - solvent property tools unavailable")
    
    # Check tools
    if not SQL_AGENT_TOOLS:
        issues.append("No tools loaded")
    else:
        logger.info(f"Tools OK: {len(SQL_AGENT_TOOLS)} tools available")
    
    # Check LLM (skip if not critical to avoid slow startup)
    try:
        # Just check llm exists, don't invoke to save time
        if llm is not None:
            logger.info("LLM OK: Model configured")
        else:
            issues.append("LLM not configured")
    except Exception as e:
        issues.append(f"LLM error: {e}")
    
    # Log warnings
    for w in warnings:
        logger.warning(f"⚠️ {w}")
    
    if issues:
        logger.warning(f"System initialization issues: {issues}")
        return False, issues
    
    return True, warnings

# Verify system on startup
system_ready, system_issues = verify_system_ready()
if not system_ready:
    logger.warning("System not fully ready - some features may not work correctly")
    logger.warning(f"Issues: {system_issues}")


def clear_session():
    """Clear session with proper cleanup."""
    global config
    
    old_id = config.get("configurable", {}).get("thread_id", "unknown")
    
    try:
        if hasattr(agent_graph, 'checkpointer') and agent_graph.checkpointer:
            try:
                agent_graph.checkpointer.delete_thread(old_id)
            except AttributeError:
                pass  # MemorySaver may not have delete_thread
            except Exception as e:
                logger.warning(f"Could not delete thread: {e}")
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Create new session
    config = create_thread_id()
    
    return f"✅ Session cleared. New ID: {config['configurable']['thread_id'][:8]}..."


def chat_with_agent(message, history):
    """Robust chat handler with proper error handling."""
    
    if not message or not message.strip():
        return "Please enter a message.", []
    
    message = message.strip()
    start_time = time.time()
    
    # Track plots before
    try:
        existing_plots = set(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
    except:
        existing_plots = set()
    
    try:
        # Ensure we have a fresh config for each new conversation
        logger.debug(f"chat_with_agent called with message: {message[:50]}...")
        logger.debug(f"Config thread_id: {config.get('configurable', {}).get('thread_id', 'unknown')}")
        
        # Initialize state properly with explicit types
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "iteration_count": 0,
            "max_iterations": MAX_ITERATIONS
        }
        
        logger.debug(f"Initial state messages type: {type(initial_state['messages'])}")
        
        result = agent_graph.invoke(initial_state, config)
        
        elapsed = time.time() - start_time
        
        # Extract response safely
        raw_messages = result.get("messages")
        
        # Handle various result types
        if raw_messages is None:
            return "No response generated (messages is None).", []
        elif isinstance(raw_messages, str):
            # If messages became a string somehow, use it directly
            logger.warning(f"Result messages was a string: {raw_messages[:100]}")
            content = raw_messages
            iterations = result.get("iteration_count", 0)
        elif isinstance(raw_messages, list):
            if not raw_messages:
                return "No response generated (empty messages).", []
            final = raw_messages[-1]
            content = getattr(final, 'content', str(final)) or "Processing complete."
            iterations = result.get("iteration_count", 0)
        else:
            logger.warning(f"Unexpected messages type: {type(raw_messages)}")
            content = str(raw_messages)
            iterations = 0
        
        # Find new plots
        time.sleep(0.2)
        try:
            new_plots = list(set(glob.glob(os.path.join(PLOTS_DIR, "*.png"))) - existing_plots)
            new_plots.sort(key=os.path.getmtime, reverse=True)
        except:
            new_plots = []
        
        footer = f"\n\n---\n⏱️ {elapsed:.1f}s | 🔄 {iterations} iterations"
        if new_plots:
            footer += f" | 📊 {len(new_plots)} plot(s)"
        
        return content + footer, new_plots
        
    except TypeError as e:
        # Special handling for type errors (like list + str)
        elapsed = time.time() - start_time
        error_trace = traceback.format_exc()
        logger.error(f"Type error in chat: {e}\n{error_trace}")
        
        # Try to identify the specific issue
        if "can only concatenate" in str(e):
            return (
                f"**❌ Type Error ({elapsed:.1f}s):**\n```\n{str(e)}\n```\n\n"
                f"**Debug info:**\n"
                f"This usually happens when the database isn't fully loaded.\n\n"
                f"**Solution:** Click 'Reindex All Data' in the Data Management tab, then try again.\n\n"
                f"**Or try:**\n"
                f"1. Clear session and try again\n"
                f"2. Restart the notebook"
            ), []
        else:
            return (
                f"**❌ Type Error ({elapsed:.1f}s):**\n```\n{str(e)[:400]}\n```\n\n"
                f"**Try:**\n"
                f"1. 'What tables are available?'\n"
                f"2. 'Describe the [table] table'\n"
                f"3. Clear session button"
            ), []
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        
        return (
            f"**❌ Error ({elapsed:.1f}s):**\n```\n{str(e)[:400]}\n```\n\n"
            f"**Try:**\n"
            f"1. 'What tables are available?'\n"
            f"2. 'Describe the [table] table'\n"
            f"3. Clear session button"
        ), []


def reindex_all():
    """Reindex all CSV files with enhanced reporting."""
    try:
        start_time = time.time()
        sql_db.load_csv_files()
        sql_tables = len(sql_db.table_schemas)
        elapsed = time.time() - start_time

        total_rows = sum(schema['row_count'] for schema in sql_db.table_schemas.values())
        total_cols = sum(len(schema['columns']) for schema in sql_db.table_schemas.values())

        # Cleanup old plots
        cleanup_old_plots()

        return f"""**✅ Reindexing Complete**

📊 **Database Statistics:**
- SQL Tables Loaded: {sql_tables}
- Total Rows: {total_rows:,}
- Total Columns: {total_cols}
- Time Taken: {elapsed:.2f}s

**Tables:**
{chr(10).join([f"- **{name}**: {schema['row_count']:,} rows, {len(schema['columns'])} columns" for name, schema in sql_db.table_schemas.items()])}

System is ready for queries and visualization!"""
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return f"**❌ Reindexing Failed**\n\nError: {str(e)}"


def upload_csv_file(file):
    """Handle CSV file upload with validation."""
    if file is None:
        return "❌ No file uploaded."
    try:
        import shutil

        try:
            df = pd.read_csv(file.name)
            row_count = len(df)
            col_count = len(df.columns)
        except Exception as e:
            return f"**❌ Invalid CSV File**\n\nError: {e}"

        dest_path = os.path.join(DATA_DIR, os.path.basename(file.name))
        shutil.copy(file.name, dest_path)
        sql_db.load_csv_files()

        del df
        gc.collect()

        return f"""**✅ CSV File Uploaded Successfully**

📁 **File:** {os.path.basename(file.name)}
📊 **Rows:** {row_count:,}
📋 **Columns:** {col_count}

Database reloaded. Ready for queries and visualization."""
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        return f"**❌ Upload Failed**\n\nError: {str(e)}"


def show_schema_info():
    """Display enhanced database schema information."""
    try:
        schema_info = sql_db.get_table_info()
        if not schema_info or schema_info == "No tables available.":
            return "**❌ No Tables Available**\n\nPlease upload CSV files or reindex the database."

        enhanced_info = f"""**📊 SQL Database Schema**

{schema_info}

---
**🔧 Available Analysis Capabilities:**

**Core Operations:** list_tables, describe_table, check_column_values, query_database, verify_data_accuracy, validate_and_query

**Adaptive Analysis:** find_optimal_separation_conditions, adaptive_threshold_search, analyze_selective_solubility_enhanced

**Statistical:** statistical_summary, correlation_analysis, compare_groups_statistically, regression_analysis

**Visualization:** plot_solubility_vs_temperature, plot_selectivity_heatmap, plot_multi_panel_analysis, plot_comparison_dashboard
"""
        return enhanced_info
    except Exception as e:
        return f"**❌ Error Retrieving Schema**\n\n{str(e)}"


def show_system_status():
    """Display comprehensive system status."""
    try:
        sql_tables = len(sql_db.table_schemas)
        total_rows = sum(schema['row_count'] for schema in sql_db.table_schemas.values())
        thread_id = config["configurable"]["thread_id"]

        plot_files = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
        html_files = glob.glob(os.path.join(PLOTS_DIR, "*.html"))
        plot_count = len(plot_files)
        html_count = len(html_files)

        status = f"""**🚀 Enhanced System Status (PATCHED)**

**📊 SQL Database:**
- Tables: {sql_tables}
- Total Rows: {total_rows:,}
- Engine: DuckDB (in-memory)
- Validator: ✅ Active
- Adaptive Analyzer: ✅ Active

**🎨 Visualization:**
- PNG Plots Saved: {plot_count}
- Interactive HTML: {html_count}
- Auto-cleanup: ✅ Enabled (keeps {MAX_PLOTS_TO_KEEP} latest)

**💬 Session:**
- Thread ID: `{thread_id[:12]}...`
- Max Iterations: {MAX_ITERATIONS}
- Max Message History: {MAX_MESSAGE_HISTORY}
- Status: ✅ Active

**🔧 Agent Tools:** {len(SQL_AGENT_TOOLS)} total

**📁 Directories:**
- Data: `{DATA_DIR}`
- Plots: `{PLOTS_DIR}`

**⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🛡️ Robustness Features:**
✓ Graceful error handling
✓ Memory-efficient state management
✓ Tool output truncation
✓ Automatic plot cleanup
✓ Session memory limits
✓ Garbage collection"""
        return status
    except Exception as e:
        return f"**❌ Error Retrieving Status**\n\n{str(e)}"


def list_saved_plots():
    """List all saved plots with enhanced info."""
    try:
        png_files = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
        html_files = glob.glob(os.path.join(PLOTS_DIR, "*.html"))

        if not png_files and not html_files:
            return "**No plots saved yet.**\n\nCreate visualizations using the chat interface!"

        all_files = png_files + html_files
        all_files.sort(key=os.path.getmtime, reverse=True)

        output = f"**📊 Saved Plots ({len(png_files)} PNG, {len(html_files)} HTML)**\n\n"

        for i, filepath in enumerate(all_files[:25], 1):
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath) / 1024
            modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            file_type = "🖼️" if filepath.endswith('.png') else "🌐"
            output += f"{i}. {file_type} `{filename}` ({filesize:.1f} KB) - {modified.strftime('%m/%d %H:%M')}\n"

        if len(all_files) > 25:
            output += f"\n... and {len(all_files) - 25} more files"

        output += f"\n\n📁 Location: `{PLOTS_DIR}`"
        return output
    except Exception as e:
        return f"❌ Error listing plots: {e}"


def get_latest_plots(n=15):
    """Get n most recent plots."""
    try:
        plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
        if not plots:
            return []
        plots.sort(key=os.path.getmtime, reverse=True)
        return plots[:n]
    except Exception as e:
        logger.error(f"Error getting latest plots: {e}")
        return []


# Custom CSS for enhanced UI
custom_css = """
.gradio-container {
    max-width: 1800px !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button-primary {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.gr-button-secondary {
    background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%) !important;
    border: none !important;
    color: white !important;
}
.gr-box {
    border-radius: 10px;
}
"""

# Create Enhanced Gradio Interface
with gr.Blocks(title="Enhanced Polymer Solubility Analysis System (PATCHED)") as demo:
    gr.Markdown("""
    # 🧪 Enhanced Polymer Solubility Analysis System (PATCHED)
    **Adaptive Analysis | Intelligent Verification | Advanced Visualization | Robust Error Handling**

    *Features: Auto-threshold searching • Targeted comparisons • Statistical analysis • Memory-efficient • Graceful error recovery*
    """)

    with gr.Tabs():
        # ========== CHAT INTERFACE TAB ==========
        with gr.Tab("💬 Analysis Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about polymer separation, solubility analysis, or create visualizations...",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)

                    with gr.Row():
                        clear_chat_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        new_session_btn = gr.Button("🔄 New Session", size="sm", variant="secondary")
                        cleanup_btn = gr.Button("🧹 Cleanup Plots", size="sm")

                    plot_gallery = gr.Gallery(
                        label="📊 Generated Plots",
                        show_label=True,
                        columns=2,
                        height=400,
                        object_fit="contain",
                        preview=True,
                        allow_preview=True
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Quick Commands")
                    gr.Markdown("""
                    **Data Discovery:**
                    - What tables are available?
                    - Describe the [table] table
                    - What polymers are in the data?
                    
                    **Analysis:**
                    - What is the solubility of LDPE in dodecane?
                    - Can I separate LDPE from PET?
                    - Find selective solvents for EVOH
                    
                    **Visualization:**
                    - Plot solubility vs temperature
                    - Create selectivity heatmap
                    - Generate comparison dashboard
                    """)

            # Event handlers - matching working version
            def respond(message, history):
                response, images = chat_with_agent(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return history, "", images

            submit_btn.click(
                fn=respond,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, plot_gallery]
            )
            msg_input.submit(
                fn=respond,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, plot_gallery]
            )
            clear_chat_btn.click(fn=lambda: ([], []), outputs=[chatbot, plot_gallery])
            new_session_btn.click(fn=clear_session, outputs=gr.Textbox(visible=False))
            cleanup_btn.click(fn=lambda: cleanup_old_plots(), outputs=gr.Textbox(visible=False))

        # ========== DATA MANAGEMENT TAB ==========
        with gr.Tab("📁 Data Management"):
            gr.Markdown("### 📤 Upload CSV Data")
            with gr.Row():
                with gr.Column():
                    csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
                    csv_upload_btn = gr.Button("📤 Upload & Load", variant="primary")
                    csv_status = gr.Markdown()

                with gr.Column():
                    gr.Markdown("### 📊 Current Schema")
                    refresh_schema_btn = gr.Button("🔄 Refresh Schema", size="sm")
                    schema_display = gr.Markdown()

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 Database Operations")
                    reindex_all_btn = gr.Button("♻️ Reindex All Data", variant="secondary", size="lg")
                    reindex_status = gr.Markdown()

            csv_upload_btn.click(fn=upload_csv_file, inputs=csv_upload, outputs=csv_status)
            refresh_schema_btn.click(fn=show_schema_info, outputs=schema_display)
            reindex_all_btn.click(fn=reindex_all, outputs=reindex_status)

        # ========== VISUALIZATION GALLERY TAB ==========
        with gr.Tab("🎨 Visualization Gallery"):
            gr.Markdown("""
            ### Available Visualization Types

            The enhanced system supports **multiple visualization types** with intelligent data verification:
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    **📈 Temperature vs Solubility Curves**
                    - Multi-polymer, multi-solvent analysis
                    - Optional 95% confidence bands
                    - Automatic input validation

                    **🔥 Selectivity Heatmaps**
                    - Cross-comparison matrices
                    - Optional target polymer highlighting
                    - Color-coded selectivity
                    """)

                with gr.Column():
                    gr.Markdown("""
                    **📊 Multi-Panel Analysis**
                    - 4-panel comprehensive view
                    - Separation windows
                    - Summary statistics

                    **📉 Comparison Dashboards**
                    - Grouped bar charts
                    - Heatmaps
                    - Box plots
                    - Rankings
                    """)

            gr.Markdown("---")
            gr.Markdown("### 📊 All Saved Plots")
            all_plots_gallery = gr.Gallery(
                label="Plots",
                columns=3,
                height=600,
                object_fit="contain",
                preview=True
            )
            with gr.Row():
                refresh_all_plots_btn = gr.Button("🔄 Load All Plots", size="sm", variant="primary")
                plots_list_display = gr.Markdown()

            refresh_all_plots_btn.click(fn=lambda: get_latest_plots(50), outputs=all_plots_gallery)
            refresh_all_plots_btn.click(fn=list_saved_plots, outputs=plots_list_display)

        # ========== SYSTEM INFO TAB ==========
        with gr.Tab("ℹ️ System Information"):
            gr.Markdown("### 🖥️ System Overview")
            status_refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            system_status_display = gr.Markdown(value=show_system_status())
            status_refresh_btn.click(fn=show_system_status, outputs=system_status_display)

            gr.Markdown("---")
            gr.Markdown("""
            ### 🛡️ Robustness Features (PATCHED)
            
            This version includes:
            - **Graceful error handling** - Tools recover from errors without crashing
            - **Memory efficiency** - Automatic cleanup of old plots and message history trimming
            - **Tool output truncation** - Very long outputs are truncated to prevent memory issues
            - **Session management** - Proper cleanup when clearing sessions
            - **Garbage collection** - Automatic memory cleanup after operations
            
            ### 📚 Usage Tips

            **For Separation Questions:**
            ```
            "Can [polymer A] be separated from [polymer B] using [solvent]?"
            "Find optimal conditions to separate [A] from [B]"
            ```

            **For Solubility Queries:**
            ```
            "What is the solubility of [polymer] in [solvent] at [temp]?"
            "Plot solubility vs temperature for [polymer] in [solvent]"
            ```
            """)

    gr.Markdown("""
    ---
    **🔧 Status:** Ready | **⚡ Engine:** DuckDB + LangGraph + AI | **🛡️ Version:** PATCHED (Memory-efficient + Error-handling)
    """)


# ============================================================
# Launch Application
# ============================================================

logger.info("\n" + "="*70)
logger.info("🧪 ENHANCED POLYMER SOLUBILITY ANALYSIS SYSTEM - PATCHED VERSION")
logger.info("="*70)
logger.info(f"📊 SQL Tables: {len(sql_db.table_schemas)}")
logger.info(f"🔧 Agent Tools: {len(SQL_AGENT_TOOLS)}")
logger.info(f"🛡️ Robustness: Memory-efficient + Error-handling")
logger.info(f"📁 Data Directory: {DATA_DIR}")
logger.info(f"📊 Plots Directory: {PLOTS_DIR}")
logger.info("="*70 + "\n")

# Launch Gradio interface
demo.launch(
    share=True,
    server_port=None,
    show_error=True,
    inbrowser=False
)
