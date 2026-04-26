"""Shared helpers for all STRAP tool modules.

Provides: output formatting, safe_tool_wrapper, DataValidator,
AdaptiveAnalyzer, fuzzy matching, solvent name normalization.
"""

from __future__ import annotations

import gc
import json
import re
import os
import logging
import subprocess
import time
import asyncio
from functools import wraps
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MAX_TOOL_OUTPUT_LENGTH = 50_000
PLOTS_DIR = os.environ.get("STRAP_PLOTS_DIR", "./plots")

def get_plots_dir() -> str:
    return os.environ.get("STRAP_PLOTS_DIR", PLOTS_DIR)

def set_plots_dir(new_dir: str) -> str:
    global PLOTS_DIR
    old = get_plots_dir()
    PLOTS_DIR = new_dir
    os.environ["STRAP_PLOTS_DIR"] = new_dir
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return old


# ------------------------------------------------------------------
# Output formatting / truncation
# ------------------------------------------------------------------

def truncate_output(text: str, max_length: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 50
    return text[:half] + f"\n\n... [TRUNCATED {len(text) - max_length} chars] ...\n\n" + text[-half:]


def _format_tool_result(result) -> str:
    if result is None:
        return "Operation completed (no output)."
    result_str = str(result)
    if len(result_str) > MAX_TOOL_OUTPUT_LENGTH:
        return truncate_output(result_str)
    return result_str


def _is_tool_envelope(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and "display" in parsed and "data" in parsed


def _normalize_tool_envelope(tool_name: str, text: str) -> str:
    from strap.services.tool_response_service import json_tool_response

    parsed = json.loads(text)
    data = parsed.get("data")
    if not isinstance(data, dict):
        data = {"value": data}
    success = data.get("success") if "success" in data else None
    return json_tool_response(
        parsed.get("display", ""),
        data,
        tool_name=tool_name,
        success=success,
    )


def _looks_like_error_text(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False

    failure_prefixes = (
        "error:",
        "failed",
        "search failed",
        "literature search failed",
        "ingestion failed",
        "no results found",
        "no relevant",
        "no data found",
        "no documents",
        "no pdf",
        "no tables",
        "google scholar search requires",
        "failed to ",
        "could not ",
    )
    if normalized.startswith("❌") or normalized.startswith("⚠️"):
        return True
    if any(normalized.startswith(prefix) for prefix in failure_prefixes):
        return True
    failure_substrings = (
        " not ready",
        " cannot be empty",
        " not found",
        " failed:",
        " failed ",
        " unavailable",
    )
    if any(substr in normalized for substr in failure_substrings):
        return True
    return False


def _wrap_structured_result(tool_name: str, result) -> str:
    from strap.services.tool_response_service import json_tool_error, json_tool_response

    if isinstance(result, str):
        if _is_tool_envelope(result):
            return _normalize_tool_envelope(tool_name, result)
        result_str = truncate_output(result)
        try:
            parsed = json.loads(result_str)
        except (json.JSONDecodeError, TypeError, ValueError):
            if _looks_like_error_text(result_str):
                return json_tool_error(
                    result_str,
                    tool_name=tool_name,
                    error_code="tool_reported_failure",
                )
            return json_tool_response(result_str, tool_name=tool_name, success=True)
        if isinstance(parsed, dict):
            return json_tool_response(
                result_str,
                parsed,
                tool_name=tool_name,
            )
        if isinstance(parsed, list):
            return json_tool_response(
                result_str,
                {"items": parsed},
                tool_name=tool_name,
            )
        return json_tool_response(result_str, {"value": parsed}, tool_name=tool_name)

    if isinstance(result, dict):
        if "display" in result and "data" in result:
            return json.dumps(result, indent=2, ensure_ascii=False)
        display = json.dumps(result, indent=2, ensure_ascii=False)
        return json_tool_response(display, result, tool_name=tool_name)
    if isinstance(result, list):
        display = json.dumps(result, indent=2, ensure_ascii=False)
        return json_tool_response(display, {"items": result}, tool_name=tool_name)

    result_str = _format_tool_result(result)
    return json_tool_response(result_str, tool_name=tool_name, success=True)


def _format_tool_error(func_name: str, error: Exception) -> str:
    """Format tool error with context-aware recovery suggestions."""
    error_msg = str(error)[:500]
    header = f"ERROR in {func_name}:\n{error_msg}\n\nSuggestions:\n"

    if isinstance(error, NameError):
        return header + (
            "- This tool depends on an engine/module that failed to load.\n"
            "- Try a different tool or approach for this task.\n"
            "- Report this as a system configuration issue."
        )
    if isinstance(error, ImportError):
        return header + (
            "- A required dependency is not installed.\n"
            "- Try a different tool or approach for this task.\n"
            "- Report this as a system configuration issue."
        )
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return header + (
            "- An external service (API/network) timed out or is unreachable.\n"
            "- Wait a moment and retry, or try with fewer items.\n"
            "- If the issue persists, skip this step and note the data gap."
        )
    if isinstance(error, subprocess.CalledProcessError) or "subprocess" in error_msg.lower() or "biosteam" in error_msg.lower():
        return header + (
            "- A subprocess simulation failed.\n"
            "- Check that the polymer and solvent names are valid.\n"
            "- Try a different solvent or check supported polymers with get_biosteam_solvents()."
        )
    if "sql" in error_msg.lower() or (isinstance(error, RuntimeError) and "query" in error_msg.lower()):
        return header + (
            "- Verify input parameters with describe_table()\n"
            "- Check values with check_column_values()\n"
            "- Use verify_data_accuracy() to confirm data exists"
        )
    # Default fallback — keep the database suggestions for genuinely unknown errors
    return header + (
        "- Verify input parameters with describe_table()\n"
        "- Check values with check_column_values()\n"
        "- Use verify_data_accuracy() to confirm data exists"
    )


def _format_structured_tool_error(
    tool_name: str,
    error: Exception,
    *,
    error_code: str = "tool_execution_failed",
) -> str:
    from strap.services.tool_response_service import json_tool_error

    return json_tool_error(
        str(error),
        tool_name=tool_name,
        error_code=error_code,
        exception_type=type(error).__name__,
    )


def _run_coroutine(coro):
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def safe_tool_wrapper(
    func=None,
    *,
    structured_output: bool = False,
    tool_name: str | None = None,
    error_code: str = "tool_execution_failed",
):
    """Decorator for safe tool execution with error handling and memory cleanup.

    Always produces a sync wrapper so LangGraph/deepagents can invoke tools
    synchronously.  Async tool functions are called via _run_coroutine().
    """
    if func is None:
        return lambda real_func: safe_tool_wrapper(
            real_func,
            structured_output=structured_output,
            tool_name=tool_name,
            error_code=error_code,
        )

    resolved_tool_name = tool_name or func.__name__

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = _run_coroutine(func(*args, **kwargs))
                if structured_output:
                    return _wrap_structured_result(resolved_tool_name, result)
                return _format_tool_result(result)
            except Exception as e:
                logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
                if structured_output:
                    return _format_structured_tool_error(
                        resolved_tool_name,
                        e,
                        error_code=error_code,
                    )
                return _format_tool_error(func.__name__, e)
            finally:
                gc.collect()
        return sync_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if structured_output:
                    return _wrap_structured_result(resolved_tool_name, result)
                return _format_tool_result(result)
            except Exception as e:
                logger.error(f"Tool {func.__name__} error: {e}", exc_info=True)
                if structured_output:
                    return _format_structured_tool_error(
                        resolved_tool_name,
                        e,
                        error_code=error_code,
                    )
                return _format_tool_error(func.__name__, e)
            finally:
                gc.collect()
        return sync_wrapper


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug (lowercase, underscores)."""
    import re
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


def descriptive_plot_name(
    base: str,
    polymers: list[str] | None = None,
    solvents: list[str] | None = None,
) -> str:
    """Build a descriptive plot filename from base name + polymer/solvent lists."""
    parts = []
    if polymers:
        parts.append("_".join(_slugify(p) for p in polymers[:4]))
    if solvents:
        parts.append("_".join(_slugify(s) for s in solvents[:3]))
    parts.append(base)
    return "_".join(parts)


def normalize_wsl_path(path: str | os.PathLike[str]) -> str:
    """Normalize Windows WSL UNC paths to paths usable inside WSL/Linux."""
    raw = os.fspath(path).strip().strip('"').strip("'")
    if not raw:
        return raw
    raw = re.sub(r"\r?\n[ \t]*", "", raw)

    normalized = raw.replace("\\", "/")
    lowered = normalized.lower()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if lowered.startswith(prefix):
            remainder = normalized[len(prefix):]
            parts = remainder.split("/", 1)
            return "/" + parts[1] if len(parts) == 2 else "/"

    if re.match(r"^[A-Za-z]:/", normalized):
        drive = normalized[0].lower()
        return f"/mnt/{drive}{normalized[2:]}"

    return normalized


def _default_plot_extension(plot_type: str) -> str:
    return ".html" if plot_type == "plotly" else ".png"


def resolve_plot_output_path(
    plot_name: str,
    plot_type: str = "matplotlib",
    *,
    output_dir: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a plot stem/name plus optional user destination to a file path."""
    extension = _default_plot_extension(plot_type)
    plot_path = Path(normalize_wsl_path(plot_name))
    filename = plot_path.name
    if not Path(filename).suffix:
        filename = f"{filename}{extension}"

    if output_path:
        candidate = Path(normalize_wsl_path(output_path))
        output_text = os.fspath(output_path).strip()
        if output_text.endswith(("\\", "/")) or not candidate.suffix or candidate.is_dir():
            return str(candidate / filename)
        if not candidate.suffix:
            candidate = candidate.with_suffix(extension)
        return str(candidate)

    base_dir = Path(normalize_wsl_path(output_dir)) if output_dir else Path(normalize_wsl_path(get_plots_dir()))
    return str(base_dir / filename)


def save_plot(
    fig,
    plot_name: str,
    plot_type: str = "matplotlib",
    *,
    output_dir: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    dpi: int = 300,
    write_html_kwargs: dict[str, Any] | None = None,
    **savefig_kwargs,
) -> str:
    """Save a matplotlib/plotly figure and return the file path."""
    filepath = resolve_plot_output_path(
        plot_name,
        plot_type,
        output_dir=output_dir,
        output_path=output_path,
    )
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if plot_type == "plotly":
        fig.write_html(filepath, **(write_html_kwargs or {}))
    else:
        savefig_kwargs.setdefault("bbox_inches", "tight")
        fig.savefig(filepath, dpi=dpi, **savefig_kwargs)
        import matplotlib.pyplot as plt
        plt.close(fig)
    return filepath


# ------------------------------------------------------------------
# Solvent name mapping (cross-database normalization)
# ------------------------------------------------------------------

from strap.solvent_registry import resolve_for_databases


def normalize_solvent_name(solvent_name: str, target_database: str = "property") -> Optional[str]:
    """Normalize a solvent name from solubility DB to property or GSK DB."""
    return resolve_for_databases(solvent_name, target_database)


def get_cross_database_properties(solvent_name: str, conn) -> Dict[str, Any]:
    """Get properties for a solvent by looking up in property and GSK databases."""
    props = {
        'bp': None, 'logp': None, 'energy': None, 'cp': None,
        'g_score': None, 'gsk_class': None
    }

    prop_name = normalize_solvent_name(solvent_name, "property")
    if prop_name:
        try:
            query = """
            SELECT bp__oc_, logp, energy__j_g_, cp__j_g_k_
            FROM solvent_data
            WHERE LOWER(solvent_name) = LOWER(?)
            OR LOWER(solvent_name) LIKE '%' || LOWER(?) || '%'
            LIMIT 1
            """
            result = conn.execute(query, [prop_name, prop_name]).fetchdf()
            if len(result) > 0:
                row = result.iloc[0]
                props['bp'] = row.get('bp__oc_')
                props['logp'] = row.get('logp')
                props['energy'] = row.get('energy__j_g_')
                props['cp'] = row.get('cp__j_g_k_')
        except Exception as e:
            logger.debug(f"Property lookup failed for {solvent_name}: {e}")

    gsk_name = normalize_solvent_name(solvent_name, "gsk")
    if gsk_name:
        try:
            query = """
            SELECT g_score, classification
            FROM gsk_dataset
            WHERE LOWER(solvent_common_name) = LOWER(?)
            OR LOWER(solvent_common_name) LIKE '%' || LOWER(?) || '%'
            LIMIT 1
            """
            result = conn.execute(query, [gsk_name, gsk_name]).fetchdf()
            if len(result) > 0:
                row = result.iloc[0]
                props['g_score'] = row.get('g_score')
                props['gsk_class'] = row.get('classification')
        except Exception as e:
            logger.debug(f"GSK lookup failed for {solvent_name}: {e}")

    return props


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class ValidationResult:
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
    is_feasible: bool
    conditions: Dict[str, Any] = field(default_factory=dict)
    selectivity: float = 0.0
    confidence: float = 0.0
    alternative_conditions: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ThresholdSearchResult:
    found: bool
    threshold_used: float
    results: List[Dict] = field(default_factory=list)
    thresholds_tried: List[float] = field(default_factory=list)
    search_path: str = ""


# ------------------------------------------------------------------
# DataValidator
# ------------------------------------------------------------------

class DataValidator:
    """Data validation and verification with caching."""

    def __init__(self, db_connection):
        self.conn = db_connection
        self._schema_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 60

    def _get_cached_schema(self, table_name: str):
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
        self._schema_cache.clear()
        self._cache_timestamps.clear()

    def verify_table_exists(self, table_name: str) -> ValidationResult:
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
        result = ValidationResult(is_valid=True)
        try:
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
            del df
            gc.collect()
        except Exception as e:
            result.add_issue(f"Query validation failed: {e}")
        return result

    def verify_numeric_range(self, table_name: str, column_name: str,
                            min_val: Optional[float] = None,
                            max_val: Optional[float] = None) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        try:
            stats_query = f"""
            SELECT MIN({column_name}) as min_val, MAX({column_name}) as max_val,
                   AVG({column_name}) as avg_val, STDDEV({column_name}) as std_val
            FROM {table_name} WHERE {column_name} IS NOT NULL
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


# ------------------------------------------------------------------
# AdaptiveAnalyzer
# ------------------------------------------------------------------

class AdaptiveAnalyzer:
    """Intelligent adaptive analysis with threshold searching and temperature exploration."""

    SELECTIVITY_THRESHOLDS = [50, 30, 20, 15, 10, 5, 2, 1, 0.5, 0.1]
    SOLUBILITY_THRESHOLDS = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
    TEMPERATURE_STEPS = [25, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150]

    def __init__(self, db_connection, validator: DataValidator):
        self.conn = db_connection
        self.validator = validator

    def find_threshold_with_results(self, query_func, thresholds: List[float],
                                    min_results: int = 1,
                                    prefer_stringent: bool = True) -> ThresholdSearchResult:
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

    def explore_temperature_range(self, table_name: str, polymer_column: str,
                                  solvent_column: str, temperature_column: str,
                                  solubility_column: str, target_polymer: str,
                                  comparison_polymers: List[str],
                                  start_temp: float = 25,
                                  min_selectivity: float = 10.0,
                                  max_temp: float = 160.0) -> Dict[str, Any]:
        results = {
            'optimal_conditions': None,
            'all_conditions': [],
            'temperatures_explored': [],
            'recommendation': ''
        }
        if isinstance(comparison_polymers, str):
            comparison_polymers = [p.strip() for p in comparison_polymers.split(',') if p.strip()]
        elif not isinstance(comparison_polymers, list):
            comparison_polymers = list(comparison_polymers) if comparison_polymers else []
        if not comparison_polymers:
            results['recommendation'] = "No comparison polymers provided"
            return results

        from strap.solubility import (
            FITTED_TEMP_MAX_C,
            FITTED_TEMP_MIN_C,
            RECOMMENDED_EXTRAPOLATION_MAX_C,
            SENSITIVITY_EXTRAPOLATION_MAX_C,
            get_solubility,
            get_available_solvents,
            temperature_extrapolation_status,
        )

        effective_max_temp = min(float(max_temp), SENSITIVITY_EXTRAPOLATION_MAX_C)
        # Temperature bins from interpolation model range, with optional
        # Apelblat extrapolation to 200 C when requested.
        temp_bins = [
            float(t)
            for t in range(
                max(int(start_temp), int(FITTED_TEMP_MIN_C)),
                int(effective_max_temp) + 1,
                10,
            )
        ]
        if not temp_bins:
            temp_bins = [float(t) for t in self.TEMPERATURE_STEPS if t >= start_temp]

        all_solvents = get_available_solvents()

        best_selectivity = -float('inf')
        for temp in temp_bins:
            results['temperatures_explored'].append(temp)
            all_polymers = [target_polymer] + comparison_polymers

            # Average solubility across all solvents for each polymer at this temp
            data = {}
            for poly in all_polymers:
                sols = []
                for sv in all_solvents:
                    sol = get_solubility(poly, sv, temp)
                    if sol is not None:
                        sols.append(sol)
                if sols:
                    data[poly.upper()] = float(np.mean(sols))

            target_sol = data.get(target_polymer.upper(), 0)
            other_sols = [data.get(p.upper(), 0) for p in comparison_polymers]
            max_other = max(other_sols) if other_sols else 0
            selectivity = target_sol - max_other

            condition = {
                'temperature': temp,
                'temperature_extrapolation': temperature_extrapolation_status(temp),
                'selectivity': selectivity,
                'target_solubility': target_sol,
                'max_other_solubility': max_other,
            }
            results['all_conditions'].append(condition)

            if selectivity > best_selectivity:
                best_selectivity = selectivity
                results['optimal_conditions'] = condition

        if results['optimal_conditions']:
            opt = results['optimal_conditions']
            if opt['selectivity'] >= min_selectivity:
                results['recommendation'] = (
                    f"Optimal at {opt['temperature']}°C with selectivity "
                    f"{opt['selectivity']:.1f}% (viable)"
                )
            else:
                results['recommendation'] = (
                    f"Best found at {opt['temperature']}°C with selectivity "
                    f"{opt['selectivity']:.1f}% (below {min_selectivity}% threshold)"
                )
        else:
            results['recommendation'] = "No data found for specified conditions"

        if any(t > FITTED_TEMP_MAX_C for t in temp_bins):
            results['recommendation'] += (
                f" Temperatures above {FITTED_TEMP_MAX_C:.0f}°C are Apelblat extrapolations "
                "outside the fitted range and should be treated as lower-confidence estimates."
            )
        if any(t > RECOMMENDED_EXTRAPOLATION_MAX_C for t in temp_bins):
            results['recommendation'] += (
                f" Temperatures above {RECOMMENDED_EXTRAPOLATION_MAX_C:.0f}°C are sensitivity-only screening data."
            )

        return results


# ------------------------------------------------------------------
# Polymer / solvent list helpers
# ------------------------------------------------------------------

def parse_polymer_list(polymers: str) -> List[str]:
    return [p.strip() for p in polymers.split(",") if p.strip()]


def parse_solvent_list(solvents: str) -> List[str]:
    return [s.strip() for s in solvents.split(",") if s.strip()]
