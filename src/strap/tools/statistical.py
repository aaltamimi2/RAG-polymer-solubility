"""Statistical analysis tools for STRAP.

Provides: statistical_summary, correlation_analysis,
compare_groups_statistically (async), regression_analysis.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from strap.database import get_connection
from strap.services.tool_response_service import json_tool_error, json_tool_response
from strap.tools._helpers import safe_tool_wrapper, truncate_output, save_plot, get_plots_dir

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_DANGEROUS_KEYWORDS = [
    "drop", "delete", "insert", "update", "alter", "create", "truncate",
    "copy", "attach", "export", "load", "import",
]


def _check_filters(filters: Optional[str]) -> Optional[str]:
    """Return an error string if *filters* contains a blocklisted keyword, else None."""
    if not filters:
        return None
    filters_lower = filters.lower().strip()
    if any(kw in filters_lower.split() for kw in _DANGEROUS_KEYWORDS):
        return f"Unsafe keyword detected in filters: '{filters}'"
    return None


def _sanitize_identifier(conn, table_name: str, column_name: Optional[str] = None) -> Optional[str]:
    """Validate *table_name* (and optionally *column_name*) against the live schema.

    Returns an error string if validation fails, else None.
    """
    try:
        tables_df = conn.execute("SHOW TABLES").fetchdf()
        valid_tables = set(tables_df["name"].values)
        if table_name not in valid_tables:
            return f"Table '{table_name}' not found. Available: {sorted(valid_tables)}"
        if column_name is not None:
            schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
            valid_columns = set(schema_df["column_name"].values)
            if column_name not in valid_columns:
                return (
                    f"Column '{column_name}' not found in '{table_name}'. "
                    f"Available: {sorted(valid_columns)}"
                )
    except Exception as e:
        return f"Schema validation error: {e}"
    return None


def _execute_query(query: str, limit: int = 100) -> dict:
    """Execute a read-only query via the shared DuckDB connection.

    Returns a dict compatible with the legacy ``sql_db.execute_query`` API
    used by the original monolith.
    """
    try:
        query_lower = query.lower().strip()
        if any(kw in query_lower.split() for kw in _DANGEROUS_KEYWORDS):
            return {"success": False, "error": "Unsafe operation detected", "query": query}

        if "limit" not in query_lower and not query_lower.strip().endswith(";"):
            query = f"{query.rstrip(';')} LIMIT {limit}"

        conn = get_connection()
        result_df = conn.execute(query).fetchdf()

        preview = (
            result_df.head(10).to_markdown(index=False) if len(result_df) > 0 else "No data"
        )

        return {
            "success": True,
            "query": query,
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "data": result_df.to_dict("records"),
            "dataframe": result_df,
            "preview": preview,
            "dtypes": {str(k): str(v) for k, v in result_df.dtypes.to_dict().items()},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


def _get_plot_url(filepath: str) -> str:
    """Convert a filepath to a displayable string."""
    return f"Plot saved: `{filepath}`"


def _tool_error(tool_name: str, message: str, *, error_code: str = "invalid_input") -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code)


def _tool_success(tool_name: str, display: str, **data) -> str:
    return json_tool_response(display, data, tool_name=tool_name, success=True)


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

@safe_tool_wrapper(structured_output=True)
def statistical_summary(
    table_name: str,
    value_column: str,
    group_by_column: Optional[str] = None,
    filters: Optional[str] = None,
) -> str:
    """Compute descriptive statistics (mean, std, quartiles) with 95% confidence intervals.

    Args:
        table_name: Database table name
        value_column: Numeric column to summarize
        group_by_column: Optional column to group results by
        filters: Optional SQL WHERE clause filter

    WHEN TO USE:
    - "Give me summary statistics for solubility"
    - "What is the average solubility grouped by polymer?"
    """
    tool_name = "statistical_summary"
    conn = get_connection()
    err = _sanitize_identifier(conn, table_name, value_column)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    if group_by_column is not None:
        err = _sanitize_identifier(conn, table_name, group_by_column)
        if err:
            err_msg = f"Input validation failed: {err}"
            return _tool_error(tool_name, err_msg)
    filter_err = _check_filters(filters)
    if filter_err:
        err_msg = f"Input validation failed: {filter_err}"
        return _tool_error(tool_name, err_msg)

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

    result = _execute_query(query, limit=1000)
    if not result["success"]:
        err_msg = f"Query failed: {result.get('error')}"
        return _tool_error(tool_name, err_msg, error_code="query_failed")

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
                output.append(f"  - {row[group_by_column]}: {row['mean']:.4f} +/- {ci:.4f}")
    else:
        if df['n'].iloc[0] > 1 and df['std'].iloc[0] is not None and not pd.isna(df['std'].iloc[0]):
            ci = 1.96 * df['std'].iloc[0] / np.sqrt(df['n'].iloc[0])
            output.append(f"\n**95% CI:** {df['mean'].iloc[0]:.4f} +/- {ci:.4f}")

    if group_by_column:
        groups_list = []
        for _, row in df.iterrows():
            groups_list.append({
                "group": str(row[group_by_column]),
                "n": int(row['n']),
                "mean": float(row['mean']),
                "std": float(row['std']) if not pd.isna(row['std']) else None,
            })
        data_dict = {
            "success": True, "table_name": table_name, "value_column": value_column,
            "group_by_column": group_by_column, "filters": filters,
            "n": int(df['n'].sum()), "mean": None, "groups": groups_list,
        }
    else:
        data_dict = {
            "success": True, "table_name": table_name, "value_column": value_column,
            "group_by_column": None, "filters": filters,
            "n": int(df['n'].iloc[0]),
            "mean": float(df['mean'].iloc[0]),
            "std": float(df['std'].iloc[0]) if not pd.isna(df['std'].iloc[0]) else None,
            "min": float(df['min'].iloc[0]),
            "median": float(df['median'].iloc[0]),
            "max": float(df['max'].iloc[0]),
            "groups": None,
        }

    del df
    return _tool_success(tool_name, "\n".join(output), **data_dict)


@safe_tool_wrapper(structured_output=True)
def correlation_analysis(
    table_name: str,
    columns: str,
    filters: Optional[str] = None,
    method: str = "pearson",
) -> str:
    """Compute pairwise correlations between columns and generate a heatmap.

    Args:
        table_name: Database table name
        columns: Comma-separated column names to correlate
        filters: Optional SQL WHERE clause filter
        method: Correlation method ('pearson', 'spearman', or 'kendall')

    WHEN TO USE:
    - "Is there a correlation between temperature and solubility?"
    - "Show the correlation matrix for these columns"
    """
    tool_name = "correlation_analysis"
    if method not in ("pearson", "spearman", "kendall"):
        err_msg = f"Error: method must be 'pearson', 'spearman', or 'kendall' (got '{method}')."
        return _tool_error(tool_name, err_msg)
    conn = get_connection()
    col_list = [c.strip() for c in columns.split(',')]
    err = _sanitize_identifier(conn, table_name)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    for col in col_list:
        err = _sanitize_identifier(conn, table_name, col)
        if err:
            err_msg = f"Input validation failed: {err}"
            return _tool_error(tool_name, err_msg)
    filter_err = _check_filters(filters)
    if filter_err:
        err_msg = f"Input validation failed: {filter_err}"
        return _tool_error(tool_name, err_msg)

    where_clause = f"WHERE {filters}" if filters else ""

    query = f"SELECT {', '.join(col_list)} FROM {table_name} {where_clause}"
    result = _execute_query(query, limit=100000)

    if not result["success"]:
        err_msg = f"Query failed: {result.get('error')}"
        return _tool_error(tool_name, err_msg, error_code="query_failed")

    df = result["dataframe"].dropna()

    if len(df) < 3:
        err_msg = f"Insufficient data for correlation analysis (n={len(df)})"
        return _tool_error(tool_name, err_msg, error_code="insufficient_data")

    corr_matrix = df.corr(method=method)

    output = [f"**Correlation Analysis ({method.title()})**\n"]
    output.append(f"Data points: {len(df)}\n")
    output.append("**Correlation Matrix:**")
    output.append(corr_matrix.round(3).to_markdown())

    output.append("\n**Significant Correlations (p < 0.05):**")
    significant_pairs = []
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
                    significant_pairs.append({"col1": col1, "col2": col2, "r": float(r), "p_value": float(p), "strength": strength})
            except Exception:
                pass

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f'{method.title()} Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = save_plot(fig, "correlation_matrix", "matplotlib")
    output.append(f"\n{_get_plot_url(filepath)}")

    data_dict = {
        "success": True, "table_name": table_name, "columns": col_list,
        "method": method, "n": len(df),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "significant_pairs": significant_pairs,
        "plot_filepath": filepath,
    }

    del df
    return _tool_success(tool_name, "\n".join(output), **data_dict)


@safe_tool_wrapper(structured_output=True)
async def compare_groups_statistically(
    table_name: str,
    value_column: str,
    group_column: str,
    group1: str,
    group2: str,
    filters: Optional[str] = None,
) -> str:
    """Compare two groups using t-test, Mann-Whitney U, and Cohen's d effect size.

    Args:
        table_name: Database table name
        value_column: Numeric column to compare
        group_column: Column defining the groups
        group1: First group name
        group2: Second group name
        filters: Optional SQL WHERE clause filter

    WHEN TO USE:
    - "Is solubility significantly different between PET and PS?"
    - "Compare solubility distributions of two polymers"
    """
    tool_name = "compare_groups_statistically"
    conn = get_connection()
    err = _sanitize_identifier(conn, table_name, value_column)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    err = _sanitize_identifier(conn, table_name, group_column)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    filter_err = _check_filters(filters)
    if filter_err:
        err_msg = f"Input validation failed: {filter_err}"
        return _tool_error(tool_name, err_msg)

    where_clause = f"WHERE {filters} AND" if filters else "WHERE"

    # Use parameterized queries for group1/group2 to prevent SQL injection
    base_query = (
        f"SELECT {value_column} FROM {table_name} {where_clause} "
        f"LOWER({group_column}) = LOWER(?)"
    )

    # PARALLEL EXECUTION - Run both queries concurrently via thread pool
    loop = asyncio.get_event_loop()
    try:
        df1, df2 = await asyncio.gather(
            loop.run_in_executor(None, lambda: conn.execute(base_query, [group1]).fetchdf()),
            loop.run_in_executor(None, lambda: conn.execute(base_query, [group2]).fetchdf()),
        )
    except Exception as e:
        err_msg = f"Query failed: {str(e)[:300]}"
        return _tool_error(tool_name, err_msg, error_code="query_failed")

    if len(df1) == 0 or len(df2) == 0:
        err_msg = f"No data returned for groups: {group1} ({len(df1)} rows), {group2} ({len(df2)} rows)"
        return _tool_error(tool_name, err_msg, error_code="no_data")

    data1 = df1[value_column].dropna()
    data2 = df2[value_column].dropna()

    if len(data1) < 3 or len(data2) < 3:
        err_msg = f"Insufficient data: {group1} has {len(data1)}, {group2} has {len(data2)} samples"
        return _tool_error(tool_name, err_msg, error_code="insufficient_data")

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
        output.append(f"  Significant difference (p < 0.05): {group1} has {direction} values")
    else:
        output.append(f"  No significant difference (p = {t_p:.4f})")

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
    output.append(f"\n{_get_plot_url(filepath)}")

    data_dict = {
        "success": True, "table_name": table_name,
        "value_column": value_column, "group_column": group_column,
        "group1": {"name": group1, "n": int(len(data1)), "mean": float(data1.mean()), "std": float(data1.std())},
        "group2": {"name": group2, "n": int(len(data2)), "mean": float(data2.mean()), "std": float(data2.std())},
        "t_test": {"t_statistic": float(t_stat), "p_value": float(t_p), "significant": bool(t_p < 0.05)},
        "mann_whitney": {"u_statistic": float(u_stat), "p_value": float(u_p), "significant": bool(u_p < 0.05)},
        "cohens_d": float(cohens_d), "effect_size": effect_size,
        "plot_filepath": filepath,
    }
    return _tool_success(tool_name, "\n".join(output), **data_dict)


@safe_tool_wrapper(structured_output=True)
def regression_analysis(
    table_name: str,
    x_column: str,
    y_column: str,
    group_by: Optional[str] = None,
    filters: Optional[str] = None,
    degree: int = 1,
) -> str:
    """Fit a polynomial regression model and produce scatter + residual plots.

    Args:
        table_name: Database table name
        x_column: Independent variable column
        y_column: Dependent variable column
        group_by: Optional column to fit separate regressions per group
        filters: Optional SQL WHERE clause filter
        degree: Polynomial degree (default 1 for linear)

    WHEN TO USE:
    - "Fit a regression of solubility vs temperature"
    - "Is there a linear relationship between temperature and solubility?"
    """
    tool_name = "regression_analysis"
    if degree < 1 or degree > 5:
        err_msg = f"Error: degree must be between 1 and 5 (got {degree}). High-degree polynomials overfit on small datasets."
        return _tool_error(tool_name, err_msg)
    conn = get_connection()
    err = _sanitize_identifier(conn, table_name, x_column)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    err = _sanitize_identifier(conn, table_name, y_column)
    if err:
        err_msg = f"Input validation failed: {err}"
        return _tool_error(tool_name, err_msg)
    if group_by is not None:
        err = _sanitize_identifier(conn, table_name, group_by)
        if err:
            err_msg = f"Input validation failed: {err}"
            return _tool_error(tool_name, err_msg)
    filter_err = _check_filters(filters)
    if filter_err:
        err_msg = f"Input validation failed: {filter_err}"
        return _tool_error(tool_name, err_msg)

    where_clause = f"WHERE {filters}" if filters else ""

    if group_by:
        query = f"SELECT {x_column}, {y_column}, {group_by} FROM {table_name} {where_clause}"
    else:
        query = f"SELECT {x_column}, {y_column} FROM {table_name} {where_clause}"

    result = _execute_query(query, limit=100000)
    if not result["success"]:
        err_msg = f"Query failed: {result.get('error')}"
        return _tool_error(tool_name, err_msg, error_code="query_failed")

    df = result["dataframe"].dropna()

    output = [f"**Regression Analysis: {y_column} ~ {x_column}**\n"]
    output.append(f"Model: Polynomial degree {degree}")
    output.append(f"Data points: {len(df)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

        output.append("**Regression Results by Group:**\n")

        group_results = []
        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group]
            x = group_data[x_column].values
            y = group_data[y_column].values

            if len(x) < degree + 1:
                continue

            if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isinf(x)) or np.any(np.isinf(y)):
                continue

            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            output.append(f"**{group}:** R²={r2:.4f}, RMSE={rmse:.4f}")
            group_results.append({"group": str(group), "n": len(x), "r_squared": float(r2), "rmse": float(rmse)})

            axes[0].scatter(x, y, alpha=0.5, color=colors[i], label=f'{group} (R²={r2:.3f})')
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[0].plot(x_line, poly(x_line), color=colors[i], linewidth=2)

        axes[0].legend(fontsize=9)
    else:
        x = df[x_column].values
        y = df[y_column].values

        if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isinf(x)) or np.any(np.isinf(y)):
            err_msg = "Error: data contains NaN or infinite values. Clean the data before running regression."
            return _tool_error(tool_name, err_msg, error_code="invalid_data")

        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        slope = intercept = p_value = std_err = None
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
    output.append(f"\n{_get_plot_url(filepath)}")

    if group_by and group_by in df.columns:
        data_dict = {
            "success": True, "table_name": table_name,
            "x_column": x_column, "y_column": y_column,
            "degree": degree, "n": len(df),
            "r_squared": None, "rmse": None, "slope": None,
            "group_results": group_results, "plot_filepath": filepath,
        }
    else:
        data_dict = {
            "success": True, "table_name": table_name,
            "x_column": x_column, "y_column": y_column,
            "degree": degree, "n": len(df),
            "r_squared": float(r2), "rmse": float(rmse),
            "slope": float(slope) if slope is not None else None,
            "intercept": float(intercept) if intercept is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "group_results": None, "plot_filepath": filepath,
        }

    del df
    return _tool_success(tool_name, "\n".join(output), **data_dict)
