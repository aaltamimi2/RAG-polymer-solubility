"""
Pydantic models for structured tool outputs and CSV exports.

This module defines the data structures for tool outputs, enabling:
- Structured data export to CSV
- Programmatic access to analysis results
- Type validation and serialization
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryResult(BaseModel):
    """Structured output for database queries"""
    success: bool
    query: str
    row_count: int
    columns: List[str]
    data: List[Dict[str, Any]]
    export_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "SELECT * FROM solvent_data LIMIT 10",
                "row_count": 10,
                "columns": ["polymer", "solvent", "temperature", "solubility"],
                "data": [{"polymer": "PVDF", "solvent": "DMF", "temperature": 25, "solubility": 45.2}],
                "export_id": "a3b4c5d6",
                "timestamp": "2025-01-15T10:30:00.000Z"
            }
        }


class SeparationResult(BaseModel):
    """Structured output for separation analysis"""
    is_feasible: bool
    target_polymer: str
    contaminant_polymers: List[str]
    optimal_solvent: Optional[str]
    optimal_temperature: Optional[float]
    selectivity: float
    alternatives: List[Dict[str, Any]]
    export_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "is_feasible": True,
                "target_polymer": "PVDF",
                "contaminant_polymers": ["PET", "PP"],
                "optimal_solvent": "NMP",
                "optimal_temperature": 80.0,
                "selectivity": 35.5,
                "alternatives": [
                    {"solvent": "DMF", "temperature": 75.0, "selectivity": 32.1},
                    {"solvent": "DMSO", "temperature": 85.0, "selectivity": 28.7}
                ],
                "export_id": "b7c8d9e0",
                "timestamp": "2025-01-15T10:35:00.000Z"
            }
        }


class StatisticalResult(BaseModel):
    """Structured output for statistical tests"""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    n_samples: int
    interpretation: str
    export_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "Independent t-test",
                "test_statistic": 3.45,
                "p_value": 0.0012,
                "effect_size": 0.82,
                "confidence_interval": (-5.2, -1.8),
                "n_samples": 45,
                "interpretation": "Significant difference (p < 0.05)",
                "export_id": "c9d0e1f2",
                "timestamp": "2025-01-15T10:40:00.000Z"
            }
        }


class ExportMetadata(BaseModel):
    """Metadata for CSV exports"""
    export_id: str
    tool_name: str
    created_at: str
    expires_at: str
    row_count: int
    file_path: str

    class Config:
        json_schema_extra = {
            "example": {
                "export_id": "a3b4c5d6",
                "tool_name": "query_database",
                "created_at": "2025-01-15T10:30:00.000Z",
                "expires_at": "2025-01-15T11:00:00.000Z",
                "row_count": 150,
                "file_path": "./exports/query_database_a3b4c5d6_20250115_103000.csv"
            }
        }
