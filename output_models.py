from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportMetadata:
    export_id: str
    tool_name: str
    created_at: str
    expires_at: str
    row_count: int
    file_path: str
