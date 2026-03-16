from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from output_models import ExportMetadata

logger = logging.getLogger(__name__)


class ExportManager:
    """Manage short-lived CSV exports for tool results and session downloads."""

    def __init__(self, export_dir: str | os.PathLike[str] = "./exports", ttl_minutes: int = 30):
        self.export_dir = Path(export_dir)
        self.ttl = timedelta(minutes=ttl_minutes)
        self.exports: dict[str, ExportMetadata] = {}
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def create_export(
        self,
        data: list[dict[str, Any]],
        tool_name: str,
        columns: list[str] | None = None,
    ) -> str:
        export_id = uuid.uuid4().hex[:8]
        timestamp = datetime.utcnow()

        df = pd.DataFrame(data)
        if columns:
            ordered = [col for col in columns if col in df.columns]
            extras = [col for col in df.columns if col not in ordered]
            df = df[ordered + extras]

        filename = f"{tool_name}_{export_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.export_dir / filename
        df.to_csv(filepath, index=False)

        metadata = ExportMetadata(
            export_id=export_id,
            tool_name=tool_name,
            created_at=timestamp.isoformat(),
            expires_at=(timestamp + self.ttl).isoformat(),
            row_count=len(df),
            file_path=str(filepath),
        )
        self.exports[export_id] = metadata
        return export_id

    def get_export_path(self, export_id: str) -> str | None:
        metadata = self.exports.get(export_id)
        if metadata is None:
            return None

        expires = datetime.fromisoformat(metadata.expires_at)
        if datetime.utcnow() > expires:
            self.cleanup_export(export_id)
            return None

        return metadata.file_path if Path(metadata.file_path).exists() else None

    def cleanup_export(self, export_id: str) -> bool:
        metadata = self.exports.pop(export_id, None)
        if metadata is None:
            return False
        try:
            path = Path(metadata.file_path)
            if path.exists():
                path.unlink()
            return True
        except OSError as exc:
            logger.warning("Failed to clean export %s: %s", export_id, exc)
            return False

    def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        expired = [
            export_id
            for export_id, metadata in self.exports.items()
            if datetime.fromisoformat(metadata.expires_at) < now
        ]
        cleaned = 0
        for export_id in expired:
            cleaned += int(self.cleanup_export(export_id))
        return cleaned


export_manager = ExportManager(export_dir=os.environ.get("EXPORTS_DIR", "./exports"))
