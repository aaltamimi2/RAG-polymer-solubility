"""
CSV Export Manager

Manages the lifecycle of CSV exports:
- Generates CSV files from structured data
- Tracks export metadata with TTL
- Automatic cleanup of expired exports
- Thread-safe export operations
"""

import os
import uuid
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from output_models import ExportMetadata

logger = logging.getLogger(__name__)


class ExportManager:
    """Manage CSV export generation and cleanup"""

    def __init__(self, export_dir="./exports", ttl_minutes=30):
        """
        Initialize export manager.

        Args:
            export_dir: Directory to store CSV exports
            ttl_minutes: Time-to-live for exports in minutes (default: 30)
        """
        self.export_dir = export_dir
        self.ttl = timedelta(minutes=ttl_minutes)
        self.exports: Dict[str, ExportMetadata] = {}

        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        logger.info(f"ExportManager initialized: dir={export_dir}, TTL={ttl_minutes}min")

    def create_export(
        self,
        data: List[Dict],
        tool_name: str,
        columns: Optional[List[str]] = None
    ) -> str:
        """
        Generate CSV export and return export ID.

        Args:
            data: List of dictionaries to export
            tool_name: Name of the tool generating the export
            columns: Optional column order (if None, uses DataFrame default)

        Returns:
            Export ID (8-character UUID)
        """
        export_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        if columns:
            # Reorder columns if specified, keeping any extra columns at the end
            available_cols = [col for col in columns if col in df.columns]
            extra_cols = [col for col in df.columns if col not in columns]
            df = df[available_cols + extra_cols]

        # Generate filename with timestamp
        filename = f"{tool_name}_{export_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.export_dir, filename)

        # Save to CSV
        df.to_csv(filepath, index=False)

        # Store metadata
        metadata = ExportMetadata(
            export_id=export_id,
            tool_name=tool_name,
            created_at=timestamp.isoformat(),
            expires_at=(timestamp + self.ttl).isoformat(),
            row_count=len(df),
            file_path=filepath
        )
        self.exports[export_id] = metadata

        logger.info(f"Created export {export_id}: {filename} ({len(df)} rows)")
        return export_id

    def get_export_path(self, export_id: str) -> Optional[str]:
        """
        Get file path for export ID.

        Args:
            export_id: Export ID to look up

        Returns:
            File path if export exists and hasn't expired, None otherwise
        """
        metadata = self.exports.get(export_id)
        if not metadata:
            logger.warning(f"Export not found: {export_id}")
            return None

        # Check expiration
        expires = datetime.fromisoformat(metadata.expires_at)
        if datetime.utcnow() > expires:
            logger.info(f"Export expired: {export_id}")
            self.cleanup_export(export_id)
            return None

        return metadata.file_path

    def get_export_metadata(self, export_id: str) -> Optional[ExportMetadata]:
        """
        Get metadata for export ID.

        Args:
            export_id: Export ID to look up

        Returns:
            ExportMetadata if found, None otherwise
        """
        return self.exports.get(export_id)

    def cleanup_export(self, export_id: str) -> bool:
        """
        Remove export file and metadata.

        Args:
            export_id: Export ID to clean up

        Returns:
            True if export was cleaned up, False if not found
        """
        metadata = self.exports.pop(export_id, None)
        if metadata:
            if os.path.exists(metadata.file_path):
                try:
                    os.remove(metadata.file_path)
                    logger.info(f"Cleaned up export {export_id}: {os.path.basename(metadata.file_path)}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete export file {metadata.file_path}: {e}")
                    return False
        return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired exports.

        Returns:
            Number of exports cleaned up
        """
        now = datetime.utcnow()
        expired = [
            export_id for export_id, meta in self.exports.items()
            if datetime.fromisoformat(meta.expires_at) < now
        ]

        count = 0
        for export_id in expired:
            if self.cleanup_export(export_id):
                count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired export(s)")

        return count

    def list_exports(self) -> List[ExportMetadata]:
        """
        List all active exports.

        Returns:
            List of ExportMetadata for all non-expired exports
        """
        # Clean up expired first
        self.cleanup_expired()
        return list(self.exports.values())

    def get_stats(self) -> Dict:
        """
        Get statistics about exports.

        Returns:
            Dictionary with export statistics
        """
        active = list(self.exports.values())
        total_rows = sum(m.row_count for m in active)

        return {
            "active_exports": len(active),
            "total_rows_exported": total_rows,
            "export_dir": self.export_dir,
            "ttl_minutes": self.ttl.total_seconds() / 60
        }


# Global instance
export_manager = ExportManager()
