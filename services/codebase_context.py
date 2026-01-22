"""
Codebase context provider for AI diagnosis.

Loads pre-bundled codebase context from codebase_context.json
generated during build time.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Path to bundled codebase context
CONTEXT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codebase_context.json")


class CodebaseContextProvider:
    """Provides codebase context for AI diagnosis."""

    def __init__(self, context_file: str = CONTEXT_FILE):
        self.context_file = context_file
        self._context: Optional[Dict[str, Any]] = None
        self._loaded = False

    def load(self) -> bool:
        """Load the bundled codebase context."""
        if self._loaded:
            return True

        try:
            if not os.path.exists(self.context_file):
                logger.warning(f"Codebase context file not found: {self.context_file}")
                return False

            with open(self.context_file, 'r', encoding='utf-8') as f:
                self._context = json.load(f)

            self._loaded = True
            logger.info(f"Loaded codebase context: {len(self._context.get('files', {}))} files")
            return True

        except Exception as e:
            logger.error(f"Failed to load codebase context: {e}")
            return False

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        if not self._loaded:
            self.load()

        if not self._context:
            return None

        files = self._context.get("files", {})
        return files.get(file_path)

    def get_all_files(self) -> Dict[str, str]:
        """Get all file contents."""
        if not self._loaded:
            self.load()

        if not self._context:
            return {}

        return self._context.get("files", {})

    def get_file_list(self) -> List[str]:
        """Get list of all bundled files."""
        if not self._loaded:
            self.load()

        if not self._context:
            return []

        return list(self._context.get("files", {}).keys())

    def get_tools(self) -> List[str]:
        """Get list of extracted tool names."""
        if not self._loaded:
            self.load()

        if not self._context:
            return []

        return self._context.get("tools", [])

    def get_endpoints(self) -> List[str]:
        """Get list of extracted API endpoints."""
        if not self._loaded:
            self.load()

        if not self._context:
            return []

        return self._context.get("endpoints", [])

    def get_context_for_issue_type(self, issue_type: str) -> Dict[str, str]:
        """Get relevant files based on issue type."""
        all_files = self.get_all_files()

        # Define relevance mapping
        relevance_map = {
            "incorrect_response": ["agent_sql_final_1212_patched.py", "app_server.py"],
            "ui_bug": ["frontend/src/App.js"],
            "api_error": ["app_server.py", "agent_sql_final_1212_patched.py"],
            "performance": ["agent_sql_final_1212_patched.py", "app_server.py"],
            "data_issue": ["agent_sql_final_1212_patched.py"],
            "other": list(all_files.keys())[:5],  # Return first 5 files for general issues
        }

        # Get relevant file patterns
        patterns = relevance_map.get(issue_type, relevance_map["other"])

        # Filter files
        relevant_files = {}
        for file_path, content in all_files.items():
            for pattern in patterns:
                if pattern in file_path:
                    relevant_files[file_path] = content
                    break

        # If no matches, return all files
        if not relevant_files:
            return all_files

        return relevant_files

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the bundled codebase."""
        if not self._loaded:
            self.load()

        if not self._context:
            return {"error": "Context not loaded"}

        files = self._context.get("files", {})

        return {
            "total_files": len(files),
            "files": list(files.keys()),
            "tools_count": len(self._context.get("tools", [])),
            "endpoints_count": len(self._context.get("endpoints", [])),
            "bundled_at": self._context.get("bundled_at", "unknown"),
        }


# Global singleton instance
_provider: Optional[CodebaseContextProvider] = None


def get_codebase_context() -> CodebaseContextProvider:
    """Get the global codebase context provider."""
    global _provider
    if _provider is None:
        _provider = CodebaseContextProvider()
    return _provider
