"""
Codebase context loader for issue diagnosis.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTEXT_FILE = ROOT_DIR / "codebase_context.json"

RELEVANCE_MAP: dict[str, list[str]] = {
    "incorrect_response": [
        "app_server.py",
        "src/strap/agent.py",
        "src/strap/routing_classifier.py",
        "src/strap/routing_guards.py",
        "src/strap/routing_message_state.py",
        "src/strap/handoff_adapters.py",
        "src/strap/query_context.py",
    ],
    "ui_bug": [
        "frontend/src/App.js",
        "frontend/src/api.js",
        "frontend/src/index.css",
        "app_server.py",
    ],
    "api_error": [
        "app_server.py",
        "src/strap/agent.py",
        "src/strap/routing_guards.py",
        "src/strap/result_extractor.py",
    ],
    "performance": [
        "app_server.py",
        "src/strap/agent.py",
        "src/strap/routing_progress.py",
        "src/strap/langsmith_tracing.py",
    ],
    "data_issue": [
        "src/strap/database.py",
        "src/strap/query_context.py",
        "src/strap/services/contaminant_screening_service.py",
        "src/strap/tools/_helpers.py",
    ],
    "other": [
        "app_server.py",
        "frontend/src/App.js",
        "src/strap/agent.py",
        "src/strap/routing_classifier.py",
    ],
}


class CodebaseContextProvider:
    def __init__(self, context_file: Path = CONTEXT_FILE):
        self.context_file = context_file
        self._context: dict[str, Any] | None = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            if self.context_file.exists():
                with self.context_file.open("r", encoding="utf-8") as fh:
                    self._context = json.load(fh)
                logger.info("Loaded bundled codebase context from %s", self.context_file)
            else:
                self._context = self._build_live_context()
                logger.info("Built live codebase context fallback with %s files", len(self._context.get("files", {})))
            self._loaded = True
            return True
        except Exception as exc:
            logger.warning("Failed to load codebase context: %s", exc)
            self._context = {"files": {}, "tools": [], "endpoints": []}
            self._loaded = True
            return False

    def get_all_files(self) -> dict[str, str]:
        if not self._loaded:
            self.load()
        return dict(self._context.get("files", {})) if self._context else {}

    def get_file_list(self) -> list[str]:
        return list(self.get_all_files().keys())

    def get_tools(self) -> list[str]:
        if not self._loaded:
            self.load()
        if self._context and self._context.get("tools"):
            return list(self._context["tools"])
        return self._load_tools_from_runtime()

    def _load_tools_from_runtime(self) -> list[str]:
        try:
            from strap.tools import get_all_tools

            return sorted({tool.name for tool in get_all_tools() if getattr(tool, "name", None)})
        except Exception as exc:
            logger.debug("Unable to enumerate tools for report context: %s", exc)
            return []

    def get_endpoints(self) -> list[str]:
        if not self._loaded:
            self.load()
        if self._context and self._context.get("endpoints"):
            return list(self._context["endpoints"])
        return self._extract_endpoints()

    def get_context_for_issue_type(self, issue_type: str) -> dict[str, str]:
        all_files = self.get_all_files()
        patterns = RELEVANCE_MAP.get(issue_type, RELEVANCE_MAP["other"])
        relevant_files: dict[str, str] = {}
        for file_path, content in all_files.items():
            if any(pattern in file_path for pattern in patterns):
                relevant_files[file_path] = content
        return relevant_files or all_files

    def get_summary(self) -> dict[str, Any]:
        files = self.get_all_files()
        return {
            "total_files": len(files),
            "files": list(files.keys()),
            "tools_count": len(self.get_tools()),
            "endpoints_count": len(self.get_endpoints()),
            "source": "bundle" if self.context_file.exists() else "live_fallback",
        }

    def _build_live_context(self) -> dict[str, Any]:
        files: dict[str, str] = {}
        relevant_paths = sorted({path for paths in RELEVANCE_MAP.values() for path in paths})
        for relative_path in relevant_paths:
            path = ROOT_DIR / relative_path
            if not path.exists() or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="replace")
            files[relative_path] = content[:20000]
        return {
            "files": files,
            "tools": self._load_tools_from_runtime(),
            "endpoints": self._extract_endpoints(),
        }

    def _extract_endpoints(self) -> list[str]:
        app_server_path = ROOT_DIR / "app_server.py"
        if not app_server_path.exists():
            return []
        content = app_server_path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r'@app\.(?:get|post|put|patch|delete)\("([^"]+)"')
        return sorted({match.group(1) for match in pattern.finditer(content)})


_provider: CodebaseContextProvider | None = None


def get_codebase_context() -> CodebaseContextProvider:
    global _provider
    if _provider is None:
        _provider = CodebaseContextProvider()
    return _provider
