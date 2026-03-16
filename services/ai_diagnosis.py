"""
AI diagnosis for user-submitted issue reports.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ISSUE_REPORT_MODEL = os.environ.get("ISSUE_REPORT_MODEL", "gemini-3.1-pro-preview")


@dataclass
class DiagnosisResult:
    summary: str
    root_cause: str
    fix_category: str
    affected_files: list[str]
    proposed_changes: list[dict[str, Any]]
    additional_notes: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIDiagnosisService:
    """AI-powered diagnosis with structured code-change proposals."""

    def __init__(self, api_key: str = GOOGLE_API_KEY, model_name: str = ISSUE_REPORT_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured")
        if self._model is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.2,
                max_output_tokens=8192,
            )
            logger.info("Initialized diagnosis model %s", self.model_name)
        return self._model

    def diagnose(
        self,
        *,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        codebase_context: dict[str, str],
        tools: list[str] | None = None,
        endpoints: list[str] | None = None,
    ) -> DiagnosisResult:
        if not self.api_key:
            return DiagnosisResult(
                summary="AI diagnosis unavailable because GOOGLE_API_KEY is not configured.",
                root_cause="The reporting pipeline could not initialize a diagnosis model in this environment.",
                fix_category="informational",
                affected_files=[],
                proposed_changes=[],
                additional_notes=f"Manual triage required. User reported: {user_description}",
                confidence=0.0,
            )

        prompt = self._build_prompt(
            user_question=user_question,
            assistant_response=assistant_response,
            user_description=user_description,
            issue_type=issue_type,
            severity=severity,
            codebase_context=codebase_context,
            tools=tools or [],
            endpoints=endpoints or [],
        )
        try:
            response = self._get_model().invoke(prompt)
            return self._parse_response(self._coerce_content(response.content))
        except Exception as exc:
            logger.warning("AI diagnosis failed: %s", exc)
            return DiagnosisResult(
                summary=f"Automated diagnosis failed: {exc}",
                root_cause="The diagnosis model failed during analysis.",
                fix_category="informational",
                affected_files=[],
                proposed_changes=[],
                additional_notes=f"Manual triage required. User reported: {user_description}",
                confidence=0.0,
            )

    def _build_prompt(
        self,
        *,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        codebase_context: dict[str, str],
        tools: list[str],
        endpoints: list[str],
    ) -> str:
        codebase_sections: list[str] = []
        for file_path, content in codebase_context.items():
            excerpt = content[:15000]
            if len(content) > 15000:
                excerpt += "\n... [truncated]"
            codebase_sections.append(f"### {file_path}\n```python\n{excerpt}\n```")

        codebase_blob = "\n\n".join(codebase_sections) if codebase_sections else "No codebase context available."

        return f"""You are diagnosing a bug report for DISSOLVE v9, a multi-agent polymer separation and analysis system.

## Issue Report
- Issue type: {issue_type}
- Severity: {severity}

### User question
{user_question}

### Assistant response
{assistant_response}

### User description
{user_description}

## Runtime context
- Available tools: {", ".join(tools) if tools else "Unknown"}
- API endpoints: {", ".join(endpoints) if endpoints else "Unknown"}

## Relevant code
{codebase_blob}

## Output contract
Return only valid JSON with this exact shape:
{{
  "summary": "One-line summary",
  "root_cause": "Detailed explanation",
  "fix_category": "informational|simple_fix|complex_fix",
  "affected_files": ["path/to/file.py"],
  "proposed_changes": [
    {{
      "file": "path/to/file.py",
      "description": "What changes",
      "old_code": "exact code to replace or null",
      "new_code": "replacement code"
    }}
  ],
  "additional_notes": "Any caveats, manual follow-up, or testing notes",
  "confidence": 0.0
}}

Guidelines:
- Use "informational" if no safe code change can be proposed.
- Only include proposed_changes when you can point to a concrete file and concrete replacement/addition.
- Keep changes focused and realistic for this codebase.
- Confidence must be between 0 and 1.
"""

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content)

    def _parse_response(self, raw_response: str) -> DiagnosisResult:
        response = raw_response.strip()
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end if end != -1 else None].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end if end != -1 else None].strip()

        if not response.startswith("{"):
            start = response.find("{")
            if start != -1:
                response = response[start:]

        if response.count("{") > response.count("}"):
            response += "}" * (response.count("{") - response.count("}"))
        if response.count("[") > response.count("]"):
            response += "]" * (response.count("[") - response.count("]"))

        response = re.sub(r',\s*"[^"]*$', "", response)
        response = re.sub(r':\s*"[^"]*$', ': ""', response)

        try:
            data = json.loads(response)
            return DiagnosisResult(
                summary=data.get("summary", "No summary provided"),
                root_cause=data.get("root_cause", "Unknown"),
                fix_category=data.get("fix_category", "informational"),
                affected_files=list(data.get("affected_files", [])),
                proposed_changes=list(data.get("proposed_changes", [])),
                additional_notes=data.get("additional_notes", ""),
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception as exc:
            logger.warning("Failed to parse diagnosis response: %s", exc)
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response)
            root_cause_match = re.search(r'"root_cause"\s*:\s*"([^"]+)"', response)
            return DiagnosisResult(
                summary=summary_match.group(1) if summary_match else "Diagnosis partially parsed",
                root_cause=root_cause_match.group(1) if root_cause_match else "See additional notes",
                fix_category="informational",
                affected_files=[],
                proposed_changes=[],
                additional_notes=f"Unable to fully parse structured diagnosis. Raw excerpt: {response[:300]}...",
                confidence=0.25,
            )


_service: AIDiagnosisService | None = None


def get_ai_diagnosis_service() -> AIDiagnosisService:
    global _service
    if _service is None:
        _service = AIDiagnosisService()
    return _service
