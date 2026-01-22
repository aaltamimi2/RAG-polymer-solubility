"""
AI Diagnosis Service using Gemini 2.5 Pro.

Analyzes issue reports against the codebase context
and provides structured diagnosis with proposed fixes.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Gemini configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


@dataclass
class DiagnosisResult:
    """Structured diagnosis result."""
    summary: str
    root_cause: str
    fix_category: str  # "informational", "simple_fix", "complex_fix"
    affected_files: List[str]
    proposed_changes: List[Dict[str, Any]]  # {file, description, old_code, new_code}
    additional_notes: str
    confidence: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIDiagnosisService:
    """Service for AI-powered issue diagnosis."""

    def __init__(self, api_key: str = GOOGLE_API_KEY):
        self.api_key = api_key
        self._model = None

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=self.api_key,
                    temperature=0.3,
                    max_output_tokens=8192,
                )
                logger.info("Initialized Gemini 2.5 Flash for diagnosis")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                raise

        return self._model

    def diagnose(
        self,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        codebase_context: Dict[str, str],
        tools: List[str] = None,
        endpoints: List[str] = None,
    ) -> DiagnosisResult:
        """
        Analyze an issue and generate diagnosis.

        Args:
            user_question: The original user question
            assistant_response: The assistant's response that had an issue
            user_description: User's description of what went wrong
            issue_type: Category of the issue
            severity: Severity level
            codebase_context: Relevant source files {path: content}
            tools: List of available agent tools
            endpoints: List of API endpoints

        Returns:
            DiagnosisResult with structured diagnosis
        """
        model = self._get_model()

        # Build the prompt
        prompt = self._build_diagnosis_prompt(
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
            # Call the model
            response = model.invoke(prompt)
            content = response.content

            # Parse the response
            result = self._parse_diagnosis_response(content)
            return result

        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            # Return a fallback diagnosis
            return DiagnosisResult(
                summary=f"Automated diagnosis failed: {str(e)}",
                root_cause="Unable to determine - diagnosis service error",
                fix_category="informational",
                affected_files=[],
                proposed_changes=[],
                additional_notes=f"User reported: {user_description}",
                confidence=0.0,
            )

    def _build_diagnosis_prompt(
        self,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        codebase_context: Dict[str, str],
        tools: List[str],
        endpoints: List[str],
    ) -> str:
        """Build the diagnosis prompt."""

        # Format codebase context
        codebase_section = ""
        for file_path, content in codebase_context.items():
            # Truncate very long files
            if len(content) > 15000:
                content = content[:15000] + "\n... [truncated]"
            codebase_section += f"\n### {file_path}\n```python\n{content}\n```\n"

        prompt = f"""You are an expert software engineer analyzing a bug report for a polymer solubility analysis application.

## Issue Report

**Issue Type:** {issue_type}
**Severity:** {severity}

**User's Original Question:**
{user_question}

**Assistant's Response:**
{assistant_response}

**User's Description of the Problem:**
{user_description}

## System Context

**Available Tools:** {', '.join(tools) if tools else 'Not specified'}
**API Endpoints:** {', '.join(endpoints) if endpoints else 'Not specified'}

## Codebase
{codebase_section}

## Your Task

Analyze this issue and provide a diagnosis. You must respond with a valid JSON object with this exact structure:

```json
{{
  "summary": "Brief one-line summary of the issue",
  "root_cause": "Detailed explanation of what's causing the issue",
  "fix_category": "informational|simple_fix|complex_fix",
  "affected_files": ["list", "of", "affected", "files"],
  "proposed_changes": [
    {{
      "file": "path/to/file.py",
      "description": "What this change does",
      "old_code": "// existing code snippet to replace (or null for new additions)",
      "new_code": "// new code to add or replace with"
    }}
  ],
  "additional_notes": "Any other relevant information, caveats, or testing suggestions",
  "confidence": 0.85
}}
```

**Guidelines:**
- fix_category should be:
  - "informational" if no code change is needed (user error, expected behavior, etc.)
  - "simple_fix" if the fix is straightforward (1-2 file changes, clear solution)
  - "complex_fix" if it requires significant refactoring or multiple changes
- proposed_changes should contain actual code snippets that can be applied
- confidence should be 0.0-1.0 based on how certain you are about the diagnosis
- Be specific about file paths and line locations when possible

Respond ONLY with the JSON object, no additional text."""

        return prompt

    def _parse_diagnosis_response(self, response: str) -> DiagnosisResult:
        """Parse the model's response into a DiagnosisResult."""
        try:
            # Try to extract JSON from the response
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end == -1:
                    end = len(response)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end == -1:
                    end = len(response)
                response = response[start:end].strip()

            # Try to find JSON object boundaries
            if not response.startswith("{"):
                json_start = response.find("{")
                if json_start != -1:
                    response = response[json_start:]

            # Try to fix incomplete JSON by adding closing brackets
            if response.count("{") > response.count("}"):
                response = response + "}" * (response.count("{") - response.count("}"))
            if response.count("[") > response.count("]"):
                response = response + "]" * (response.count("[") - response.count("]"))

            # Try to fix truncated strings by closing them
            import re
            # Remove any trailing incomplete string/field
            response = re.sub(r',\s*"[^"]*$', '', response)
            response = re.sub(r':\s*"[^"]*$', ': ""', response)

            data = json.loads(response)

            return DiagnosisResult(
                summary=data.get("summary", "No summary provided"),
                root_cause=data.get("root_cause", "Unknown"),
                fix_category=data.get("fix_category", "informational"),
                affected_files=data.get("affected_files", []),
                proposed_changes=data.get("proposed_changes", []),
                additional_notes=data.get("additional_notes", ""),
                confidence=float(data.get("confidence", 0.5)),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse diagnosis response: {e}")
            logger.debug(f"Raw response: {response[:1000]}")

            # Try to extract key fields manually from malformed JSON
            summary = "Diagnosis partially parsed"
            root_cause = "See additional notes"

            # Try to extract summary
            import re
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response)
            if summary_match:
                summary = summary_match.group(1)

            root_cause_match = re.search(r'"root_cause"\s*:\s*"([^"]+)"', response)
            if root_cause_match:
                root_cause = root_cause_match.group(1)

            return DiagnosisResult(
                summary=summary,
                root_cause=root_cause,
                fix_category="informational",
                affected_files=[],
                proposed_changes=[],
                additional_notes=f"Partial parse of AI response. Raw excerpt: {response[:300]}...",
                confidence=0.3,
            )


# Global singleton instance
_service: Optional[AIDiagnosisService] = None


def get_ai_diagnosis_service() -> AIDiagnosisService:
    """Get the global AI diagnosis service."""
    global _service
    if _service is None:
        _service = AIDiagnosisService()
    return _service
