"""
Main orchestrator for report -> diagnosis -> issue/PR workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .ai_diagnosis import DiagnosisResult, get_ai_diagnosis_service
from .codebase_context import get_codebase_context
from .github_pr import get_github_pr_service

logger = logging.getLogger(__name__)


@dataclass
class IssueReportResult:
    success: bool
    diagnosis: dict[str, Any] | None = None
    pr_result: dict[str, Any] | None = None
    issue_result: dict[str, Any] | None = None
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "diagnosis": self.diagnosis,
            "pr_result": self.pr_result,
            "issue_result": self.issue_result,
            "message": self.message,
            "error": self.error,
        }


class IssueReporter:
    def __init__(self):
        self.context_provider = get_codebase_context()
        self.diagnosis_service = get_ai_diagnosis_service()
        self.github_service = get_github_pr_service()

    async def process_report(
        self,
        *,
        user_question: str,
        assistant_response: str,
        elapsed_time: float,
        iterations: int,
        images: list[dict[str, str]],
        user_description: str,
        issue_type: str,
        severity: str,
        session_id: str | None = None,
    ) -> IssueReportResult:
        try:
            if not self.context_provider.load():
                logger.warning("Codebase context unavailable; continuing with minimal context")

            codebase_context = self.context_provider.get_context_for_issue_type(issue_type)
            tools = self.context_provider.get_tools()
            endpoints = self.context_provider.get_endpoints()

            diagnosis = self.diagnosis_service.diagnose(
                user_question=user_question,
                assistant_response=assistant_response,
                user_description=user_description,
                issue_type=issue_type,
                severity=severity,
                codebase_context=codebase_context,
                tools=tools,
                endpoints=endpoints,
            )

            runtime_notes = self._runtime_notes(
                diagnosis=diagnosis,
                session_id=session_id,
                elapsed_time=elapsed_time,
                iterations=iterations,
                image_count=len(images),
            )

            if diagnosis.fix_category == "informational" or not diagnosis.proposed_changes:
                issue_result = await self.github_service.create_diagnostic_issue(
                    diagnosis_summary=diagnosis.summary,
                    root_cause=diagnosis.root_cause,
                    user_question=user_question,
                    assistant_response=assistant_response,
                    user_description=user_description,
                    issue_type=issue_type,
                    severity=severity,
                    additional_notes=runtime_notes,
                )
                if issue_result.success:
                    return IssueReportResult(
                        success=True,
                        diagnosis=diagnosis.to_dict(),
                        issue_result=issue_result.to_dict(),
                        message=f"Issue analyzed and logged on GitHub: {issue_result.issue_url}",
                    )
                return IssueReportResult(
                    success=True,
                    diagnosis=diagnosis.to_dict(),
                    issue_result=issue_result.to_dict(),
                    message=f"Issue analyzed. GitHub issue creation unavailable: {issue_result.error}",
                    error=issue_result.error,
                )

            pr_result = await self.github_service.create_issue_fix_pr(
                diagnosis_summary=diagnosis.summary,
                root_cause=diagnosis.root_cause,
                proposed_changes=diagnosis.proposed_changes,
                issue_description=user_description,
                issue_type=issue_type,
                severity=severity,
                additional_notes=runtime_notes,
            )

            if pr_result.success:
                return IssueReportResult(
                    success=True,
                    diagnosis=diagnosis.to_dict(),
                    pr_result=pr_result.to_dict(),
                    message=f"Issue diagnosed and PR created: {pr_result.pr_url}",
                )

            issue_result = await self.github_service.create_diagnostic_issue(
                diagnosis_summary=diagnosis.summary,
                root_cause=diagnosis.root_cause,
                user_question=user_question,
                assistant_response=assistant_response,
                user_description=user_description,
                issue_type=issue_type,
                severity=severity,
                additional_notes=runtime_notes,
            )
            message = f"Issue diagnosed but PR creation failed: {pr_result.error}"
            if issue_result.success:
                message += f" Diagnostic issue created: {issue_result.issue_url}"
            return IssueReportResult(
                success=True,
                diagnosis=diagnosis.to_dict(),
                pr_result=pr_result.to_dict(),
                issue_result=issue_result.to_dict(),
                message=message,
                error=pr_result.error,
            )
        except Exception as exc:
            logger.error("Failed to process issue report: %s", exc)
            return IssueReportResult(
                success=False,
                message="Failed to process issue report",
                error=str(exc),
            )

    @staticmethod
    def _runtime_notes(
        *,
        diagnosis: DiagnosisResult,
        session_id: str | None,
        elapsed_time: float,
        iterations: int,
        image_count: int,
    ) -> str:
        notes: list[str] = []
        if diagnosis.additional_notes:
            notes.append(diagnosis.additional_notes)
        notes.append(f"Session ID: {session_id or 'n/a'}")
        notes.append(f"Elapsed time: {elapsed_time:.2f}s")
        notes.append(f"Iterations: {iterations}")
        notes.append(f"Attached images: {image_count}")
        return "\n".join(notes)


_reporter: IssueReporter | None = None


def get_issue_reporter() -> IssueReporter:
    global _reporter
    if _reporter is None:
        _reporter = IssueReporter()
    return _reporter
