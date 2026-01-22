"""
Issue Reporter - Main orchestrator for error reporting.

Coordinates AI diagnosis and GitHub PR creation.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .codebase_context import get_codebase_context
from .ai_diagnosis import get_ai_diagnosis_service, DiagnosisResult
from .github_pr import get_github_pr_service, PRResult, IssueResult

logger = logging.getLogger(__name__)


@dataclass
class IssueReportResult:
    """Result of processing an issue report."""
    success: bool
    diagnosis: Optional[Dict[str, Any]] = None
    pr_result: Optional[Dict[str, Any]] = None
    issue_result: Optional[Dict[str, Any]] = None  # For GitHub Issues (non-PR reports)
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "diagnosis": self.diagnosis,
            "pr_result": self.pr_result,
            "issue_result": self.issue_result,
            "message": self.message,
            "error": self.error,
        }


class IssueReporter:
    """Main orchestrator for issue reporting."""

    def __init__(self):
        self.context_provider = get_codebase_context()
        self.diagnosis_service = get_ai_diagnosis_service()
        self.github_service = get_github_pr_service()

    async def process_report(
        self,
        user_question: str,
        assistant_response: str,
        elapsed_time: float,
        iterations: int,
        images: List[Dict[str, str]],  # [{filename, base64}]
        user_description: str,
        issue_type: str,
        severity: str,
        session_id: Optional[str] = None,
    ) -> IssueReportResult:
        """
        Process an issue report end-to-end.

        Args:
            user_question: The original user question
            assistant_response: The assistant's response that had an issue
            elapsed_time: Time taken for the response
            iterations: Number of agent iterations
            images: List of attached images (base64 encoded)
            user_description: User's description of the problem
            issue_type: Category (incorrect_response, ui_bug, api_error, etc.)
            severity: Severity level (low, medium, high, critical)
            session_id: Optional session identifier

        Returns:
            IssueReportResult with diagnosis and PR information
        """
        try:
            logger.info(f"Processing issue report: type={issue_type}, severity={severity}")

            # Step 1: Load codebase context
            if not self.context_provider.load():
                logger.warning("Codebase context not available, using minimal context")

            # Get relevant files based on issue type
            codebase_context = self.context_provider.get_context_for_issue_type(issue_type)
            tools = self.context_provider.get_tools()
            endpoints = self.context_provider.get_endpoints()

            logger.info(f"Loaded context: {len(codebase_context)} files, {len(tools)} tools, {len(endpoints)} endpoints")

            # Step 2: Run AI diagnosis
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

            logger.info(f"Diagnosis complete: category={diagnosis.fix_category}, confidence={diagnosis.confidence}")

            # Step 3: Determine action based on diagnosis
            if diagnosis.fix_category == "informational":
                # No code changes needed - create a GitHub Issue for tracking
                issue_result = await self.github_service.create_diagnostic_issue(
                    diagnosis_summary=diagnosis.summary,
                    root_cause=diagnosis.root_cause,
                    user_question=user_question,
                    assistant_response=assistant_response,
                    user_description=user_description,
                    issue_type=issue_type,
                    severity=severity,
                    additional_notes=diagnosis.additional_notes,
                )

                if issue_result.success:
                    return IssueReportResult(
                        success=True,
                        diagnosis=diagnosis.to_dict(),
                        issue_result=issue_result.to_dict(),
                        message=f"Issue analyzed and logged: {issue_result.issue_url}. No code changes required.",
                    )
                else:
                    return IssueReportResult(
                        success=True,
                        diagnosis=diagnosis.to_dict(),
                        message=f"Issue analyzed: {diagnosis.summary}. No code changes required. (GitHub Issue creation failed: {issue_result.error})",
                    )

            # Step 4: Create PR for simple or complex fixes
            if diagnosis.proposed_changes:
                pr_result = await self.github_service.create_issue_fix_pr(
                    diagnosis_summary=diagnosis.summary,
                    root_cause=diagnosis.root_cause,
                    proposed_changes=diagnosis.proposed_changes,
                    issue_description=user_description,
                    issue_type=issue_type,
                    severity=severity,
                    additional_notes=diagnosis.additional_notes,
                )

                if pr_result.success:
                    return IssueReportResult(
                        success=True,
                        diagnosis=diagnosis.to_dict(),
                        pr_result=pr_result.to_dict(),
                        message=f"Issue diagnosed and PR created: {pr_result.pr_url}",
                    )
                else:
                    return IssueReportResult(
                        success=True,  # Diagnosis succeeded even if PR failed
                        diagnosis=diagnosis.to_dict(),
                        pr_result=pr_result.to_dict(),
                        message=f"Issue diagnosed but PR creation failed: {pr_result.error}",
                        error=pr_result.error,
                    )
            else:
                return IssueReportResult(
                    success=True,
                    diagnosis=diagnosis.to_dict(),
                    message="Issue diagnosed but no specific code changes could be determined.",
                )

        except Exception as e:
            logger.error(f"Failed to process issue report: {e}")
            return IssueReportResult(
                success=False,
                error=str(e),
                message="Failed to process issue report",
            )


# Global singleton instance
_reporter: Optional[IssueReporter] = None


def get_issue_reporter() -> IssueReporter:
    """Get the global issue reporter."""
    global _reporter
    if _reporter is None:
        _reporter = IssueReporter()
    return _reporter
