"""
GitHub PR Service using REST API.

Creates branches and pull requests without requiring local git.
Uses httpx for async HTTP requests.
"""

import os
import re
import base64
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# GitHub configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "aaltamimi2/polymer-solubility-app")
GITHUB_API_BASE = "https://api.github.com"


@dataclass
class PRResult:
    """Result of PR creation."""
    success: bool
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    branch_name: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "pr_url": self.pr_url,
            "pr_number": self.pr_number,
            "branch_name": self.branch_name,
            "error": self.error,
        }


@dataclass
class IssueResult:
    """Result of GitHub Issue creation."""
    success: bool
    issue_url: Optional[str] = None
    issue_number: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "issue_url": self.issue_url,
            "issue_number": self.issue_number,
            "error": self.error,
        }


class GitHubPRService:
    """Service for creating GitHub PRs via REST API."""

    def __init__(
        self,
        token: str = GITHUB_TOKEN,
        repo: str = GITHUB_REPO,
    ):
        self.token = token
        self.repo = repo
        self.api_base = GITHUB_API_BASE
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _sanitize_branch_name(self, name: str) -> str:
        """Create a valid git branch name."""
        # Remove or replace invalid characters
        name = re.sub(r'[^a-zA-Z0-9\-_/]', '-', name)
        name = re.sub(r'-+', '-', name)  # Collapse multiple dashes
        name = name.strip('-')
        return name[:50]  # Limit length

    async def get_default_branch(self) -> str:
        """Get the default branch of the repo."""
        client = await self._get_client()

        try:
            response = await client.get(f"/repos/{self.repo}")
            response.raise_for_status()
            data = response.json()
            return data.get("default_branch", "main")
        except Exception as e:
            logger.warning(f"Failed to get default branch, assuming 'main': {e}")
            return "main"

    async def get_branch_sha(self, branch: str) -> Optional[str]:
        """Get the SHA of a branch's HEAD."""
        client = await self._get_client()

        try:
            response = await client.get(f"/repos/{self.repo}/git/ref/heads/{branch}")
            response.raise_for_status()
            data = response.json()
            return data["object"]["sha"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_branch(self, branch_name: str, from_branch: str = None) -> bool:
        """Create a new branch from the specified base branch."""
        client = await self._get_client()

        if from_branch is None:
            from_branch = await self.get_default_branch()

        # Get the SHA of the base branch
        base_sha = await self.get_branch_sha(from_branch)
        if not base_sha:
            logger.error(f"Base branch '{from_branch}' not found")
            return False

        try:
            response = await client.post(
                f"/repos/{self.repo}/git/refs",
                json={
                    "ref": f"refs/heads/{branch_name}",
                    "sha": base_sha,
                }
            )
            response.raise_for_status()
            logger.info(f"Created branch: {branch_name}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                # Branch already exists
                logger.info(f"Branch already exists: {branch_name}")
                return True
            logger.error(f"Failed to create branch: {e}")
            return False

    async def get_file_content(self, path: str, branch: str) -> Optional[Dict[str, Any]]:
        """Get content and SHA of a file."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"/repos/{self.repo}/contents/{path}",
                params={"ref": branch}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_or_update_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str,
    ) -> bool:
        """Create or update a file in the repository."""
        client = await self._get_client()

        # Check if file exists to get its SHA
        existing = await self.get_file_content(path, branch)

        # Encode content to base64
        content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

        payload = {
            "message": message,
            "content": content_b64,
            "branch": branch,
        }

        if existing:
            payload["sha"] = existing["sha"]

        try:
            response = await client.put(
                f"/repos/{self.repo}/contents/{path}",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"{'Updated' if existing else 'Created'} file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create/update file {path}: {e}")
            return False

    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a pull request."""
        client = await self._get_client()

        if base_branch is None:
            base_branch = await self.get_default_branch()

        try:
            response = await client.post(
                f"/repos/{self.repo}/pulls",
                json={
                    "title": title,
                    "body": body,
                    "head": head_branch,
                    "base": base_branch,
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to create PR: {e.response.text}")
            return None

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: List[str] = None,
    ) -> IssueResult:
        """
        Create a GitHub Issue for tracking.

        Args:
            title: Issue title
            body: Issue body (markdown)
            labels: Optional list of labels

        Returns:
            IssueResult with success status and issue URL
        """
        if not self.token:
            return IssueResult(
                success=False,
                error="GitHub token not configured"
            )

        client = await self._get_client()

        try:
            payload = {
                "title": title,
                "body": body,
            }
            if labels:
                payload["labels"] = labels

            response = await client.post(
                f"/repos/{self.repo}/issues",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            issue_url = data.get("html_url")
            issue_number = data.get("number")

            logger.info(f"Created GitHub Issue #{issue_number}: {issue_url}")

            return IssueResult(
                success=True,
                issue_url=issue_url,
                issue_number=issue_number,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to create issue: {e.response.text}"
            logger.error(error_msg)
            return IssueResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Failed to create issue: {str(e)}"
            logger.error(error_msg)
            return IssueResult(success=False, error=error_msg)

    async def create_diagnostic_issue(
        self,
        diagnosis_summary: str,
        root_cause: str,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        additional_notes: str = "",
    ) -> IssueResult:
        """
        Create a GitHub Issue for an informational diagnosis (no code fix needed).

        Args:
            diagnosis_summary: AI's summary of the issue
            root_cause: AI's analysis of root cause
            user_question: Original user question
            assistant_response: The assistant's response that had an issue
            user_description: User's description of the problem
            issue_type: Category of the issue
            severity: Severity level
            additional_notes: Any additional context

        Returns:
            IssueResult with success status and issue URL
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build issue body
        body = f"""## 🔍 AI Diagnostic Report

**Generated:** {timestamp}
**Issue Type:** {issue_type}
**Severity:** {severity}
**Category:** Informational (no code changes proposed)

---

### Summary
{diagnosis_summary}

### Root Cause Analysis
{root_cause}

---

### Original Interaction

**User's Question:**
> {user_question[:500]}{'...' if len(user_question) > 500 else ''}

**Assistant's Response:**
```
{assistant_response[:1000]}{'...' if len(assistant_response) > 1000 else ''}
```

**User's Description of Problem:**
> {user_description}

---

### Additional Notes
{additional_notes if additional_notes else 'None'}

---

*This issue was automatically created by the AI diagnostic system.*
"""

        # Determine labels based on issue type and severity
        labels = ["ai-diagnostic", f"type:{issue_type}"]
        if severity in ["high", "critical"]:
            labels.append(f"severity:{severity}")

        title = f"[Diagnostic] {diagnosis_summary[:80]}{'...' if len(diagnosis_summary) > 80 else ''}"

        return await self.create_issue(title=title, body=body, labels=labels)

    async def create_issue_fix_pr(
        self,
        diagnosis_summary: str,
        root_cause: str,
        proposed_changes: List[Dict[str, Any]],
        issue_description: str,
        issue_type: str,
        severity: str,
        additional_notes: str = "",
    ) -> PRResult:
        """
        Create a PR with the proposed fix.

        Args:
            diagnosis_summary: Brief summary of the issue
            root_cause: Explanation of the root cause
            proposed_changes: List of {file, description, old_code, new_code}
            issue_description: User's description of the issue
            issue_type: Category of the issue
            severity: Severity level
            additional_notes: Any additional context

        Returns:
            PRResult with success status and PR URL
        """
        if not self.token:
            return PRResult(
                success=False,
                error="GitHub token not configured"
            )

        if not proposed_changes:
            return PRResult(
                success=False,
                error="No changes proposed"
            )

        try:
            # Generate branch name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = self._sanitize_branch_name(f"fix/{issue_type}-{timestamp}")

            # Create branch
            if not await self.create_branch(branch_name):
                return PRResult(
                    success=False,
                    error="Failed to create branch"
                )

            # Apply each proposed change
            for change in proposed_changes:
                file_path = change.get("file", "")
                new_code = change.get("new_code", "")
                description = change.get("description", "Apply fix")

                if not file_path or not new_code:
                    continue

                # Get current file content if it exists
                existing = await self.get_file_content(file_path, branch_name)

                if existing:
                    # File exists, need to apply the change
                    current_content = base64.b64decode(existing["content"]).decode('utf-8')
                    old_code = change.get("old_code")

                    if old_code and old_code in current_content:
                        # Replace the old code with new code
                        updated_content = current_content.replace(old_code, new_code, 1)
                    else:
                        # Can't find old code, append the new code with a comment
                        updated_content = current_content + f"\n\n# Auto-generated fix\n{new_code}"

                    final_content = updated_content
                else:
                    # New file
                    final_content = new_code

                # Update the file
                success = await self.create_or_update_file(
                    path=file_path,
                    content=final_content,
                    message=f"fix: {description}",
                    branch=branch_name,
                )

                if not success:
                    logger.warning(f"Failed to update file: {file_path}")

            # Create PR body
            pr_body = f"""## Issue Report Fix

### Summary
{diagnosis_summary}

### Root Cause
{root_cause}

### Issue Details
- **Type:** {issue_type}
- **Severity:** {severity}
- **User Description:** {issue_description}

### Changes Made
"""
            for change in proposed_changes:
                pr_body += f"\n- **{change.get('file', 'Unknown')}**: {change.get('description', 'No description')}"

            if additional_notes:
                pr_body += f"\n\n### Additional Notes\n{additional_notes}"

            pr_body += "\n\n---\n*This PR was automatically generated by the Issue Reporter system.*"

            # Create the PR
            pr_title = f"fix({issue_type}): {diagnosis_summary[:50]}"
            pr_data = await self.create_pull_request(
                title=pr_title,
                body=pr_body,
                head_branch=branch_name,
            )

            if pr_data:
                return PRResult(
                    success=True,
                    pr_url=pr_data.get("html_url"),
                    pr_number=pr_data.get("number"),
                    branch_name=branch_name,
                )
            else:
                return PRResult(
                    success=False,
                    branch_name=branch_name,
                    error="Failed to create pull request"
                )

        except Exception as e:
            logger.error(f"Failed to create issue fix PR: {e}")
            return PRResult(
                success=False,
                error=str(e)
            )
        finally:
            await self.close()


# Global singleton instance
_service: Optional[GitHubPRService] = None


def get_github_pr_service() -> GitHubPRService:
    """Get the global GitHub PR service."""
    global _service
    if _service is None:
        _service = GitHubPRService()
    return _service
