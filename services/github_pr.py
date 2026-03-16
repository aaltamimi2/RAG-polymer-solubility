"""
GitHub issue/PR automation for issue reports.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "aaltamimi2/RAG-polymer-solubility")
GITHUB_API_BASE = "https://api.github.com"


@dataclass
class PRResult:
    success: bool
    pr_url: str | None = None
    pr_number: int | None = None
    branch_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "pr_url": self.pr_url,
            "pr_number": self.pr_number,
            "branch_name": self.branch_name,
            "error": self.error,
        }


@dataclass
class IssueResult:
    success: bool
    issue_url: str | None = None
    issue_number: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "issue_url": self.issue_url,
            "issue_number": self.issue_number,
            "error": self.error,
        }


class GitHubPRService:
    def __init__(self, token: str = GITHUB_TOKEN, repo: str = GITHUB_REPO):
        self.token = token
        self.repo = repo
        self.api_base = GITHUB_API_BASE
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def _sanitize_branch_name(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9._/-]", "-", name)
        name = re.sub(r"-+", "-", name).strip("-")
        return name[:60]

    async def get_default_branch(self) -> str:
        client = await self._get_client()
        try:
            response = await client.get(f"/repos/{self.repo}")
            response.raise_for_status()
            return response.json().get("default_branch", "main")
        except Exception as exc:
            logger.warning("Failed to resolve default branch, assuming main: %s", exc)
            return "main"

    async def get_branch_sha(self, branch: str) -> str | None:
        client = await self._get_client()
        try:
            response = await client.get(f"/repos/{self.repo}/git/ref/heads/{branch}")
            response.raise_for_status()
            return response.json()["object"]["sha"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def create_branch(self, branch_name: str, from_branch: str | None = None) -> bool:
        client = await self._get_client()
        base_branch = from_branch or await self.get_default_branch()
        base_sha = await self.get_branch_sha(base_branch)
        if not base_sha:
            logger.error("Base branch %s not found", base_branch)
            return False
        try:
            response = await client.post(
                f"/repos/{self.repo}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                return True
            logger.error("Failed to create branch %s: %s", branch_name, exc.response.text)
            return False

    async def get_file_content(self, path: str, branch: str) -> dict[str, Any] | None:
        client = await self._get_client()
        try:
            response = await client.get(f"/repos/{self.repo}/contents/{path}", params={"ref": branch})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def create_or_update_file(self, *, path: str, content: str, message: str, branch: str) -> bool:
        client = await self._get_client()
        existing = await self.get_file_content(path, branch)
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if existing:
            payload["sha"] = existing["sha"]
        try:
            response = await client.put(f"/repos/{self.repo}/contents/{path}", json=payload)
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Failed to update %s on %s: %s", path, branch, exc)
            return False

    async def create_pull_request(
        self,
        *,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str | None = None,
    ) -> dict[str, Any] | None:
        client = await self._get_client()
        base = base_branch or await self.get_default_branch()
        try:
            response = await client.post(
                f"/repos/{self.repo}/pulls",
                json={"title": title, "body": body, "head": head_branch, "base": base},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to create PR: %s", exc.response.text)
            return None

    async def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> IssueResult:
        if not self.token:
            return IssueResult(success=False, error="GitHub token not configured")
        client = await self._get_client()
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        try:
            response = await client.post(f"/repos/{self.repo}/issues", json=payload)
            response.raise_for_status()
            data = response.json()
            return IssueResult(
                success=True,
                issue_url=data.get("html_url"),
                issue_number=data.get("number"),
            )
        except httpx.HTTPStatusError as exc:
            return IssueResult(success=False, error=f"Failed to create issue: {exc.response.text}")
        except Exception as exc:
            return IssueResult(success=False, error=f"Failed to create issue: {exc}")

    async def create_diagnostic_issue(
        self,
        *,
        diagnosis_summary: str,
        root_cause: str,
        user_question: str,
        assistant_response: str,
        user_description: str,
        issue_type: str,
        severity: str,
        additional_notes: str = "",
    ) -> IssueResult:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""## AI Diagnostic Report

**Generated:** {timestamp}
**Issue Type:** {issue_type}
**Severity:** {severity}
**Category:** Informational / manual triage

### Summary
{diagnosis_summary}

### Root Cause Analysis
{root_cause}

### Original Interaction
**User Question**
> {user_question[:700]}{'...' if len(user_question) > 700 else ''}

**Assistant Response**
```
{assistant_response[:1400]}{'...' if len(assistant_response) > 1400 else ''}
```

**User Description**
> {user_description}

### Additional Notes
{additional_notes or 'None'}

---
Automatically generated by the DISSOLVE issue reporter.
"""
        labels = ["ai-diagnostic", f"type:{issue_type}"]
        if severity in {"high", "critical"}:
            labels.append(f"severity:{severity}")
        title = f"[Diagnostic] {diagnosis_summary[:80]}{'...' if len(diagnosis_summary) > 80 else ''}"
        return await self.create_issue(title=title, body=body, labels=labels)

    async def create_issue_fix_pr(
        self,
        *,
        diagnosis_summary: str,
        root_cause: str,
        proposed_changes: list[dict[str, Any]],
        issue_description: str,
        issue_type: str,
        severity: str,
        additional_notes: str = "",
    ) -> PRResult:
        if not self.token:
            return PRResult(success=False, error="GitHub token not configured")
        if not proposed_changes:
            return PRResult(success=False, error="No proposed changes were provided")
        branch_name = self._sanitize_branch_name(f"fix/{issue_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        try:
            if not await self.create_branch(branch_name):
                return PRResult(success=False, error="Failed to create branch")

            for change in proposed_changes:
                file_path = str(change.get("file", "")).strip()
                new_code = str(change.get("new_code", "") or "")
                description = str(change.get("description", "Apply issue reporter fix"))
                if not file_path or not new_code:
                    continue

                existing = await self.get_file_content(file_path, branch_name)
                if existing and existing.get("content"):
                    current_content = base64.b64decode(existing["content"]).decode("utf-8")
                    old_code = change.get("old_code")
                    if old_code and isinstance(old_code, str) and old_code in current_content:
                        final_content = current_content.replace(old_code, new_code, 1)
                    else:
                        final_content = current_content + f"\n\n# Auto-generated issue reporter patch\n{new_code}"
                else:
                    final_content = new_code

                await self.create_or_update_file(
                    path=file_path,
                    content=final_content,
                    message=f"fix: {description}",
                    branch=branch_name,
                )

            pr_body_lines = [
                "## Issue Report Fix",
                "",
                "### Summary",
                diagnosis_summary,
                "",
                "### Root Cause",
                root_cause,
                "",
                "### Issue Details",
                f"- Type: {issue_type}",
                f"- Severity: {severity}",
                f"- User description: {issue_description}",
                "",
                "### Proposed Changes",
            ]
            for change in proposed_changes:
                pr_body_lines.append(
                    f"- **{change.get('file', 'unknown')}**: {change.get('description', 'No description')}"
                )
            if additional_notes:
                pr_body_lines.extend(["", "### Additional Notes", additional_notes])
            pr_body_lines.extend(["", "---", "Automatically generated by the DISSOLVE issue reporter."])

            pr_data = await self.create_pull_request(
                title=f"fix({issue_type}): {diagnosis_summary[:50]}",
                body="\n".join(pr_body_lines),
                head_branch=branch_name,
            )
            if pr_data:
                return PRResult(
                    success=True,
                    pr_url=pr_data.get("html_url"),
                    pr_number=pr_data.get("number"),
                    branch_name=branch_name,
                )
            return PRResult(success=False, branch_name=branch_name, error="Failed to create pull request")
        except Exception as exc:
            logger.error("Failed to create fix PR: %s", exc)
            return PRResult(success=False, branch_name=branch_name, error=str(exc))
        finally:
            await self.close()


_service: GitHubPRService | None = None


def get_github_pr_service() -> GitHubPRService:
    global _service
    if _service is None:
        _service = GitHubPRService()
    return _service
