from __future__ import annotations

from strap import langsmith_tracing


def test_friendly_langsmith_error_for_missing_workspace() -> None:
    exc = Exception(
        "403 Client Error: Forbidden ... This API key is org-scoped and requires workspace specification. "
        "Please provide 'X-Tenant-ID' header."
    )
    message = langsmith_tracing._friendly_langsmith_error(exc)
    assert "LANGSMITH_WORKSPACE_ID" in message


def test_summarize_subagent_tool_runs_returns_clear_workspace_message(monkeypatch) -> None:
    class FakeClient:
        def read_run(self, run_id: str, load_child_runs: bool = False):
            raise Exception(
                "403 Client Error: Forbidden for url: https://api.smith.langchain.com/runs/foo "
                "{\"detail\":\"This API key is org-scoped and requires workspace specification. "
                "Please provide 'X-Tenant-ID' header.\"}"
            )

    monkeypatch.setattr(langsmith_tracing, "_LANGSMITH_CLIENT", FakeClient())
    monkeypatch.setattr(langsmith_tracing, "tracing_enabled", lambda: True)

    result = langsmith_tracing.summarize_subagent_tool_runs("run-123")
    assert result["tool_count"] == 0
    assert "LANGSMITH_WORKSPACE_ID" in result["error"]


def test_workspace_id_accepts_tenant_aliases(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "tenant-123")

    assert langsmith_tracing._workspace_id() == "tenant-123"
