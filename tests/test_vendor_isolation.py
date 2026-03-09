from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "strap"


def _non_vendor_python_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "vendor":
            continue
        yield path


def test_non_vendor_modules_do_not_import_vendor_rag_directly():
    allowed = {
        ROOT / "services" / "rag_service.py",
    }

    offenders = []
    for path in _non_vendor_python_files():
        if path in allowed:
            continue
        text = path.read_text()
        if "from strap.vendor import rag" in text or "from strap.vendor.rag import" in text:
            offenders.append(str(path.relative_to(ROOT.parent)))

    assert offenders == []


def test_non_vendor_modules_do_not_import_agent_sql_source_directly():
    offenders = []
    for path in _non_vendor_python_files():
        text = path.read_text()
        if "strap.vendor._agent_sql_source" in text or "from strap.vendor import _agent_sql_source" in text:
            offenders.append(str(path.relative_to(ROOT.parent)))

    assert offenders == []


def test_rag_service_proxies_vendor_module(monkeypatch):
    from strap.services import rag_service

    class FakeVendorRag:
        def get_rag_status(self):
            return {"ready": True}

    monkeypatch.setattr(rag_service, "_get_vendor_rag", lambda: FakeVendorRag())

    assert rag_service.get_rag_status() == {"ready": True}
