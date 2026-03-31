from pathlib import Path

from app.ingestion.workspace_discovery import discover_workspace_files


def test_workspace_discovery_finds_project_pdfs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("WORKSPACE_ROOT", str(root.parent))
    result = discover_workspace_files()
    assert len(result.pdf_files) >= 1
    assert any(path.suffix.lower() == ".pdf" for path in result.pdf_files)
