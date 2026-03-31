from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.workspace_discovery import discover_workspace_files


def main() -> None:
    files = discover_workspace_files()
    print("Workspace Root:", files.workspace_root)
    print("\nPDF files:")
    for pdf in files.pdf_files:
        print(" -", pdf)
    print("\nER diagram/image files:")
    for image in files.diagram_files:
        print(" -", image)


if __name__ == "__main__":
    main()
