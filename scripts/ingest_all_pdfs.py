from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.services.ingest_service import IngestService


def main() -> None:
    with SessionLocal() as session:
        result = IngestService(session).ingest_workspace()
        print(result)


if __name__ == "__main__":
    main()
