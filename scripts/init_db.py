from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.base import Base
from app.db.session import engine


def main() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    main()
