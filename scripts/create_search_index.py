from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrievers.azure_search import AzureSearchIndexer


def main() -> None:
    AzureSearchIndexer().create_or_update_index()
    print("Azure AI Search index created or updated.")


if __name__ == "__main__":
    main()
