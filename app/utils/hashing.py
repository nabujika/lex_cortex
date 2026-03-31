from __future__ import annotations

import hashlib


def stable_hash(*parts: object, length: int = 24) -> str:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:length]

