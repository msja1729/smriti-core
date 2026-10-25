"""On-disk JSON cache so benchmark reruns make no repeat hosted calls (SPEC §4)."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / _digest(namespace)[:16] / f"{_digest(key)}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        try:
            return json.loads(self._path(namespace, key).read_text())
        except FileNotFoundError:
            return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)                  # atomic: readers never see half a file
