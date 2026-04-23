from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: Path) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SEM output pickle references old module paths. "
                "Regenerate output.pkl from the new runtime package structure."
            ) from exc
