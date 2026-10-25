"""Load strategy classes from this package."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path

import strategies

load_errors: list[str] = []
_loaded_mtime: dict[str, int] = {}


def _is_strategy_class(obj) -> bool:
    """A strategy has a key and the two methods. Subclass identity is not required.

    Streamlit can reload engine.strategy and leave an older Strategy class in
    memory. issubclass then rejects every strategy, including the book.
    """

    if not isinstance(obj, type) or obj.__name__ == "Strategy":
        return False
    key = getattr(obj, "key", "")
    if not isinstance(key, str) or not key:
        return False
    return callable(getattr(obj, "target_positions", None)) and callable(getattr(obj, "param_space", None))


def _load(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise ImportError(f"Cannot find {module_name}")
    mtime = Path(spec.origin).stat().st_mtime_ns
    loaded = sys.modules.get(module_name)
    if loaded is not None and _loaded_mtime.get(module_name) == mtime:
        return loaded
    if loaded is None:
        module = importlib.import_module(module_name)
    else:
        module = importlib.reload(loaded)
    _loaded_mtime[module_name] = mtime
    return module


def discover() -> list:
    """Return one fresh instance of each strategy in this package."""

    load_errors.clear()
    found = []
    seen: set[str] = set()
    for module_info in pkgutil.iter_modules(strategies.__path__):
        if module_info.name.startswith("_") or module_info.name == "registry":
            continue
        module_name = f"strategies.{module_info.name}"
        try:
            module = _load(module_name)
        except Exception as exc:
            load_errors.append(f"{module_info.name}: {exc}")
            continue
        for obj in vars(module).values():
            if not _is_strategy_class(obj) or obj.key in seen:
                continue
            seen.add(obj.key)
            found.append(obj())
    found.sort(key=lambda item: item.name)
    return found
