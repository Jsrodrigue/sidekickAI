"""
Utilities for path handling.
"""

import os
from typing import Optional, List


def normalize_path(path: str) -> str:
    """Normalize a path in a safe way."""
    try:
        return os.path.normpath(path) if path else ""
    except Exception:
        return path


def is_excluded_path(path: str, excluded_dirs: Optional[List[str]] = None) -> bool:
    """Verify if a path must be excluded."""
    if not path:
        return True

    excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
    parts = normalize_path(path).replace("\\", "/").split("/")

    return any(ex in parts for ex in excluded_dirs)


def validate_directory(directory: str) -> bool:
    """Validate if a directory exists and is accessible."""
    if not directory:
        return False

    normalized = normalize_path(directory)
    return os.path.exists(normalized) and os.path.isdir(normalized)


def get_absolute_path(path: str) -> str:
    """Get the absolute path in a safe way."""
    try:
        return os.path.abspath(normalize_path(path))
    except Exception:
        return path
