"""
Utilidades para manejo de paths y validaciones.
"""
import os
from typing import Optional, List


def normalize_path(path: str) -> str:
    """Normaliza un path de manera segura."""
    try:
        return os.path.normpath(path) if path else ""
    except Exception:
        return path


def is_excluded_path(path: str, excluded_dirs: Optional[List[str]] = None) -> bool:
    """Verifica si un path debe ser excluido."""
    if not path:
        return True
    
    excluded_dirs = excluded_dirs or [".venv", "venv", "__pycache__"]
    parts = normalize_path(path).replace("\\", "/").split("/")
    
    return any(ex in parts for ex in excluded_dirs)


def validate_directory(directory: str) -> bool:
    """Valida que un directorio existe y es accesible."""
    if not directory:
        return False
    
    normalized = normalize_path(directory)
    return os.path.exists(normalized) and os.path.isdir(normalized)


def get_absolute_path(path: str) -> str:
    """Obtiene el path absoluto de manera segura."""
    try:
        return os.path.abspath(normalize_path(path))
    except Exception:
        return path