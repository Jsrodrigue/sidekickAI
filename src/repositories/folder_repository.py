"""
Repositorio para gestionar carpetas en la base de datos.
"""
from typing import List, Tuple
from src.db.db import get_conn


class FolderRepository:
    """Repositorio para operaciones CRUD de carpetas."""

    @staticmethod
    def add(username: str, folder_path: str) -> Tuple[bool, str, List[str]]:
        """Añade una carpeta para un usuario."""
        if not username or not folder_path:
            return False, "Username and folder path required", []
        
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT OR IGNORE INTO folders (username, folder_path) VALUES (?, ?)",
                (username, folder_path)
            )
            conn.commit()
            folders = FolderRepository.get_all(username)
            return True, "Folder added successfully", folders
        except Exception as e:
            return False, f"Error adding folder: {e}", []
        finally:
            conn.close()

    @staticmethod
    def remove(username: str, folder_path: str) -> Tuple[bool, str, List[str]]:
        """Elimina una carpeta de un usuario."""
        if not username or not folder_path:
            return False, "Username and folder path required", []
        
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM folders WHERE username = ? AND folder_path = ?",
                (username, folder_path)
            )
            conn.commit()
            folders = FolderRepository.get_all(username)
            return True, "Folder removed successfully", folders
        except Exception as e:
            return False, f"Error removing folder: {e}", []
        finally:
            conn.close()

    @staticmethod
    def get_all(username: str) -> List[str]:
        """Obtiene todas las carpetas de un usuario."""
        if not username:
            return []
        
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT folder_path FROM folders WHERE username = ?",
                (username,)
            )
            folders = [row[0] for row in cur.fetchall()]
            return folders
        except Exception as e:
            print(f"Error getting folders: {e}")
            return []
        finally:
            conn.close()

    @staticmethod
    def exists(username: str, folder_path: str) -> bool:
        """Verifica si una carpeta existe para un usuario."""
        if not username or not folder_path:
            return False
        
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM folders WHERE username = ? AND folder_path = ?",
                (username, folder_path)
            )
            count = cur.fetchone()[0]
            return count > 0
        except Exception as e:
            print(f"Error checking folder existence: {e}")
            return False
        finally:
            conn.close()