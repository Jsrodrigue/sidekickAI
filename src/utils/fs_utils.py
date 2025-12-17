"""
Filesystem utilities (Windows-friendly).

Purpose:
- Robust delete for vectorstore directories (Chroma) on Windows.
- Handles read-only files, retries, and provides a manual fallback to surface locks.

"""

from __future__ import annotations

import os
import shutil
import stat
import time
from typing import Optional, Tuple


def rm_onerror_make_writable(func, path, exc_info) -> None:
    """
    shutil.rmtree onerror handler:
    - removes read-only bit and retries the failing operation.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        # allow caller to observe failure if it persists
        pass


def force_delete_dir(dir_path: str) -> Tuple[bool, Optional[str]]:
    """
    Last-resort delete: remove files/subdirs manually.
    Returns (ok, error_message).
    """
    try:
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    os.chmod(fp, stat.S_IWRITE)
                except Exception:
                    pass
                os.remove(fp)

            for name in dirs:
                dp = os.path.join(root, name)
                try:
                    os.chmod(dp, stat.S_IWRITE)
                except Exception:
                    pass
                os.rmdir(dp)

        os.rmdir(dir_path)
        return True, None
    except Exception as e:
        return False, str(e)


def delete_dir_verified(
    dir_path: str,
    *,
    retries: int = 3,
    sleep_s: float = 0.25,
    debug: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Try to delete a directory using shutil.rmtree and verify it is gone.
    If it still exists, fallback to force_delete_dir.

    Returns (deleted_ok, error_message).
    """
    dir_abs = os.path.abspath(dir_path)
    last_err: Optional[str] = None

    if debug:
        print("[DEBUG] persist_dir abs:", repr(dir_abs))
        print("[DEBUG] exists before:", os.path.exists(dir_abs))

    if not os.path.exists(dir_abs):
        return True, None

    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(dir_abs):
                shutil.rmtree(dir_abs, onerror=rm_onerror_make_writable)

            time.sleep(sleep_s)

            if not os.path.exists(dir_abs):
                if debug:
                    print(f"[DEBUG] rmtree SUCCESS (verified) on attempt {attempt}")
                return True, None

            if debug:
                print(f"[DEBUG] rmtree returned but dir STILL EXISTS (attempt {attempt})")

        except Exception as e:
            last_err = str(e)
            if debug:
                print(f"[DEBUG] rmtree FAILED attempt {attempt}: {e}")
            time.sleep(sleep_s)

    if debug:
        print("[DEBUG] attempting FORCE delete fallback...")

    ok, err = force_delete_dir(dir_abs)

    if debug:
        print("[DEBUG] force delete ok:", ok, "err:", err)
        print("[DEBUG] exists after fallback:", os.path.exists(dir_abs))

    if ok and (not os.path.exists(dir_abs)):
        return True, None

    return False, err or last_err or "Unknown delete failure"
