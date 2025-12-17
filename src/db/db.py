import sqlite3
from pathlib import Path
from typing import List

DB_PATH = Path("database.db")


def get_conn() -> sqlite3.Connection:
    """
    Returns a SQLite connection to the app database.

    check_same_thread=False allows reuse of the same connection across
    different threads (useful for Gradio / async contexts).
    """
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """
    Helper: returns the list of column names for a given table.
    Returns [] if the table does not exist.
    """
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table_name});")
        rows = cur.fetchall()
        return [r[1] for r in rows]  # r[1] = column name
    except Exception:
        return []


def init_db() -> None:
    """
    Initializes the database:
      - users:   auth info
      - folders: folders registered per user
      - sessions: serialized SidekickState per (username, folder)

    The sessions table is compatible with the SessionRepository / SessionService
    you showed earlier, which save/load by (username, folder).
    """
    conn = get_conn()
    cur = conn.cursor()

    # Always enforce foreign key constraints in SQLite
    cur.execute("PRAGMA foreign_keys = ON;")

    # ---------- users table ----------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        );
        """
    )

    # ---------- folders table ----------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS folders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT NOT NULL,
            folder_path TEXT NOT NULL,
            UNIQUE(username, folder_path),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        """
    )

    # ---------- sessions table ----------
    # We want: username + folder as composite primary key + JSON data
    #
    # To be safe with old DBs, we detect if sessions exists without "folder"
    # and recreate it in that case.

    existing_cols = _table_columns(conn, "sessions")

    if not existing_cols:
        # Table does not exist -> create fresh
        cur.execute(
            """
            CREATE TABLE sessions (
                username TEXT NOT NULL,
                folder   TEXT NOT NULL,
                data     TEXT NOT NULL,
                PRIMARY KEY (username, folder),
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );
            """
        )
        print("[DB] Created sessions table (username, folder, data).")

    elif "folder" not in existing_cols:
        # Old schema (only username, data) -> drop & recreate
        print("[DB] Detected old sessions schema (no 'folder' column). Dropping and recreating...")
        cur.execute("DROP TABLE IF EXISTS sessions;")
        cur.execute(
            """
            CREATE TABLE sessions (
                username TEXT NOT NULL,
                folder   TEXT NOT NULL,
                data     TEXT NOT NULL,
                PRIMARY KEY (username, folder),
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );
            """
        )
        print("[DB] Recreated sessions table with (username, folder, data).")
    else:
        # Already in the correct format
        print("[DB] sessions table already up to date.")

    conn.commit()
    conn.close()
