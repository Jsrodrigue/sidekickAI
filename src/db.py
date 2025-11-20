import sqlite3
from pathlib import Path

DB_PATH = Path("database.db")

def get_conn():
    """
    Establishes a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: A connection object that can be used to interact with the database.
    """
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

def init_db():
    """
    Initializes the database by creating necessary tables if they do not exist.
    
    This function creates three tables: 'users', 'folders', and 'sessions'.
    The 'users' table stores username and password, the 'folders' table manages user folders,
        and the 'sessions' table stores JSON data representing the user's session state.
    
    Returns:
        None
    """
    conn = get_conn()
    cur = conn.cursor()

    # user table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    );
    """)

    # folders table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS folders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        folder_path TEXT NOT NULL,
        UNIQUE(username, folder_path),
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );
    """)

    # sessions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        username TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()