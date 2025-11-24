from src.db import get_conn
import hashlib

# -------------------- Utilities --------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------- DB User Management --------------------
def register_user(username, password):
    if not username or not password:
        return False, "Username and password required"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        conn.close()
        return False, "Username already exists"
    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hash_password(password)))
    conn.commit()
    conn.close()
    return True, "User registered successfully"

def login_user(username, password):
    if not username or not password:
        return False, "Username and password required"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False, "User not found"
    if row[0] != hash_password(password):
        return False, "Wrong password"
    return True, "Logged in"
