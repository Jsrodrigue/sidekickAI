from typing import Optional

from src.db.db import get_conn


class UserRepository:
    def get_user(self, username: str) -> Optional[dict]:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT username, password FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        conn.close()

        if row:
            return {"username": row[0], "password": row[1]}
        return None

    def create_user(self, username: str, hashed_password: str) -> bool:
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password),
            )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
