"""
Repository for managing the sessions
"""
import json
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from src.db.db import get_conn
from src.core.state import SidekickState


class SessionRepository:
    """Repository for CRUD operations on sessions, keyed by (username, folder)."""

    @staticmethod
    def save(username: str, folder: str, state: SidekickState):
        """Save the state of a session in the DB for a specific (username, folder)."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            # Serialize messages
            messages_data = []
            for m in getattr(state, "messages", []):
                msg_type = type(m).__name__
                msg_dict = {
                    "type": msg_type,
                    "content": getattr(m, "content", ""),
                }

                # Handles tool_calls if present
                if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
                    try:
                        msg_dict["tool_calls"] = [
                            {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "id": tc.get("id", ""),
                            }
                            for tc in m.tool_calls
                        ]
                    except Exception:
                        msg_dict["tool_calls"] = []

                messages_data.append(msg_dict)

            # Build a dict with all relevant fields
            data = {
                "messages": messages_data,
                "session_id": getattr(state, "session_id", str(uuid.uuid4())),
                "current_directory": getattr(state, "current_directory", None),
                "indexed_directories": getattr(state, "indexed_directories", []),
                "success_criteria": getattr(state, "success_criteria", None),
                "criteria_met": getattr(state, "criteria_met", False),
                "needs_user_input": getattr(state, "needs_user_input", False),
                "task_metadata": getattr(state, "task_metadata", {}),
            }

            json_data = json.dumps(data, ensure_ascii=False)

            # NOTE: requires a table with columns (username, folder, data)
            # and PRIMARY KEY(username, folder)
            cur.execute(
                """
                INSERT INTO sessions (username, folder, data)
                VALUES (?, ?, ?)
                ON CONFLICT(username, folder)
                DO UPDATE SET data = excluded.data;
                """,
                (username, folder, json_data),
            )

            conn.commit()
        except Exception as e:
            print(f"Error saving session: {e}")
        finally:
            conn.close()

    @staticmethod
    def load(username: str, folder: str) -> SidekickState:
        """Load the state of a session for a specific (username, folder)."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT data FROM sessions WHERE username = ? AND folder = ?",
                (username, folder),
            )
            row = cur.fetchone()

            if not row:
                # No session yet for this (user, folder)
                return SidekickState()

            data = json.loads(row[0])

            # Rebuild messages
            messages = []
            for m in data.get("messages", []):
                msg_type = m.get("type", "")
                content = m.get("content", "") or ""

                if msg_type == "HumanMessage":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

            # Rebuild the state
            state = SidekickState(
                messages=messages,
                session_id=data.get("session_id") or str(uuid.uuid4()),
                current_directory=data.get("current_directory"),
                indexed_directories=data.get("indexed_directories") or [],
                success_criteria=data.get("success_criteria"),
                criteria_met=data.get("criteria_met", False),
                needs_user_input=data.get("needs_user_input", False),
                task_metadata=data.get("task_metadata") or {},
            )

            return state
        except Exception as e:
            print(f"Error loading session: {e}")
            return SidekickState()
        finally:
            conn.close()

    @staticmethod
    def delete(username: str, folder: str) -> bool:
        """Delete the session for a specific (username, folder)."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM sessions WHERE username = ? AND folder = ?",
                (username, folder),
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
        finally:
            conn.close()

    @staticmethod
    def clear_messages(username: str, folder: str) -> bool:
        """Clear messages for a specific (username, folder) session."""
        state = SessionRepository.load(username, folder)
        state.messages = []
        SessionRepository.save(username, folder, state)
        return True
