"""
Repositorio para gestionar sesiones en la base de datos.
"""
import json
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from src.db.db import get_conn
from src.core.state import SidekickState


class SessionRepository:
    """Repositorio para operaciones CRUD de sesiones."""

    @staticmethod
    def save(username: str, state: SidekickState):
        """Guarda el estado de sesión en la base de datos."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            # Serializar mensajes
            messages_data = []
            for m in getattr(state, "messages", []):
                msg_type = type(m).__name__
                msg_dict = {
                    "type": msg_type,
                    "content": getattr(m, "content", "")
                }
                
                # Manejar tool_calls si existen
                if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
                    try:
                        msg_dict["tool_calls"] = [
                            {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "id": tc.get("id", "")
                            }
                            for tc in m.tool_calls
                        ]
                    except Exception:
                        msg_dict["tool_calls"] = []
                
                messages_data.append(msg_dict)

            # Crear objeto de datos completo
            data = {
                "messages": messages_data,
                "session_id": getattr(state, "session_id", str(uuid.uuid4())),
                "current_directory": getattr(state, "current_directory", None),
                "indexed_directories": getattr(state, "indexed_directories", []),
                "success_criteria": getattr(state, "success_criteria", None),
                "criteria_met": getattr(state, "criteria_met", False),
                "needs_user_input": getattr(state, "needs_user_input", False),
                "evaluation_history": getattr(state, "evaluation_history", []),
                "task_metadata": getattr(state, "task_metadata", {}),
            }

            json_data = json.dumps(data, ensure_ascii=False)
            
            cur.execute("""
                INSERT INTO sessions (username, data) VALUES (?, ?)
                ON CONFLICT(username) DO UPDATE SET data=excluded.data;
            """, (username, json_data))
            
            conn.commit()
        except Exception as e:
            print(f"Error saving session: {e}")
        finally:
            conn.close()

    @staticmethod
    def load(username: str) -> SidekickState:
        """Carga el estado de sesión desde la base de datos."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT data FROM sessions WHERE username = ?", (username,))
            row = cur.fetchone()
            
            if not row:
                return SidekickState()

            data = json.loads(row[0])
            
            # Reconstruir mensajes
            messages = []
            for m in data.get("messages", []):
                msg_type = m.get("type", "")
                content = m.get("content", "") or ""
                
                if msg_type == "HumanMessage":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

            # Reconstruir estado
            state = SidekickState(
                messages=messages,
                session_id=data.get("session_id") or str(uuid.uuid4()),
                current_directory=data.get("current_directory"),
                indexed_directories=data.get("indexed_directories") or [],
                success_criteria=data.get("success_criteria"),
                criteria_met=data.get("criteria_met", False),
                needs_user_input=data.get("needs_user_input", False),
                evaluation_history=data.get("evaluation_history") or [],
                task_metadata=data.get("task_metadata") or {}
            )
            
            return state
        except Exception as e:
            print(f"Error loading session: {e}")
            return SidekickState()
        finally:
            conn.close()

    @staticmethod
    def delete(username: str) -> bool:
        """Elimina la sesión de un usuario."""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM sessions WHERE username = ?", (username,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
        finally:
            conn.close()

    @staticmethod
    def clear_messages(username: str) -> bool:
        """Limpia los mensajes de una sesión."""
        state = SessionRepository.load(username)
        state.messages = []
        SessionRepository.save(username, state)
        return True