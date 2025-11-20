# app.py (updated)
import os
import json
import uuid
import hashlib
import asyncio
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.sidekick import SidekickState
from src.sidekick import Sidekick
from src.db import get_conn, init_db

load_dotenv()
init_db()

# -------------------- Sidekick setup --------------------
sidekick = Sidekick()

def init_sidekick():
    """Run the async sidekick.setup() once synchronously at startup."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sidekick.setup())
    loop.close()

init_sidekick()

# in-memory sessions store
sessions = {}

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

# -------------------- Folder Management --------------------
def add_folder(username, folder_path):
    if not username or not folder_path:
        return False, "Username and folder path required", []
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR IGNORE INTO folders (username, folder_path) VALUES (?, ?)",
                    (username, folder_path))
        conn.commit()
        cur.execute("SELECT folder_path FROM folders WHERE username = ?", (username,))
        folders = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    return True, "Folder added", folders

def remove_folder(username, folder_path):
    if not username or not folder_path:
        return False, "Username and folder path required", []
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM folders WHERE username = ? AND folder_path = ?",
                    (username, folder_path))
        conn.commit()
        cur.execute("SELECT folder_path FROM folders WHERE username = ?", (username,))
        folders = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    return True, "Folder removed", folders

def get_folders(username):
    if not username:
        return []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT folder_path FROM folders WHERE username = ?", (username,))
    folders = [r[0] for r in cur.fetchall()]
    conn.close()
    return folders

# -------------------- Session helpers (robust) --------------------
def save_session(username: str, state: SidekickState):
    """Save session state to DB (serialized JSON)."""
    conn = get_conn()
    cur = conn.cursor()
    # serialise messages
    messages_data = []
    for m in getattr(state, "messages", []):
        msg_type = type(m).__name__
        msg_dict = {"type": msg_type, "content": getattr(m, "content", "")}
        # optionally capture tool_calls metadata if present (safe)
        if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
            try:
                msg_dict["tool_calls"] = [
                    {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                    for tc in m.tool_calls
                ]
            except Exception:
                # ignore serialization errors from unusual tool_call structures
                msg_dict["tool_calls"] = []
        messages_data.append(msg_dict)

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
    conn.close()

def load_session(username: str) -> SidekickState:
    """Load session from DB and return a SidekickState with defaults for missing fields."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT data FROM sessions WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        # new state
        return SidekickState()

    data = json.loads(row[0])

    # messages reconstruction (defensive)
    messages = []
    for m in data.get("messages", []):
        msg_type = m.get("type", "")
        content = m.get("content", "") or ""
        if msg_type == "HumanMessage":
            messages.append(HumanMessage(content=content))
        elif msg_type == "AIMessage":
            messages.append(AIMessage(content=content))
        else:
            # unknown -> store as AIMessage to keep visible
            messages.append(AIMessage(content=content))

    # provide defaults for missing keys (backwards compatibility)
    session_id = data.get("session_id") or str(uuid.uuid4())
    current_directory = data.get("current_directory") if data.get("current_directory") not in (None, "") else None
    indexed_directories = data.get("indexed_directories") or []
    success_criteria = data.get("success_criteria")
    criteria_met = data.get("criteria_met", False)
    needs_user_input = data.get("needs_user_input", False)
    evaluation_history = data.get("evaluation_history") or []
    task_metadata = data.get("task_metadata") or {}

    # instantiate pydantic model with defaults
    try:
        state = SidekickState(
            messages=messages,
            session_id=session_id,
            current_directory=current_directory,
            indexed_directories=indexed_directories,
            success_criteria=success_criteria,
            criteria_met=criteria_met,
            needs_user_input=needs_user_input,
            evaluation_history=evaluation_history,
            task_metadata=task_metadata
        )
    except Exception as e:
        # Last-resort fallback to fresh state if validation fails
        print(f"⚠️ load_session: pydantic validation failed: {e}")
        state = SidekickState()
        state.messages = messages
        state.session_id = session_id
        state.current_directory = current_directory
        state.indexed_directories = indexed_directories
        state.success_criteria = success_criteria
        state.criteria_met = criteria_met
        state.needs_user_input = needs_user_input
        state.evaluation_history = evaluation_history
        state.task_metadata = task_metadata

    return state

# -------------------- Index folder helper --------------------
def index_folder(username, folder_path):
    if not folder_path:
        return "❌ Select a folder first"

    # DEBUG
    print("DEBUG -> index_folder(): username:", username)
    if username in sessions:
        print("DEBUG session keys:", sessions[username].__dict__)

    result = sidekick.load_and_register_directory(folder_path)

    if username in sessions:
        state = sessions[username]

        print("DEBUG state before updating:", state.__dict__)  # 👈 AQUI SE VE TODO

        state.current_directory = folder_path

        # SI EL CAMPO NO EXISTE, LO CREAMOS
        if not hasattr(state, "indexed_directories"):
            print("⚠️ indexed_directories missing, creating it")
            state.indexed_directories = []

        if folder_path not in state.indexed_directories:
            state.indexed_directories.append(folder_path)

        save_session(username, state)

    return result

# -------------------- Chat function --------------------
async def chat(user_id, folder_path, message):
    """Handle conversation: ensure directory indexed, append message, invoke graph, save session."""
    if not user_id:
        return [], "Please log in first"
    if message is None:
        message = ""
    message = message.strip()
    if not message:
        return [], ""

    # load or get session
    if user_id in sessions:
        state: SidekickState = sessions[user_id]
    else:
        state = load_session(user_id)
        sessions[user_id] = state  # cache loaded state

    # change directory if user selected one
    if folder_path and folder_path != state.current_directory:
        state.current_directory = folder_path
        # index in sidekick if not present
        if folder_path not in sidekick.retriever_registry:
            idx_result = sidekick.load_and_register_directory(folder_path)
            print(f"📁 {idx_result}")
        # remember in user's state
        if folder_path not in getattr(state, "indexed_directories", []):
            state.indexed_directories.append(folder_path)

    # append user's message
    state.messages.append(HumanMessage(content=message))

    # default success criteria
    if not state.success_criteria:
        state.success_criteria = "Answer the user's question completely and accurately."

    # invoke langgraph graph (async)
    try:
        # graph updates state in-place (Sidekick.graph built to work with SidekickState)
        invoke_config = {
            "configurable": {
                "thread_id": state.thread_id
            }
        }

        print(f"DEBUG invoking with thread_id={state.thread_id}")

        result_state = await sidekick.graph.ainvoke(state, invoke_config)

        # result_state should be SidekickState-like (langgraph may return a mapping)
        # If langgraph returns dict-like, try to use it; otherwise assume it's SidekickState
        if isinstance(result_state, dict):
            # If dict, merge into our SidekickState
            # we update fields we know
            if "messages" in result_state:
                state.messages = result_state["messages"]
            if "evaluation_history" in result_state:
                state.evaluation_history = result_state.get("evaluation_history", state.evaluation_history)
            state.criteria_met = result_state.get("criteria_met", state.criteria_met)
            state.needs_user_input = result_state.get("needs_user_input", state.needs_user_input)
            # keep other properties as-is
            sessions[user_id] = state
            save_session(user_id, state)
            result_state_obj = state
        else:
            # assume SidekickState returned
            sessions[user_id] = result_state
            save_session(user_id, result_state)
            result_state_obj = result_state

        # Build chat history for Gradio
        chat_history = []
        for m in result_state_obj.messages:
            if isinstance(m, HumanMessage):
                chat_history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                # filter out purely internal evaluator notes if you prefer
                if not getattr(m, "content", "").startswith("💭"):
                    chat_history.append({"role": "assistant", "content": m.content})
        return chat_history, ""
    except Exception as e:
        print("❌ Error in chat():", e)
        import traceback
        traceback.print_exc()
        # push error into messages and return what we have
        state.messages.append(AIMessage(content=f"Error processing request: {e}"))
        sessions[user_id] = state
        save_session(user_id, state)
        chat_history = []
        for m in state.messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_history.append({"role": role, "content": m.content})
        return chat_history, ""

# -------------------- Gradio UI (unchanged) --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Sidekick AI - Multi-User RAG Assistant")
    gr.Markdown("Sistema de asistente con RAG, autenticación y gestión de folders")

    # Login Page
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔐 Login / Register")
        username_input = gr.Textbox(label="Username", placeholder="Enter username")
        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")

        with gr.Row():
            login_btn = gr.Button("🔓 Login", variant="primary")
            register_btn = gr.Button("📝 Register", variant="secondary")

        login_status = gr.Label(label="Status")

    # Main App Page
    with gr.Column(visible=False) as main_page:
        gr.Markdown("## 📁 Folder Management")

        with gr.Row():
            folder_input = gr.Textbox(
                label="Folder Path",
                placeholder="/path/to/documents",
                scale=3
            )
            add_folder_btn = gr.Button("➕ Add", scale=1)
            remove_folder_btn = gr.Button("➖ Remove", scale=1)

        folder_dropdown = gr.Dropdown(
            choices=[],
            label="Select Active Folder",
            interactive=True
        )

        index_folder_btn = gr.Button("🔍 Index Selected Folder", variant="primary")
        folder_status = gr.Label(label="Folder Status")

        gr.Markdown("---")
        gr.Markdown("## 💬 Chat")

        chatbox = gr.Chatbot(
            type="messages",
            label="Conversation",
            height=400
        )

        message_input = gr.Textbox(
            label="Your Message",
            placeholder="Ask about your documents...",
            lines=2
        )

        with gr.Row():
            send_btn = gr.Button("📤 Send", variant="primary", scale=4)
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

    # -------------------- Callbacks --------------------
    def do_register(username, password):
        success, msg = register_user(username, password)
        return msg

    def do_login(username, password):
        success, msg = login_user(username, password)
        if success:
            folders = get_folders(username)
            # load or initialize session
            sessions[username] = load_session(username)
            return (
                msg,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(choices=folders, value=folders[0] if folders else None)
            )
        else:
            return msg, gr.update(visible=True), gr.update(visible=False), gr.update(choices=[])

    register_btn.click(
        do_register,
        inputs=[username_input, password_input],
        outputs=login_status
    )

    login_btn.click(
        do_login,
        inputs=[username_input, password_input],
        outputs=[login_status, login_page, main_page, folder_dropdown]
    )

    # Folder actions
    def add_folder_cb(username, folder_path):
        ok, msg, folders = add_folder(username, folder_path)
        return msg, gr.update(choices=folders)

    def remove_folder_cb(username, folder_path):
        ok, msg, folders = remove_folder(username, folder_path)
        return msg, gr.update(choices=folders)

    def index_folder_cb(username, folder_path):
        return index_folder(username, folder_path)

    add_folder_btn.click(
        add_folder_cb,
        inputs=[username_input, folder_input],
        outputs=[folder_status, folder_dropdown]
    )

    remove_folder_btn.click(
        remove_folder_cb,
        inputs=[username_input, folder_dropdown],
        outputs=[folder_status, folder_dropdown]
    )

    index_folder_btn.click(
        index_folder_cb,
        inputs=[username_input, folder_dropdown],
        outputs=folder_status
    )

    # Chat
    send_btn.click(
        chat,
        inputs=[username_input, folder_dropdown, message_input],
        outputs=[chatbox, message_input]
    )

    message_input.submit(
        chat,
        inputs=[username_input, folder_dropdown, message_input],
        outputs=[chatbox, message_input]
    )

    def clear_chat_history(username):
        if username in sessions:
            sessions[username].messages = []
            save_session(username, sessions[username])
        return []

    clear_btn.click(
        clear_chat_history,
        inputs=[username_input],
        outputs=chatbox
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )
