import json
import uuid
import asyncio
import gradio as gr
from dotenv import load_dotenv
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from src.gradio_utils.auth import login_user, register_user
from src.sidekick import SidekickState, init_sidekick  # init_sidekick is async
from src.db import get_conn, init_db

load_dotenv()
init_db()

# -------------------- Sidekick init --------------------
sidekick: Optional[object] = None
_sidekick_lock = asyncio.Lock()  # protect concurrent inits

async def ensure_sidekick_ready():
    """Ensure global 'sidekick' is initialized. Safe to call multiple times."""
    global sidekick
    if sidekick is not None:
        return sidekick
    async with _sidekick_lock:
        if sidekick is None:
            # init_sidekick is async and returns an instance
            sidekick_instance = await init_sidekick()
            sidekick = sidekick_instance
    return sidekick

# In-memory session store
sessions = {}

# -------------------- Folder Management (DB) --------------------
def add_folder(username, folder_path):
    if not username or not folder_path:
        return False, "Username and folder path required", []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO folders (username, folder_path) VALUES (?, ?)",
                (username, folder_path))
    conn.commit()
    cur.execute("SELECT folder_path FROM folders WHERE username = ?", (username,))
    folders = [r[0] for r in cur.fetchall()]
    conn.close()
    return True, "Folder added", folders

def remove_folder_db_only(username, folder_path):
    """Remove from DB only and return updated list (no vectorstore cleanup)."""
    if not username or not folder_path:
        return False, "Username and folder path required", []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM folders WHERE username = ? AND folder_path = ?", (username, folder_path))
    conn.commit()
    cur.execute("SELECT folder_path FROM folders WHERE username = ?", (username,))
    folders = [r[0] for r in cur.fetchall()]
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

# -------------------- Session helpers --------------------
def save_session(username: str, state: SidekickState):
    """Save session state to DB (serialized JSON)."""
    conn = get_conn()
    cur = conn.cursor()
    messages_data = []
    for m in getattr(state, "messages", []):
        msg_type = type(m).__name__
        msg_dict = {"type": msg_type, "content": getattr(m, "content", "")}
        if hasattr(m, "tool_calls") and getattr(m, "tool_calls"):
            try:
                msg_dict["tool_calls"] = [
                    {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                    for tc in m.tool_calls
                ]
            except Exception:
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
    """Load session from DB and return SidekickState with defaults."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT data FROM sessions WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return SidekickState()

    data = json.loads(row[0])
    messages = []
    for m in data.get("messages", []):
        msg_type = m.get("type", "")
        content = m.get("content", "") or ""
        if msg_type == "HumanMessage":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

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

# -------------------- Async folder indexing --------------------
async def index_folder_async(username, folder_path, force_reindex=False):
    """Index a folder asynchronously; ensures sidekick ready and runs blocking work in thread."""
    if not folder_path:
        return "❌ Select a folder first"

    sk = await ensure_sidekick_ready()

    # if already indexed and not forcing, reuse
    # note: sidekick.retriever_registry keys are stored as normalized/absolute inside Sidekick
    if folder_path in getattr(sk, "retriever_registry", {}) and not force_reindex:
        return f"✅ Folder already indexed: {folder_path}"

    # Use thread to avoid blocking
    try:
        result = await asyncio.to_thread(sk.load_and_register_directory, folder_path, force_reindex)
    except Exception as e:
        return f"❌ Indexing failed: {e}"

    # Update session state (if exists)
    if username in sessions:
        state = sessions[username]
        state.current_directory = folder_path
        if folder_path not in state.indexed_directories:
            state.indexed_directories.append(folder_path)
        save_session(username, state)

    return result

# -------------------- Async Chat --------------------
async def chat_async(user_id, folder_path, message):
    if not user_id:
        return [], "Please log in first"
    message = (message or "").strip()
    if not message:
        return [], ""

    sk = await ensure_sidekick_ready()

    state = sessions.get(user_id) or load_session(user_id)
    sessions[user_id] = state

    # Change folder if selected
    if folder_path and folder_path != state.current_directory:
        state.current_directory = folder_path
        # index if missing
        if folder_path not in sk.retriever_registry:
            await asyncio.to_thread(sk.load_and_register_directory, folder_path)
        if folder_path not in state.indexed_directories:
            state.indexed_directories.append(folder_path)

    # Append user message
    state.messages.append(HumanMessage(content=message))

    # Default success criteria
    if not state.success_criteria:
        state.success_criteria = "Answer the user's question completely and accurately."

    # Invoke LangGraph
    try:
        result_state = await sk.graph.ainvoke(state, {"configurable": {"thread_id": state.thread_id}})
        # langgraph may return SidekickState-like object or dict; handle both
        if isinstance(result_state, dict):
            # merge known fields into pydantic model
            if "messages" in result_state:
                # convert raw messages to HumanMessage/AIMessage if needed
                # here we assume messages are already in the proper form (langgraph tends to)
                state.messages = result_state["messages"]
            state.evaluation_history = result_state.get("evaluation_history", state.evaluation_history)
            state.criteria_met = result_state.get("criteria_met", state.criteria_met)
            state.needs_user_input = result_state.get("needs_user_input", state.needs_user_input)
            sessions[user_id] = state
            save_session(user_id, state)
            result_state_obj = state
        else:
            sessions[user_id] = result_state
            save_session(user_id, result_state)
            result_state_obj = result_state

        # Build chat history for Gradio
        chat_history = []
        for m in result_state_obj.messages:
            if isinstance(m, HumanMessage):
                chat_history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                if not getattr(m, "content", "").startswith("💭"):
                    chat_history.append({"role": "assistant", "content": m.content})
        return chat_history, ""
    except Exception as e:
        # push error into messages and return what we have
        state.messages.append(AIMessage(content=f"Error processing request: {e}"))
        sessions[user_id] = state
        save_session(user_id, state)
        chat_history = []
        for m in state.messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_history.append({"role": role, "content": m.content})
        return chat_history, ""

# -------------------- Gradio UI --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Sidekick AI - Multi-User RAG Assistant")
    gr.Markdown("Assistant system with RAG, authentication, and folder management.")

    # Login Page
    with gr.Column(visible=True) as login_page:
        gr.Markdown("## 🔐 Login / Register")
        username_input = gr.Textbox(label="Username", placeholder="Enter username")
        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
        login_status = gr.Label(label="Status")
        with gr.Row():
            login_btn = gr.Button("🔓 Login", variant="primary")
            register_btn = gr.Button("📝 Register", variant="secondary")

    # Main Page
    with gr.Column(visible=False) as main_page:
        gr.Markdown("## 📁 Folder Management")
        folder_dropdown = gr.Dropdown(choices=[], label="Select Folder", interactive=True)
        folder_input = gr.Textbox(label="Folder Path", placeholder="/path/to/documents")
        folder_status = gr.Label(label="Folder Status")

        with gr.Row():
            add_folder_btn = gr.Button("➕ Add Folder")
            remove_folder_btn = gr.Button("🗑️ Remove Folder")
            index_folder_btn = gr.Button("🔍 Index Folder")
            reindex_folder_btn = gr.Button("🔄 Reindex Folder")

        gr.Markdown("---")
        gr.Markdown("## 💬 Chat")
        chatbox = gr.Chatbot(type="messages", label="Conversation", height=400)
        message_input = gr.Textbox(label="Your Message", placeholder="Ask about your documents...", lines=2)
        with gr.Row():
            send_btn = gr.Button("📤 Send", variant="primary", scale=4)
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

    # -------------------- Callbacks --------------------
    register_btn.click(
        lambda u, p: register_user(u, p)[1],
        inputs=[username_input, password_input],
        outputs=login_status
    )

    login_btn.click(
        lambda u, p: (
            login_user(u, p)[1],
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(choices=get_folders(u), value=get_folders(u)[0] if get_folders(u) else None)
        ),
        inputs=[username_input, password_input],
        outputs=[login_status, login_page, main_page, folder_dropdown]
    )

    add_folder_btn.click(
        lambda u, f: (add_folder(u, f)[1], gr.update(choices=add_folder(u, f)[2], value=f)),
        inputs=[username_input, folder_input],
        outputs=[folder_status, folder_dropdown]
    )

    # Remove folder: remove from DB and also remove vectorstore if Sidekick indexed it.
    async def remove_folder_cb(username, selected_folder):
        ok, msg, folders = remove_folder_db_only(username, selected_folder)
        # attempt to remove vectorstore via sidekick if available
        try:
            sk = await ensure_sidekick_ready()
            # sidekick.remove_directory handles safely when folder not present
            await asyncio.to_thread(sk.remove_directory, selected_folder)
        except Exception:
            # if sidekick not ready or fails, ignore but log
            print(f"⚠️ remove_folder_cb: sidekick cleanup failed for {selected_folder}")
        return msg, gr.update(choices=folders, value=folders[0] if folders else None)

    remove_folder_btn.click(
        remove_folder_cb,
        inputs=[username_input, folder_dropdown],
        outputs=[folder_status, folder_dropdown]
    )

    # Index and Reindex call the async function directly
    index_folder_btn.click(
        index_folder_async,
        inputs=[username_input, folder_dropdown],
        outputs=folder_status
    )

    async def reindex_cb(username, folder):
        return await index_folder_async(username, folder, force_reindex=True)

    reindex_folder_btn.click(
        reindex_cb,
        inputs=[username_input, folder_dropdown],
        outputs=folder_status
    )


    send_btn.click(
        chat_async,
        inputs=[username_input, folder_dropdown, message_input],
        outputs=[chatbox, message_input]
    )

    message_input.submit(
        chat_async,
        inputs=[username_input, folder_dropdown, message_input],
        outputs=[chatbox, message_input]
    )

    def clear_chat_history(username):
        if username in sessions:
            sessions[username].messages = []
            save_session(username, sessions[username])
            return []
        return []

    clear_btn.click(
        clear_chat_history,
        inputs=[username_input],
        outputs=chatbox
    )

if __name__ == "__main__":
    # Let Gradio create the event loop and run async callbacks; don't call asyncio.run here.
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, share=False)
