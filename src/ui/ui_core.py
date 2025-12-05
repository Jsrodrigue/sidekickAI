"""
Gradio app - Sidekick AI
"""

import asyncio
from typing import Optional

import gradio as gr

from src.ui.auth import login_user, register_user
from src.core.sidekick import init_sidekick
from src.ui.ui_handlers import UIHandlers
from src.services.session_service import SessionService
from src.services.folder_service import FolderService
from src.services.sidekick_service import SidekickService
from src.db.session_repository import SessionRepository

# -------------------- Globals --------------------

sidekick: Optional[object] = None
handlers: Optional[UIHandlers] = None
_init_lock = asyncio.Lock()


# -------------------- Initialization --------------------

async def ensure_initialized() -> UIHandlers:
    """Ensure that Sidekick and UIHandlers are initialized exactly once."""
    global sidekick, handlers

    # Fast path: already initialized
    if sidekick is not None and handlers is not None:
        return handlers

    # Slow path: initialize with a lock
    async with _init_lock:
        if sidekick is None:
            sidekick = await init_sidekick()

        if handlers is None:
            # Create services
            session_repo = SessionRepository()
            session_service = SessionService(session_repo)
            folder_service = FolderService(sidekick)
            sidekick_service = SidekickService(sidekick)

            # Wire them into a single UIHandlers instance
            handlers = UIHandlers(
                session_service=session_service,
                folder_service=folder_service,
                sidekick_service=sidekick_service,
            )

    return handlers


# -------------------- Wrappers (for Gradio) --------------------

async def wrapped_get_folders(username: Optional[str]):
    if not username:
        return []
    h = await ensure_initialized()
    return h.get_folders(username)


async def wrapped_add_folder(username: Optional[str], folder: str):
    if not username:
        return "Not logged in", gr.update()
    h = await ensure_initialized()
    msg, folders = await h.add_folder(username, folder)
    # Return status label + updated dropdown choices
    return msg, gr.update(choices=folders, value=folder)


async def wrapped_remove_folder(username: Optional[str], folder: Optional[str]):
    if not username or not folder:
        return "Not logged in", gr.update(choices=[], value=None)
    h = await ensure_initialized()
    msg, folders = await h.remove_folder(username, folder)
    return msg, gr.update(
        choices=folders,
        value=folders[0] if folders else None,
    )


async def wrapped_index_folder(username: Optional[str], folder: Optional[str]):
    if not username or not folder:
        return "Not logged in"
    h = await ensure_initialized()
    # Returns a simple status string
    return await h.index_folder(username, folder)


async def wrapped_reindex_folder(username: Optional[str], folder: Optional[str]):
    if not username or not folder:
        return "Not logged in"
    h = await ensure_initialized()
    # Returns a simple status string
    return await h.reindex_folder(username, folder)


async def wrapped_chat(username: Optional[str], folder: Optional[str], message: str):
    """
    Wrapper for chat: returns messages in Gradio type="messages" format
    plus an empty string to clear the textbox.
    """
    if not username or not folder:
        # Mensaje de aviso si alguien consigue mandar sin estar bien logueado
        return (
            [
                {
                    "role": "assistant",
                    "content": "Please log in and select a folder before chatting.",
                }
            ],
            "",
        )
    h = await ensure_initialized()
    history, _ = await h.chat(username, folder, message)
    # 1st -> Chatbot (list of {role, content}), 2nd -> textbox (cleared)
    return history, ""


async def wrapped_load_chat(username: Optional[str], folder: Optional[str]):
    """
    Load existing chat history for (username, folder).
    """
    if not username or not folder:
        return []
    h = await ensure_initialized()
    history = await h.load_chat(username, folder)
    return history


async def wrapped_clear_chat(username: Optional[str], folder: Optional[str]):
    """
    Clears the server-side session messages for this (username, folder)
    and returns an empty list for the Chatbot.
    """
    if not username or not folder:
        return []
    h = await ensure_initialized()
    messages = h.clear_chat(username, folder)
    # Chatbot(type="messages") expects a list of messages (dicts), so [] is valid
    return messages


def format_active_folder_label(folder: Optional[str]) -> str:
    """Helper to format the 'active folder' label."""
    if folder:
        return f"**Active folder for chat:** `{folder}`"
    return "**Active folder for chat:** _none selected_"


# -------------------- UI --------------------

def create_ui():
    """Create the full Gradio interface."""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Estado de sesión (None = no logueado)
        logged_in_user = gr.State(value=None)

        gr.Markdown("# 📚 Sidekick AI - Multi-User RAG Assistant")
        gr.Markdown(
            "Log in, register some document folders, and chat with your own knowledge base.\n\n"
            "**Workflow:** 1) Add & index a folder → 2) Pick it as active → 3) Ask questions in chat."
        )

        # ---------- Login Page ----------
        with gr.Column(visible=True) as login_page:
            gr.Markdown("## 🔐 Login / Register")

            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter username",
            )
            password_input = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter password",
            )
            login_status = gr.Label(label="Status")

            with gr.Row():
                login_btn = gr.Button("🔓 Login", variant="primary")
                register_btn = gr.Button("📝 Register", variant="secondary")

        # ---------- Main Page ----------
        with gr.Column(visible=False) as main_page:
            # Barra superior con logout
            with gr.Row():
                gr.Markdown("## 👋 Welcome to Sidekick")
                logout_btn = gr.Button("🚪 Logout", variant="secondary")

            with gr.Row():
                # ----- LEFT: Folder Management -----
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### 1️⃣ Manage your folders")

                    folder_input = gr.Textbox(
                        label="Folder path to add & index",
                        placeholder="/path/to/documents",
                    )

                    with gr.Row():
                        add_folder_btn = gr.Button(
                            "➕ Add & Index Folder", variant="primary"
                        )

                    folder_status = gr.Label(label="Folder Status")

                    gr.Markdown("#### Indexed folders")

                    folder_dropdown = gr.Dropdown(
                        choices=[],
                        label="Indexed Folders (active for chat)",
                        interactive=True,
                    )

                    # Active folder indicator
                    active_folder_label = gr.Markdown(
                        format_active_folder_label(None)
                    )

                    with gr.Row():
                        index_folder_btn = gr.Button("🔍 Index Selected (if needed)")
                        reindex_folder_btn = gr.Button("🔄 Force Reindex")
                    with gr.Row():
                        remove_folder_btn = gr.Button(
                            "🗑️ Remove Folder", variant="secondary"
                        )
                        # Eliminado el botón de "Set as Active" porque ya se hace con el dropdown

                # ----- RIGHT: Chat -----
                with gr.Column(scale=2):
                    gr.Markdown("### 2️⃣ Chat with your documents")

                    chatbox = gr.Chatbot(
                        type="messages",
                        label="Conversation",
                        height=400,
                    )
                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask something about the active folder...",
                        lines=2,
                    )

                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary", scale=4)
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

        # ---------- Event Handlers ----------

        # Registration: only show result message in the status label
        register_btn.click(
            lambda u, p: register_user(u, p)[1],
            inputs=[username_input, password_input],
            outputs=login_status,
        )

        # Login handler (async)
        async def handle_login(u, p):
            success, msg = login_user(u, p)
            if success:
                folders = await wrapped_get_folders(u)

                if folders:
                    active_folder = folders[0]
                    history = await wrapped_load_chat(u, active_folder)
                    label = format_active_folder_label(active_folder)
                    dropdown_update = gr.update(
                        choices=folders,
                        value=active_folder,
                    )
                else:
                    active_folder = None
                    history = []
                    label = format_active_folder_label(None)
                    dropdown_update = gr.update(choices=[], value=None)

                return (
                    msg,                         # login_status
                    gr.update(visible=False),     # hide login_page
                    gr.update(visible=True),      # show main_page
                    dropdown_update,              # folder_dropdown
                    u,                            # logged_in_user (State)
                    history,                      # chatbox
                    label,                        # active_folder_label
                )

            # Login failed: stay on login page
            return (
                msg,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(),                    # folder_dropdown sin cambios
                None,                           # logged_in_user
                [],                             # chatbox vacío
                format_active_folder_label(None),
            )

        login_btn.click(
            handle_login,
            inputs=[username_input, password_input],
            outputs=[
                login_status,
                login_page,
                main_page,
                folder_dropdown,
                logged_in_user,
                chatbox,
                active_folder_label,
            ],
        )

        # ----- Folder Management events -----

        # Add & index a new folder
        add_folder_btn.click(
            wrapped_add_folder,
            inputs=[logged_in_user, folder_input],
            outputs=[folder_status, folder_dropdown],
        )

        # Remove selected folder
        remove_folder_btn.click(
            wrapped_remove_folder,
            inputs=[logged_in_user, folder_dropdown],
            outputs=[folder_status, folder_dropdown],
        )

        # Index selected folder (if not already indexed)
        index_folder_btn.click(
            wrapped_index_folder,
            inputs=[logged_in_user, folder_dropdown],
            outputs=folder_status,
        )

        # Force reindex selected folder
        reindex_folder_btn.click(
            wrapped_reindex_folder,
            inputs=[logged_in_user, folder_dropdown],
            outputs=folder_status,
        )

        # When user changes folder → load that folder's history + update label
        async def handle_folder_change(username: Optional[str], folder: Optional[str]):
            if not username or not folder:
                return [], format_active_folder_label(None)
            history = await wrapped_load_chat(username, folder)
            label = format_active_folder_label(folder)
            return history, label

        folder_dropdown.change(
            fn=handle_folder_change,
            inputs=[logged_in_user, folder_dropdown],
            outputs=[chatbox, active_folder_label],
        )

        # ----- Chat events -----

        # Chat send button
        send_btn.click(
            wrapped_chat,
            inputs=[logged_in_user, folder_dropdown, message_input],
            outputs=[chatbox, message_input],
        )

        # Pressing Enter in the textbox also sends
        message_input.submit(
            wrapped_chat,
            inputs=[logged_in_user, folder_dropdown, message_input],
            outputs=[chatbox, message_input],
        )

        # Clear chat button (for the current folder)
        clear_btn.click(
            wrapped_clear_chat,
            inputs=[logged_in_user, folder_dropdown],
            outputs=chatbox,
        )

        # ----- Logout event -----

        def handle_logout():
            # Limpiar campos básicos, estado y volver a la pantalla de login
            return (
                "",   # username_input
                "",   # password_input
                "Logged out",  # login_status
                gr.update(visible=True),    # login_page visible
                gr.update(visible=False),   # main_page oculto
                [],                         # chatbox vacío
                gr.update(choices=[], value=None),  # folder_dropdown vacío
                format_active_folder_label(None),   # etiqueta de carpeta activa
                None,  # logged_in_user (State) = sesión cerrada
            )

        logout_btn.click(
            handle_logout,
            inputs=[],
            outputs=[
                username_input,
                password_input,
                login_status,
                login_page,
                main_page,
                chatbox,
                folder_dropdown,
                active_folder_label,
                logged_in_user,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch()
