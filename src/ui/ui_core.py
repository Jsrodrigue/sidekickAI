"""
Gradio app
"""
import asyncio
import gradio as gr
from typing import Optional

from src.ui.auth import login_user, register_user
from src.core.sidekick import init_sidekick
from src.ui.handlers import UIHandlers


# -------------------- gl --------------------
sidekick: Optional[object] = None
handlers: Optional[UIHandlers] = None
_init_lock = asyncio.Lock()


async def ensure_initialized():
    """Ensures that Sidekick and handlers are initialized."""
    global sidekick, handlers
    
    if sidekick is not None and handlers is not None:
        return handlers
    
    async with _init_lock:
        if sidekick is None:
            sidekick = await init_sidekick()
        if handlers is None:
            handlers = UIHandlers(sidekick)
    
    return handlers


# -------------------- Wrappers for callbacks --------------------

async def wrapped_index_folder(username, folder):
    h = await ensure_initialized()
    return await h.index_folder(username, folder)


async def wrapped_reindex_folder(username, folder):
    h = await ensure_initialized()
    return await h.reindex_folder(username, folder)


async def wrapped_remove_folder(username, folder):
    h = await ensure_initialized()
    msg, folders = await h.remove_folder(username, folder)
    return msg, gr.update(choices=folders, value=folders[0] if folders else None)


async def wrapped_chat(username, folder, message):
    h = await ensure_initialized()
    return await h.chat(username, folder, message)


def wrapped_clear_chat(username):
    # Sync function for clearing chat
    if handlers and username:
        return handlers.clear_chat(username)
    return []


def wrapped_get_folders(username):
    if handlers:
        return handlers.get_folders(username)
    return []


def wrapped_add_folder(username, folder):
    if handlers:
        msg, folders = handlers.add_folder(username, folder)
        return msg, gr.update(choices=folders, value=folder)
    return "Not initialized", gr.update(choices=[])


# -------------------- Interfaz Gradio --------------------

def create_ui():
    """Crea la interfaz de usuario de Gradio."""
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Sidekick AI - Multi-User RAG Assistant")
        gr.Markdown("Assistant system with RAG, authentication, and folder management.")

        # -------------------- Página de Login --------------------
        with gr.Column(visible=True) as login_page:
            gr.Markdown("## 🔐 Login / Register")
            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter username"
            )
            password_input = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter password"
            )
            login_status = gr.Label(label="Status")
            
            with gr.Row():
                login_btn = gr.Button("🔓 Login", variant="primary")
                register_btn = gr.Button("📝 Register", variant="secondary")

        # -------------------- Página Principal --------------------
        with gr.Column(visible=False) as main_page:
            gr.Markdown("## 📁 Folder Management")
            
            folder_dropdown = gr.Dropdown(
                choices=[],
                label="Select Folder",
                interactive=True
            )
            folder_input = gr.Textbox(
                label="Folder Path",
                placeholder="/path/to/documents"
            )
            folder_status = gr.Label(label="Folder Status")

            with gr.Row():
                add_folder_btn = gr.Button("➕ Add Folder")
                remove_folder_btn = gr.Button("🗑️ Remove Folder")
                index_folder_btn = gr.Button("🔍 Index Folder")
                reindex_folder_btn = gr.Button("🔄 Reindex Folder")

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

        # -------------------- Event Handlers --------------------

        # Registro
        register_btn.click(
            lambda u, p: register_user(u, p)[1],
            inputs=[username_input, password_input],
            outputs=login_status
        )

        # Login
        def handle_login(u, p):
            success, msg = login_user(u, p)
            if success:
                folders = wrapped_get_folders(u)
                return (
                    msg,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(choices=folders, value=folders[0] if folders else None)
                )
            return msg, gr.update(visible=True), gr.update(visible=False), gr.update()

        login_btn.click(
            handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, login_page, main_page, folder_dropdown]
        )

        # Gestión de carpetas
        add_folder_btn.click(
            wrapped_add_folder,
            inputs=[username_input, folder_input],
            outputs=[folder_status, folder_dropdown]
        )

        remove_folder_btn.click(
            wrapped_remove_folder,
            inputs=[username_input, folder_dropdown],
            outputs=[folder_status, folder_dropdown]
        )

        index_folder_btn.click(
            wrapped_index_folder,
            inputs=[username_input, folder_dropdown],
            outputs=folder_status
        )

        reindex_folder_btn.click(
            wrapped_reindex_folder,
            inputs=[username_input, folder_dropdown],
            outputs=folder_status
        )

        # Chat
        send_btn.click(
            wrapped_chat,
            inputs=[username_input, folder_dropdown, message_input],
            outputs=[chatbox, message_input]
        )

        message_input.submit(
            wrapped_chat,
            inputs=[username_input, folder_dropdown, message_input],
            outputs=[chatbox, message_input]
        )

        clear_btn.click(
            wrapped_clear_chat,
            inputs=[username_input],
            outputs=chatbox
        )

    return demo
