"""
Gradio app - Sidekick AI
"""

from typing import Optional
from pathlib import Path

import gradio as gr

from src.ui.auth import login_user, register_user
from src.ui.gradio_wrappers import (
    MAX_RETRIEVAL_K,
    wrapped_get_folders,
    wrapped_add_folder,
    wrapped_remove_folder,
    wrapped_index_folder,
    wrapped_reindex_folder,
    wrapped_chat,
    wrapped_load_chat,
    wrapped_clear_chat,
    format_active_folder_label,
)


def create_ui(css_path: str | None = None):
    """
    Create the full Gradio interface with optional CSS (loaded only if file exists).
    """

    css = ""

    # Load CSS if a path is explicitly provided
    if css_path:
        css_file = Path(css_path)
        if css_file.exists():
            css = css_file.read_text()
        else:
            print(f"[Warning] CSS file not found: {css_path}. Running without custom styles.")

    with gr.Blocks(css=css) as demo:
        # Session state (None = not logged in)
        logged_in_user = gr.State(value=None)

        gr.Markdown("# Sidekick AI")
        gr.Markdown(
            "Register your document folders, and chat with your own knowledge base.\n\n"
        )

        # ---------- Login Page ----------
        with gr.Column(visible=True, elem_id="login_container") as login_page:
            gr.Markdown("## Login / Register")

            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter username",
            )

            password_input = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter password",
            )

            login_status = gr.Label(
                label="Status",
            )

            with gr.Row():
                login_btn = gr.Button("🔓 Login", variant="primary")
                register_btn = gr.Button("📝 Register", variant="secondary")

        # ---------- Main Page ----------
        with gr.Column(visible=False) as main_page:
            # Top bar with logout
            with gr.Row():
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

                    # ---- Indexing parameters (collapsible) ----
                    with gr.Accordion("Indexing parameters (advanced)", open=False):
                        chunk_size_slider = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=600,
                            step=64,
                            label="Chunk size",
                        )

                        chunk_overlap_slider = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=20,
                            step=10,
                            label="Chunk overlap",
                        )

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
                        # "Set as Active" not needed: the dropdown itself chooses the active folder

                # ----- RIGHT: Chat -----
                with gr.Column(scale=2):
                    gr.Markdown("### 2️⃣ Chat with your documents")

                    # RAG retrieval settings (collapsible)
                    with gr.Accordion("RAG retrieval settings", open=False):
                        retrieval_k_dropdown = gr.Dropdown(
                            choices=list(range(1, MAX_RETRIEVAL_K + 1)),
                            value=5,
                            label="Top-k documents per query",
                        )

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
                gr.update(),                    # folder_dropdown unchanged
                None,                           # logged_in_user
                [],                             # empty chatbox
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
            inputs=[
                logged_in_user,
                folder_dropdown,
                chunk_size_slider,
                chunk_overlap_slider,
            ],
            outputs=folder_status,
        )

        # Force reindex selected folder
        reindex_folder_btn.click(
            wrapped_reindex_folder,
            inputs=[
                logged_in_user,
                folder_dropdown,
                chunk_size_slider,
                chunk_overlap_slider,
            ],
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
            inputs=[
                logged_in_user,
                folder_dropdown,
                message_input,
                retrieval_k_dropdown,
            ],
            outputs=[chatbox, message_input],
        )

        # Pressing Enter in the textbox also sends
        message_input.submit(
            wrapped_chat,
            inputs=[
                logged_in_user,
                folder_dropdown,
                message_input,
                retrieval_k_dropdown,
            ],
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
            """
            Reset fields, state, and go back to the login screen.
            """
            return (
                "",   # username_input
                "",   # password_input
                "Logged out",  # login_status
                gr.update(visible=True),    # login_page visible
                gr.update(visible=False),   # main_page hidden
                [],                         # empty chatbox
                gr.update(choices=[], value=None),  # empty folder_dropdown
                format_active_folder_label(None),   # active folder label
                None,  # logged_in_user (State) = logged out
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
