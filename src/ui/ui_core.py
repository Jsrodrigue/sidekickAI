"""
Gradio app - Sidekick AI
"""
from pathlib import Path
from typing import Optional

import gradio as gr

from src.ui.auth import login_user, register_user
from src.ui.gradio_wrappers import (
    MAX_RETRIEVAL_K,
    format_active_folder_label,
    wrapped_add_folder,
    wrapped_chat,
    wrapped_clear_chat,
    wrapped_get_folders,
    wrapped_index_folder,
    wrapped_load_chat,
    wrapped_reindex_folder,
    wrapped_remove_folder,
)

# Sentinel label for "no folder" mode in the dropdown
NO_FOLDER_LABEL = "None"


def create_ui(css_path: str | None = None):
    """
    Create the full Gradio interface with optional CSS (loaded only if file exists).
    """
    css = ""

    if css_path:
        css_file = Path(css_path)
        if css_file.exists():
            css = css_file.read_text()
        else:
            print(
                f"[Warning] CSS file not found: {css_path}. Running without custom styles."
            )

    with gr.Blocks(css=css) as demo:
        # Session state (None = not logged in)
        logged_in_user = gr.State(value=None)

        gr.Markdown("# Sidekick AI")
        gr.Markdown(
            "Register your document folders, and chat with your own knowledge base.\n\n"
            "You can also chat with no active folder (RAG disabled)."
        )

        # ---------- Login Page ----------
        with gr.Column(visible=True, elem_id="login_container") as login_page:
            gr.Markdown("## Login / Register")

            username_input = gr.Textbox(label="Username", placeholder="Enter username")
            password_input = gr.Textbox(
                label="Password", type="password", placeholder="Enter password"
            )

            login_status = gr.Label(label="Status")

            with gr.Row():
                login_btn = gr.Button("🔓 Login", variant="primary")
                register_btn = gr.Button("📝 Register", variant="secondary")

        # ---------- Main Page ----------
        with gr.Column(visible=False) as main_page:
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
                        choices=[NO_FOLDER_LABEL],
                        value=NO_FOLDER_LABEL,
                        label="Indexed Folders (active for chat)",
                        interactive=True,
                        info=(
                            f"Select a folder to enable RAG, or choose '{NO_FOLDER_LABEL}' "
                            "to chat without documents."
                        ),
                    )

                    active_folder_label = gr.Markdown(format_active_folder_label(None))

                    with gr.Row():
                        index_folder_btn = gr.Button("🔍 Index Selected (if needed)")
                        reindex_folder_btn = gr.Button("🔄 Force Reindex")
                    with gr.Row():
                        remove_folder_btn = gr.Button(
                            "🗑️ Remove Folder", variant="secondary"
                        )

                # ----- RIGHT: Chat -----
                with gr.Column(scale=2):
                    gr.Markdown("### 2️⃣ Chat with Sidekick")

                    with gr.Accordion("RAG retrieval settings", open=False):
                        retrieval_k_dropdown = gr.Dropdown(
                            choices=list(range(1, MAX_RETRIEVAL_K + 1)),
                            value=5,
                            label="Top-k documents per query",
                            info="Only used when a folder is selected (RAG enabled).",
                        )

                    with gr.Accordion("Tool settings", open=False):
                        tool_selector = gr.CheckboxGroup(
                            choices=[
                                "rag",
                                "files",
                                "web_search",
                                "python",
                                "wikipedia",
                            ],
                            value=["rag", "files", "web_search", "python", "wikipedia"],
                            label="Enabled tool groups",
                            info="Select which groups of tools Sidekick is allowed to use.",
                        )

                    chatbox = gr.Chatbot(
                        type="messages", label="Conversation", height=400
                    )

                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask anything, with or without an active folder...",
                        lines=2,
                    )

                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary", scale=4)
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

        # ---------- Event Handlers ----------

        register_btn.click(
            lambda u, p: register_user(u, p)[1],
            inputs=[username_input, password_input],
            outputs=login_status,
        )

        async def handle_login(u, p):
            success, msg = login_user(u, p)
            if success:
                folders = await wrapped_get_folders(u)

                history = await wrapped_load_chat(u, None)
                label = format_active_folder_label(None)

                dropdown_update = gr.update(
                    choices=[NO_FOLDER_LABEL] + folders,
                    value=NO_FOLDER_LABEL,
                )

                return (
                    msg,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    dropdown_update,
                    u,
                    history,
                    label,
                )

            return (
                msg,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(),
                None,
                [],
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

        async def handle_add_folder(username, folder_path):
            if not username or not folder_path:
                return "Not logged in or empty folder path", gr.update()

            msg, dropdown_update = await wrapped_add_folder(username, folder_path)

            if isinstance(dropdown_update, dict) and "choices" in dropdown_update:
                choices = dropdown_update["choices"]
                if NO_FOLDER_LABEL not in choices:
                    choices = [NO_FOLDER_LABEL] + choices
                dropdown_update["choices"] = choices

            return msg, dropdown_update

        add_folder_btn.click(
            handle_add_folder,
            inputs=[logged_in_user, folder_input],
            outputs=[folder_status, folder_dropdown],
        )

        async def handle_remove_folder(username, folder):
            if not username or not folder or folder == NO_FOLDER_LABEL:
                return "Select a real folder to remove", gr.update(
                    choices=[NO_FOLDER_LABEL], value=NO_FOLDER_LABEL
                )

            msg, dropdown_update = await wrapped_remove_folder(username, folder)

            if isinstance(dropdown_update, dict) and "choices" in dropdown_update:
                choices = dropdown_update["choices"]
                if NO_FOLDER_LABEL not in choices:
                    choices = [NO_FOLDER_LABEL] + choices

                value = dropdown_update.get("value") or NO_FOLDER_LABEL
                dropdown_update["choices"] = choices
                dropdown_update["value"] = value

            return msg, dropdown_update

        remove_folder_btn.click(
            handle_remove_folder,
            inputs=[logged_in_user, folder_dropdown],
            outputs=[folder_status, folder_dropdown],
        )

        async def handle_index_folder(username, folder, chunk_size, chunk_overlap):
            if not username or not folder or folder == NO_FOLDER_LABEL:
                return "Select a real folder to index"
            return await wrapped_index_folder(
                username, folder, chunk_size, chunk_overlap
            )

        index_folder_btn.click(
            handle_index_folder,
            inputs=[
                logged_in_user,
                folder_dropdown,
                chunk_size_slider,
                chunk_overlap_slider,
            ],
            outputs=folder_status,
        )

        async def handle_reindex_folder(username, folder, chunk_size, chunk_overlap):
            if not username or not folder or folder == NO_FOLDER_LABEL:
                return "Select a real folder to reindex"
            return await wrapped_reindex_folder(
                username, folder, chunk_size, chunk_overlap
            )

        reindex_folder_btn.click(
            handle_reindex_folder,
            inputs=[
                logged_in_user,
                folder_dropdown,
                chunk_size_slider,
                chunk_overlap_slider,
            ],
            outputs=folder_status,
        )

        async def handle_folder_change(username: Optional[str], folder: Optional[str]):
            if not username:
                return [], format_active_folder_label(None)

            if folder == NO_FOLDER_LABEL:
                history = await wrapped_load_chat(username, None)
                return history, format_active_folder_label(None)

            history = await wrapped_load_chat(username, folder)
            return history, format_active_folder_label(folder)

        folder_dropdown.change(
            fn=handle_folder_change,
            inputs=[logged_in_user, folder_dropdown],
            outputs=[chatbox, active_folder_label],
        )

        # ----- Chat events -----

        async def handle_chat(username, folder, message, top_k, enabled_tools):
            if folder == NO_FOLDER_LABEL:
                folder = None
            return await wrapped_chat(username, folder, message, top_k, enabled_tools)

        send_btn.click(
            handle_chat,
            inputs=[
                logged_in_user,
                folder_dropdown,
                message_input,
                retrieval_k_dropdown,
                tool_selector,
            ],
            outputs=[chatbox, message_input],
        )

        message_input.submit(
            handle_chat,
            inputs=[
                logged_in_user,
                folder_dropdown,
                message_input,
                retrieval_k_dropdown,
                tool_selector,
            ],
            outputs=[chatbox, message_input],
        )

        async def handle_clear_chat(username, folder):
            if folder == NO_FOLDER_LABEL:
                folder = None
            return await wrapped_clear_chat(username, folder)

        clear_btn.click(
            handle_clear_chat,
            inputs=[logged_in_user, folder_dropdown],
            outputs=chatbox,
        )

        # ----- Logout event -----

        def handle_logout():
            return (
                "",
                "",
                "Logged out",
                gr.update(visible=True),
                gr.update(visible=False),
                [],
                gr.update(choices=[NO_FOLDER_LABEL], value=NO_FOLDER_LABEL),
                format_active_folder_label(None),
                None,
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
