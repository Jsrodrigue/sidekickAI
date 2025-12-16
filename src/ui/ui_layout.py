"""
Gradio app - Sidekick AI
"""
from pathlib import Path

import gradio as gr

from src.core.sidekick import SEARCH_K
from src.ui.ui_events import UIRefs, wire_events
from src.ui.text_utils import format_active_folder_label

# Sentinel label for "no folder" mode in the dropdown
NO_FOLDER_LABEL = "None"

# Max retrieval k (should match retriever search_kwargs["k"])
MAX_RETRIEVAL_K = SEARCH_K


def create_ui(css_path: str | None = None):
    """
    Create the full Gradio interface with optional CSS (loaded only if file exists).
    """



    if css_path:
        css_file = Path(css_path)
        if css_file.exists():
            css = css_file.read_text()
        else:
            print(f"[Warning] CSS file not found: {css_path}. Running without custom styles.")

    with gr.Blocks(css=css) as demo:
        # Session state (None = not logged in)
        logged_in_user = gr.State(value=None)

        # Header: title + logout next to it
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown("# Sidekick AI", elem_id="header_title")
            with gr.Column(scale=1, min_width=120):
                logout_btn = gr.Button("🚪 Logout", variant="secondary", visible=False)


        # ---------- Login Page ----------
        with gr.Column(visible=True, elem_id="login_container") as login_page:
            gr.Markdown("## Login / Register")

            username_input = gr.Textbox(label="Username", placeholder="Enter username")
            password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")

            login_status = gr.Label(label="Status")

            with gr.Row():
                login_btn = gr.Button("🔓 Login", variant="primary")
                register_btn = gr.Button("📝 Register", variant="secondary")

        # ---------- Main Page ----------
        with gr.Column(visible=False) as main_page:
            with gr.Row():
                # ----- LEFT: Folder Management -----
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### 1️⃣ Manage your folders")

                    folder_input = gr.Textbox(
                        label="Folder path to add & index",
                        placeholder="/path/to/documents",
                    )

                    with gr.Row():
                        add_folder_btn = gr.Button("➕ Add & Index Folder", variant="primary")

                    folder_status = gr.Label(label="Folder Status")

                    with gr.Accordion("Indexing parameters (advanced)", open=False):
                        chunk_size_slider = gr.Slider(128, 2048, value=600, step=64, label="Chunk size")
                        chunk_overlap_slider = gr.Slider(0, 512, value=20, step=10, label="Chunk overlap")

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
                        remove_folder_btn = gr.Button("🗑️ Remove Folder", variant="secondary")

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
                            choices=["rag", "files", "web_search", "python", "wikipedia"],
                            value=["rag", "files", "web_search", "python", "wikipedia"],
                            label="Enabled tool groups",
                            info="Select which groups of tools Sidekick is allowed to use.",
                        )

                    chatbox = gr.Chatbot(type="messages", label="Conversation", height=400)

                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask anything, with or without an active folder...",
                        lines=2,
                    )

                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary", scale=4)
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

        # ---------- Wire events ----------
        refs = UIRefs(
            username_input=username_input,
            password_input=password_input,
            login_status=login_status,
            login_page=login_page,
            main_page=main_page,
            login_btn=login_btn,
            register_btn=register_btn,
            logout_btn=logout_btn,
            logged_in_user=logged_in_user,
            chatbox=chatbox,
            active_folder_label=active_folder_label,
            folder_input=folder_input,
            folder_status=folder_status,
            folder_dropdown=folder_dropdown,
            add_folder_btn=add_folder_btn,
            remove_folder_btn=remove_folder_btn,
            index_folder_btn=index_folder_btn,
            reindex_folder_btn=reindex_folder_btn,
            chunk_size_slider=chunk_size_slider,
            chunk_overlap_slider=chunk_overlap_slider,
            message_input=message_input,
            send_btn=send_btn,
            clear_btn=clear_btn,
            retrieval_k_dropdown=retrieval_k_dropdown,
            tool_selector=tool_selector,
        )

        wire_events(refs, NO_FOLDER_LABEL=NO_FOLDER_LABEL)

    return demo
