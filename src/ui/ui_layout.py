"""
Gradio app - Sidekick AI
"""
from pathlib import Path

import gradio as gr

from src.core.sidekick import SEARCH_K
from src.ui.ui_events import UIRefs, wire_events
from src.ui.text_utils import format_active_folder_label

NO_FOLDER_LABEL = "None"
MAX_RETRIEVAL_K = SEARCH_K


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
            print(f"[Warning] CSS file not found: {css_path}. Running without custom styles.")

    safe_css = """
    /* Force the two-column row to align items to the top */
    #main_row { align-items: flex-start !important; }

    /* Tighten markdown spacing inside our columns */
    #settings_col .markdown p,
    #settings_col .markdown h2,
    #settings_col .markdown h3,
    #chat_col .markdown p,
    #chat_col .markdown h2,
    #chat_col .markdown h3 {
      margin: 6px 0 !important;
    }

    /* Responsive chat height */
    #chatbox { height: clamp(260px, 50vh, 520px) !important; }
    """

    css = (css or "") + "\n" + safe_css

    with gr.Blocks(css=css) as demo:
        logged_in_user = gr.State(value=None)

        # Header
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown("# Sidekick AI", elem_id="header_title")
            with gr.Column(scale=1, min_width=120):
                logout_btn = gr.Button("🚪 Logout", variant="secondary", visible=False)
        gr.Markdown("---")

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
            with gr.Row(equal_height=True, elem_id="main_row"):

                # ===== Left column: CHAT =====
                with gr.Column(scale=2, min_width=520, elem_id="chat_col"):
                    gr.Markdown("## 💬 Chat")

                    chatbox = gr.Chatbot(type="messages", label="Conversation", elem_id="chatbox")

                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask anything, with or without an active folder...",
                        lines=2,
                    )

                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary", scale=4)
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

                # ===== Right column: SETTINGS =====
                with gr.Column(scale=1, min_width=320, elem_id="settings_col"):
                    with gr.Column(elem_classes=["settings_inner"]):
                        gr.Markdown("## ⚙️ Settings")

                        # Folder menu + buttons together in a Group
                        with gr.Group():
                            folder_dropdown = gr.Dropdown(
                                choices=[NO_FOLDER_LABEL],
                                value=NO_FOLDER_LABEL,
                                label="Indexed Folders (active for chat)",
                                interactive=True,
                            )

                            active_folder_label = gr.Markdown(
                                format_active_folder_label(None),
                                visible=False,
                            )

                            with gr.Row():
                                index_folder_btn = gr.Button("🔍 Reindex Folder")
                                remove_folder_btn = gr.Button("🗑️ Remove Folder", variant="secondary")

                        # Add Folder Accordion
                        with gr.Accordion("➕ Add New Folder", open=False):
                            folder_input = gr.Textbox(
                                label="Folder path to add & index",
                                placeholder="/path/to/documents",
                            )
                            add_folder_btn = gr.Button("➕ Add & Index Folder", variant="primary")
                            folder_status = gr.Label(label="Folder Status")

                            with gr.Accordion("Indexing parameters (advanced)", open=False):
                                chunk_size_slider = gr.Slider(128, 2048, value=600, step=64, label="Chunk size")
                                chunk_overlap_slider = gr.Slider(0, 512, value=20, step=10, label="Chunk overlap")



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
                                info="Select which tool groups Sidekick is allowed to use.",
                            )

                

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
