from dataclasses import dataclass
from typing import Optional

import gradio as gr

from src.ui.auth import login_user, register_user
from src.ui.text_utils import format_active_folder_label
from src.ui.ui_runtime import get_controller


@dataclass(frozen=True)
class UIRefs:
    # Auth widgets
    username_input: gr.Textbox
    password_input: gr.Textbox
    login_status: gr.Label
    login_page: gr.Column
    main_page: gr.Column
    login_btn: gr.Button
    register_btn: gr.Button
    logout_btn: gr.Button

    # State/widgets
    logged_in_user: gr.State
    chatbox: gr.Chatbot
    active_folder_label: gr.Markdown

    # Folder widgets
    folder_input: gr.Textbox
    folder_status: gr.Label
    folder_dropdown: gr.Dropdown
    add_folder_btn: gr.Button
    remove_folder_btn: gr.Button
    index_folder_btn: gr.Button
    reindex_folder_btn: gr.Button
    chunk_size_slider: gr.Slider
    chunk_overlap_slider: gr.Slider

    # Chat widgets
    message_input: gr.Textbox
    send_btn: gr.Button
    clear_btn: gr.Button
    retrieval_k_dropdown: gr.Dropdown
    tool_selector: gr.CheckboxGroup


def wire_events(refs: UIRefs, *, NO_FOLDER_LABEL: str) -> None:
    # ---------- Register ----------
    refs.register_btn.click(
        lambda u, p: register_user(u, p)[1],
        inputs=[refs.username_input, refs.password_input],
        outputs=refs.login_status,
    )

    # ---------- Login ----------
    async def handle_login(u: str, p: str):
        success, msg = login_user(u, p)
        if not success:
            return (
                msg,  # login_status
                gr.update(visible=True),  # login_page
                gr.update(visible=False),  # main_page
                gr.update(),  # folder_dropdown unchanged
                None,  # logged_in_user
                [],  # chatbox
                format_active_folder_label(None),
                gr.update(visible=False),  # logout_btn
            )

        h = await get_controller()

        folders = h.get_folders(u)
        history = await h.load_chat(u, None)

        dropdown_update = gr.update(
            choices=[NO_FOLDER_LABEL] + folders,
            value=NO_FOLDER_LABEL,
        )

        return (
            msg,  # login_status
            gr.update(visible=False),  # login_page
            gr.update(visible=True),  # main_page
            dropdown_update,  # folder_dropdown
            u,  # logged_in_user
            history,  # chatbox
            format_active_folder_label(None),
            gr.update(visible=True),  # logout_btn
        )

    refs.login_btn.click(
        handle_login,
        inputs=[refs.username_input, refs.password_input],
        outputs=[
            refs.login_status,
            refs.login_page,
            refs.main_page,
            refs.folder_dropdown,
            refs.logged_in_user,
            refs.chatbox,
            refs.active_folder_label,
            refs.logout_btn,
        ],
    )

    # ---------- Folder: Add ----------
    async def handle_add_folder(username: Optional[str], folder_path: str):
        if not username or not folder_path:
            return "Not logged in or empty folder path", gr.update()

        h = await get_controller()
        msg, folders = await h.add_folder(username, folder_path)

        return msg, gr.update(
            choices=[NO_FOLDER_LABEL] + folders,
            value=folder_path,
        )

    refs.add_folder_btn.click(
        handle_add_folder,
        inputs=[refs.logged_in_user, refs.folder_input],
        outputs=[refs.folder_status, refs.folder_dropdown],
    )

    # ---------- Folder: Remove ----------
    async def handle_remove_folder(username: Optional[str], folder: Optional[str]):
        if not username or not folder or folder == NO_FOLDER_LABEL:
            return "Select a real folder to remove", gr.update(
                choices=[NO_FOLDER_LABEL],
                value=NO_FOLDER_LABEL,
            )

        h = await get_controller()
        msg, folders = await h.remove_folder(username, folder)

        return msg, gr.update(
            choices=[NO_FOLDER_LABEL] + folders,
            value=NO_FOLDER_LABEL,
        )

    refs.remove_folder_btn.click(
        handle_remove_folder,
        inputs=[refs.logged_in_user, refs.folder_dropdown],
        outputs=[refs.folder_status, refs.folder_dropdown],
    )

    # ---------- Folder: Index ----------
    async def handle_index_folder(
        username: Optional[str],
        folder: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
    ):
        if not username or not folder or folder == NO_FOLDER_LABEL:
            return "Select a real folder to index"
        h = await get_controller()
        return await h.index_folder(username, folder, chunk_size, chunk_overlap)

    refs.index_folder_btn.click(
        handle_index_folder,
        inputs=[
            refs.logged_in_user,
            refs.folder_dropdown,
            refs.chunk_size_slider,
            refs.chunk_overlap_slider,
        ],
        outputs=refs.folder_status,
    )

    # ---------- Folder: Reindex ----------
    async def handle_reindex_folder(
        username: Optional[str],
        folder: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
    ):
        if not username or not folder or folder == NO_FOLDER_LABEL:
            return "Select a real folder to reindex"
        h = await get_controller()
        return await h.reindex_folder(username, folder, chunk_size, chunk_overlap)

    refs.reindex_folder_btn.click(
        handle_reindex_folder,
        inputs=[
            refs.logged_in_user,
            refs.folder_dropdown,
            refs.chunk_size_slider,
            refs.chunk_overlap_slider,
        ],
        outputs=refs.folder_status,
    )

    # ---------- Folder change ----------
    async def handle_folder_change(username: Optional[str], folder: Optional[str]):
        if not username:
            return [], format_active_folder_label(None)

        h = await get_controller()

        if folder == NO_FOLDER_LABEL:
            history = await h.load_chat(username, None)
            return history, format_active_folder_label(None)

        history = await h.load_chat(username, folder)
        return history, format_active_folder_label(folder)

    refs.folder_dropdown.change(
        fn=handle_folder_change,
        inputs=[refs.logged_in_user, refs.folder_dropdown],
        outputs=[refs.chatbox, refs.active_folder_label],
    )

    # ---------- Chat ----------
    async def handle_chat(
        username: Optional[str],
        folder: Optional[str],
        message: str,
        top_k: int,
        enabled_tools,
    ):
        if folder == NO_FOLDER_LABEL:
            folder = None

        if not username:
            return (
                [{"role": "assistant", "content": "Please log in before chatting."}],
                "",
            )

        h = await get_controller()
        return await h.chat(username, folder, message, top_k, enabled_tools)

    refs.send_btn.click(
        handle_chat,
        inputs=[
            refs.logged_in_user,
            refs.folder_dropdown,
            refs.message_input,
            refs.retrieval_k_dropdown,
            refs.tool_selector,
        ],
        outputs=[refs.chatbox, refs.message_input],
    )

    refs.message_input.submit(
        handle_chat,
        inputs=[
            refs.logged_in_user,
            refs.folder_dropdown,
            refs.message_input,
            refs.retrieval_k_dropdown,
            refs.tool_selector,
        ],
        outputs=[refs.chatbox, refs.message_input],
    )

    # ---------- Clear chat ----------
    async def handle_clear_chat(username: Optional[str], folder: Optional[str]):
        if folder == NO_FOLDER_LABEL:
            folder = None
        if not username:
            return []
        h = await get_controller()
        return h.clear_chat(username, folder)

    refs.clear_btn.click(
        handle_clear_chat,
        inputs=[refs.logged_in_user, refs.folder_dropdown],
        outputs=refs.chatbox,
    )

    # ---------- Logout ----------
    def handle_logout():
        return (
            "",  # username_input
            "",  # password_input
            "Logged out",  # login_status
            gr.update(visible=True),  # login_page
            gr.update(visible=False),  # main_page
            [],  # chatbox
            gr.update(
                choices=[NO_FOLDER_LABEL], value=NO_FOLDER_LABEL
            ),  # folder_dropdown
            format_active_folder_label(None),  # active_folder_label
            None,  # logged_in_user
            gr.update(visible=False),  # logout_btn
        )

    refs.logout_btn.click(
        handle_logout,
        inputs=[],
        outputs=[
            refs.username_input,
            refs.password_input,
            refs.login_status,
            refs.login_page,
            refs.main_page,
            refs.chatbox,
            refs.folder_dropdown,
            refs.active_folder_label,
            refs.logged_in_user,
            refs.logout_btn,
        ],
    )
