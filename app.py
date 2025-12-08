from src.ui.ui_core import create_ui
from dotenv import load_dotenv
from src.db.db import init_db


load_dotenv()
init_db()


if __name__ == "__main__":
    demo = create_ui(css_path="static/style.css")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )