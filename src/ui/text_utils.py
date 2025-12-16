import re
from typing import Optional

_LATEX_BRACKET_BLOCK = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)   # \[ ... \]
_LATEX_PAREN_INLINE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)    # \( ... \)
_LATEX_SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def clean_latex_to_double_dollars(text: str) -> str:
    """
    Normalize LaTeX delimiters to $$...$$

    - Converts \\[...\\] to $$...$$
    - Converts \\(...\\) to $$...$$
    - Converts $...$ to $$...$$ only when it looks like math, to avoid breaking currency like "$10".
    """
    if not text:
        return text

    text = _LATEX_BRACKET_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    text = _LATEX_PAREN_INLINE.sub(lambda m: f"$${m.group(1).strip()}$$", text)

    def _maybe_upgrade_single_dollar(m: re.Match) -> str:
        inner = (m.group(1) or "").strip()
        looks_like_math = (
            "\\" in inner
            or any(sym in inner for sym in ["=", "+", "-", "*", "/", "^", "_", "{", "}", "\\frac", "\\sum", "\\int"])
        )
        return f"$${inner}$$" if looks_like_math else m.group(0)

    return _LATEX_SINGLE_DOLLAR.sub(_maybe_upgrade_single_dollar, text)

def format_active_folder_label(folder: Optional[str]) -> str:
    """UI helper to format the 'active folder' label."""
    if folder:
        return f"**Active folder for chat:** `{folder}`"
    return "**Active folder for chat:** _none selected_"