# -*- coding: utf-8 -*-
"""
Prompt preset loader:
- Built-in presets live in this plugin and are versioned with GitHub updates.
- User presets live in a local file ignored by git and survive plugin updates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
USER_DATA_DIR = PLUGIN_ROOT / "user_data"
USER_PROMPTS_FILE = USER_DATA_DIR / "user_prompts.py"
DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parent / "default_prompts.py"


def _load_default_prompt_presets() -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("jt_tools_default_prompts", str(DEFAULT_PROMPTS_FILE))
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[ComfyUI-JT_Tools] Failed to load default prompts: {exc}")
        return {}
    defaults = getattr(module, "DEFAULT_PROMPT_PRESETS", {})

    cleaned: Dict[str, str] = {}

    # Format A: dict[str, str]
    if isinstance(defaults, dict):
        for title, prompt_text in defaults.items():
            if not isinstance(title, str) or not isinstance(prompt_text, str):
                continue
            title_clean = title.strip()
            prompt_clean = prompt_text.strip()
            if title_clean and prompt_clean:
                cleaned[title_clean] = prompt_clean
        return cleaned

    # Format B: list[{"title": "...", "prompt": "..."}]
    if isinstance(defaults, list):
        for item in defaults:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            prompt_text = item.get("prompt")
            if not isinstance(title, str) or not isinstance(prompt_text, str):
                continue
            title_clean = title.strip()
            prompt_clean = prompt_text.strip()
            if title_clean and prompt_clean:
                cleaned[title_clean] = prompt_clean
        return cleaned

    print("[ComfyUI-JT_Tools] DEFAULT_PROMPT_PRESETS format invalid; expected dict or list.")
    return cleaned


def ensure_user_prompts_file() -> Path:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_PROMPTS_FILE.exists():
        USER_PROMPTS_FILE.write_text("# -*- coding: utf-8 -*-\n\nUSER_PROMPTS = []\n", encoding="utf-8")
    return USER_PROMPTS_FILE


def load_user_prompt_presets() -> Dict[str, str]:
    prompt_file = ensure_user_prompts_file()
    spec = importlib.util.spec_from_file_location("comfyui_jt_tools_user_prompts", str(prompt_file))
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[ComfyUI-JT_Tools] Failed to load user prompts: {exc}")
        return {}

    user_prompts = getattr(module, "USER_PROMPTS", {})

    cleaned: Dict[str, str] = {}

    # Format A: dict[str, str]
    if isinstance(user_prompts, dict):
        for title, prompt_text in user_prompts.items():
            if not isinstance(title, str) or not isinstance(prompt_text, str):
                continue
            title_clean = title.strip()
            prompt_clean = prompt_text.strip()
            if title_clean and prompt_clean:
                cleaned[title_clean] = prompt_clean
        return cleaned

    # Format B: list[{"title": "...", "prompt": "..."}]
    if isinstance(user_prompts, list):
        for item in user_prompts:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            prompt_text = item.get("prompt")
            if not isinstance(title, str) or not isinstance(prompt_text, str):
                continue
            title_clean = title.strip()
            prompt_clean = prompt_text.strip()
            if title_clean and prompt_clean:
                cleaned[title_clean] = prompt_clean
        return cleaned

    print("[ComfyUI-JT_Tools] USER_PROMPTS format invalid; expected dict or list.")
    return cleaned


def load_all_prompt_presets() -> Dict[str, str]:
    merged = dict(_load_default_prompt_presets())
    merged.update(load_user_prompt_presets())
    return merged
