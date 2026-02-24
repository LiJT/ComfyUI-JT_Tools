# -*- coding: utf-8 -*-
"""
ComfyUI node: choose a preset title and output prompt text.
"""

import importlib.util
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_STORE_FILE = _PLUGIN_ROOT / "prompts" / "prompt_store.py"
_PROMPT_STORE_SPEC = importlib.util.spec_from_file_location("jt_tools_prompt_store", str(_PROMPT_STORE_FILE))
if _PROMPT_STORE_SPEC is None or _PROMPT_STORE_SPEC.loader is None:
    raise RuntimeError("[ComfyUI-JT_Tools] Cannot load prompt_store.py")
_PROMPT_STORE_MODULE = importlib.util.module_from_spec(_PROMPT_STORE_SPEC)
_PROMPT_STORE_SPEC.loader.exec_module(_PROMPT_STORE_MODULE)
load_all_prompt_presets = _PROMPT_STORE_MODULE.load_all_prompt_presets


class JTPromptPresetOutput:
    @classmethod
    def _preset_choices(cls):
        presets = load_all_prompt_presets()
        return list(presets.keys()) or ["默认AI图像提示词"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (cls._preset_choices(),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "JT Tools/Prompt"

    def get_prompt(self, preset):
        presets = load_all_prompt_presets()
        if preset in presets:
            return (presets[preset],)

        if presets:
            first_prompt = next(iter(presets.values()))
            return (first_prompt,)

        return ("",)


NODE_CLASS_MAPPINGS = {
    "JTPromptPresetOutput": JTPromptPresetOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JTPromptPresetOutput": "JT Prompt Preset",
}
