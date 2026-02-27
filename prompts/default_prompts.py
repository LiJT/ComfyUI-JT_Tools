# -*- coding: utf-8 -*-
"""
Built-in prompt presets shipped with this plugin.
Prompt bodies are loaded from prompt_enhancer_preset.py without modification.
"""

import importlib.util
from pathlib import Path


def _load_prompt_enhancer_module():
    plugin_root = Path(__file__).resolve().parents[1]
    source_file = plugin_root / "prompts" / "prompt_enhancer_preset.py"
    spec = importlib.util.spec_from_file_location("jt_tools_prompt_enhancer_source", str(source_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("[ComfyUI-JT_Tools] Cannot load prompt_enhancer_preset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SRC = _load_prompt_enhancer_module()

DEFAULT_PROMPT_PRESETS = [
    # 默认示例
    {
        "title": "默认AI图像提示词",
        "prompt": "你是一位专业的AI图像生成提示词工程师。请详细描述这张图像主体、前景、中景、背景、构图、视觉引导、光影氛围等细节并创作出具有深度、氛围和艺术感或日常业余设备拍摄的图像提示词。要求：中文提示词，不用出现对图像水印的描述，不要出现无关的文字和符号，不需要总结，限制在500字以内：",
    },
    # Qwen Image（基础）
    {
        "title": "Qwen Image 提示词优化",
        "prompt": _SRC.QWEN_IMAGE_ZH,
    },
    {
        "title": "Qwen Image Prompt Optimizer (EN)",
        "prompt": _SRC.QWEN_IMAGE_EN,
    },
    # Qwen Image（2512）
    {
        "title": "Qwen Image 2512 提示词优化",
        "prompt": _SRC.QWEN_IMAGE_2512_ZH,
    },
    {
        "title": "Qwen Image 2512 Prompt Optimizer (EN)",
        "prompt": _SRC.QWEN_IMAGE_2512_EN,
    },
    # Qwen Image Edit
    {
        "title": "Qwen Image Edit 提示词优化",
        "prompt": _SRC.QWEN_IMAGE_EDIT_ZH,
    },
    {
        "title": "Qwen Image Edit Prompt Optimizer (EN)",
        "prompt": _SRC.QWEN_IMAGE_EDIT_EN,
    },
    {
        "title": "Qwen Image Edit 2509 提示词优化",
        "prompt": _SRC.QWEN_IMAGE_EDIT_2509,
    },
    {
        "title": "Qwen Image Edit 2511 提示词优化",
        "prompt": _SRC.QWEN_IMAGE_EDIT_2511,
    },
    # Z-Image
    {
        "title": "Z-Image Turbo 提示词优化",
        "prompt": _SRC.ZIMAGE_TURBO,
    },
    # FLUX.2
    {
        "title": "FLUX.2 T2I Prompt Optimizer (EN)",
        "prompt": _SRC.FLUX2_T2I,
    },
    {
        "title": "FLUX.2 I2I Prompt Optimizer (EN)",
        "prompt": _SRC.FLUX2_I2I,
    },
    # FLUX.2 Klein
    {
        "title": "FLUX.2 Klein T2I Prompt Optimizer (EN)",
        "prompt": _SRC.FLUX2_KLEIN_T2I_EN,
    },
    {
        "title": "FLUX.2 Klein Edit Prompt Optimizer (EN)",
        "prompt": _SRC.FLUX2_KLEIN_EDIT_EN,
    },
    # WAN T2V
    {
        "title": "WAN T2V 提示词优化",
        "prompt": _SRC.WAN_T2V_ZH,
    },
    {
        "title": "WAN T2V Prompt Optimizer (EN)",
        "prompt": _SRC.WAN_T2V_EN,
    },
    # WAN I2V（有输入提示词）
    {
        "title": "WAN I2V 提示词优化",
        "prompt": _SRC.WAN_I2V_ZH,
    },
    {
        "title": "WAN I2V Prompt Optimizer (EN)",
        "prompt": _SRC.WAN_I2V_EN,
    },
    # WAN I2V（空提示词补全）
    {
        "title": "WAN I2V 空提示词补全",
        "prompt": _SRC.WAN_I2V_EMPTY_ZH,
    },
    {
        "title": "WAN I2V Empty Prompt Filler (EN)",
        "prompt": _SRC.WAN_I2V_EMPTY_EN,
    },
    # WAN FLF2V
    {
        "title": "WAN FLF2V 提示词优化",
        "prompt": _SRC.WAN_FLF2V_ZH,
    },
    {
        "title": "WAN FLF2V Prompt Optimizer (EN)",
        "prompt": _SRC.WAN_FLF2V_EN,
    },
    # 提示词扩写
    {
        "title": "提示词扩写-通用",
        "prompt": _SRC.PROMPT_ENHANCE_GENERAL_ZH,
    },
    {
        "title": "提示词扩写-人像大师",
        "prompt": _SRC.PROMPT_ENHANCE_PORTRIT_ZH,
    },
    {
        "title": "提示词扩写-Tags风格",
        "prompt": _SRC.PROMPT_ENHANCE_SDXLTAGS_ZH,
    },
    {
        "title": "提示词扩写-Kontext指令优化翻译",
        "prompt": _SRC.PROMPT_ENHANCE_FLUXKONTEXT_EN,
    },
    # 提示词反推    
    {
        "title": "提示词反推-中文",
        "prompt": _SRC.GETPROMPT_DETAIL_CN,
    },
    {
        "title": "提示词反推-英文",
        "prompt": _SRC.GETPROMPT_DETAIL_EN,
    },
    {
        "title": "提示词反推-Tags中文",
        "prompt": _SRC.GETPROMPT_TAGS_CN,
    },
    {
        "title": "提示词反推-Tags英文",
        "prompt": _SRC.GETPROMPT_TAGS_EN,
    },
]
