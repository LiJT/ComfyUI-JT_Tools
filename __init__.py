"""
ComfyUI-JT_Tools entrypoint.
Keep plugin registration in one place to minimize extra files.
"""

import importlib.util
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_ROOT_DIR = Path(__file__).resolve().parent
_NODES_DIR = _ROOT_DIR / "nodes"

for node_file in _NODES_DIR.glob("*.py"):
    module_name = node_file.stem
    if module_name.startswith("_"):
        continue

    spec = importlib.util.spec_from_file_location(f"jt_tools_node_{module_name}", str(node_file))
    if spec is None or spec.loader is None:
        continue

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
