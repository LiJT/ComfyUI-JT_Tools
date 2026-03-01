# 利用魔法方法伪装成任意类型，以便接入任何节点
class AlwaysEqualProxy(str):
    def __eq__(self, _): return True
    def __ne__(self, _): return False

ANY_TYPE = AlwaysEqualProxy("*")

class UpstreamStateDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": (ANY_TYPE,), # 任意类型输入，且设为可选，防止上游 Mute 时本节点被阻断
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",          # 获取当前节点 ID
                "extra_pnginfo": "EXTRA_PNGINFO"   # 获取包含 UI 状态的全局工作流 JSON
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_active",)
    FUNCTION = "check_state"
    CATEGORY = "JT Tools/Logic"

    def check_state(self, any_input=None, unique_id=None, extra_pnginfo=None):
        is_active = False  # 默认无连接或上游未激活时输出 False
        
        if extra_pnginfo and unique_id:
            try:
                if isinstance(extra_pnginfo, list):
                    extra_pnginfo = extra_pnginfo[0] if extra_pnginfo else {}

                workflow = extra_pnginfo.get("workflow", {}) if isinstance(extra_pnginfo, dict) else {}
                nodes = workflow.get("nodes", [])
                links = workflow.get("links", [])
                
                # 1. 在工作流中定位当前节点
                my_node = next((n for n in nodes if str(n.get("id")) == str(unique_id)), None)
                if my_node:
                    # 2. 找到 any_input 对应的连线 ID
                    target_link_id = next((inp.get("link") for inp in my_node.get("inputs", []) if inp.get("name") == "any_input"), None)
                    
                    if target_link_id is not None:
                        # 3. 通过连线 ID 找到上游节点的 ID
                        upstream_node_id = next(
                            (link[1] for link in links if isinstance(link, (list, tuple)) and len(link) > 1 and link[0] == target_link_id),
                            None
                        )
                        
                        if upstream_node_id is not None:
                            # 4. 获取上游节点并检查其 mode
                            upstream_node = next((n for n in nodes if str(n.get("id")) == str(upstream_node_id)), None)
                            if upstream_node:
                                # ComfyUI 前端节点状态 mode 说明：
                                # 0 = Always (正常运行), 2 = Mute (静音), 4 = Bypass (绕过)
                                if upstream_node.get("mode", 0) == 0:
                                    is_active = True
            except Exception:
                pass # 出现任何解析异常时安全地 fallback 到 False

        return (is_active,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "UpstreamStateDetector": UpstreamStateDetector
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UpstreamStateDetector": "Upstream State Detector"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']