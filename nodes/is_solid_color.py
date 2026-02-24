import torch

class IsSolidColorImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "check_solid"
    CATEGORY = "logic/image"

    def check_solid(self, image, threshold):
        # 计算画面中最亮和最暗像素的极差
        color_diff = torch.max(image) - torch.min(image)
        
        # 判断：如果色差小于等于你设定的阈值，则输出 True（纯色）
        is_solid = bool(color_diff <= threshold)
        
        # print(f"👉 [Solid Color Check] 极差: {color_diff:.4f}, 阈值: {threshold}, 判定为纯色: {is_solid}")
        
        return (is_solid,)

NODE_CLASS_MAPPINGS = {
    "IsSolidColorImage": IsSolidColorImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IsSolidColorImage": "Is Solid Color"
}