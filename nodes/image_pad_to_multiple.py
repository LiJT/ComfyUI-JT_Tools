import math

import torch
import torch.nn.functional as F


class ImagePadToMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 256}),
                "pad_color_rgb": ("STRING", {"default": "0, 0, 0"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "pad_to_multiple"
    CATEGORY = "JT Tools/Tools"

    @staticmethod
    def _parse_rgb_string(rgb_string):
        try:
            parts = [p.strip() for p in str(rgb_string).split(",")]
            if len(parts) != 3:
                raise ValueError("RGB value must contain 3 channels.")
            rgb = [int(p) for p in parts]
            if any(c < 0 or c > 255 for c in rgb):
                raise ValueError("RGB channel must be in [0, 255].")
            return [c / 255.0 for c in rgb]
        except Exception:
            print("Warning: pad_color_rgb 格式无效，已回退为黑色 (0, 0, 0)。")
            return [0.0, 0.0, 0.0]

    @staticmethod
    def _normalize_mask(mask, batch, height, width, device, dtype):
        if mask is None:
            return torch.zeros((batch, height, width), dtype=dtype, device=device)

        mask = mask.to(device=device, dtype=dtype)

        # 兼容常见输入形态: [H,W] / [B,H,W] / [B,H,W,1]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask[..., 0]
        elif mask.ndim != 3:
            raise ValueError("mask 维度无效，期望 [H,W]、[B,H,W] 或 [B,H,W,1]。")

        # 批次对齐：1 张 mask 复用到整批；其他情况重复后裁切
        if mask.shape[0] != batch:
            if mask.shape[0] == 1:
                mask = mask.repeat(batch, 1, 1)
            else:
                repeat_count = (batch + mask.shape[0] - 1) // mask.shape[0]
                mask = mask.repeat(repeat_count, 1, 1)[:batch]

        # 空间尺寸对齐到 image
        if mask.shape[1] != height or mask.shape[2] != width:
            mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)

        return mask.clamp(0.0, 1.0)

    def pad_to_multiple(self, image, multiple_of, pad_color_rgb, mask=None):
        batch, height, width, channels = image.shape
        new_height = math.ceil(height / multiple_of) * multiple_of
        new_width = math.ceil(width / multiple_of) * multiple_of

        pad_bottom = new_height - height
        pad_right = new_width - width
        base_mask = self._normalize_mask(mask, batch, height, width, image.device, image.dtype)

        if pad_bottom == 0 and pad_right == 0:
            return (image, base_mask, int(new_width), int(new_height))

        r, g, b = self._parse_rgb_string(pad_color_rgb)
        pad_color = torch.tensor([r, g, b], dtype=image.dtype, device=image.device)

        if channels != 3:
            # IMAGE 正常是 3 通道；异常情况下用首通道值填充其余通道。
            pad_color = pad_color[:1].repeat(channels)

        padded = torch.empty(
            (batch, new_height, new_width, channels),
            dtype=image.dtype,
            device=image.device,
        )
        padded[:] = pad_color
        padded[:, :height, :width, :] = image

        # 输出 mask = 输入 mask 与新扩展区域的并集
        output_mask = torch.zeros((batch, new_height, new_width), dtype=image.dtype, device=image.device)
        output_mask[:, :height, :width] = base_mask
        if pad_bottom > 0:
            output_mask[:, height:, :] = 1.0
        if pad_right > 0:
            output_mask[:, :height, width:] = 1.0

        return (padded, output_mask, int(new_width), int(new_height))


NODE_CLASS_MAPPINGS = {
    "ImagePadToMultiple": ImagePadToMultiple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePadToMultiple": "Image Pad To Multiple",
}
