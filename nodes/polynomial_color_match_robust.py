import torch
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


class PolynomialColorMatch:
    RESIZE_MODE_TO_CV2 = {
        "nearest-exact": "INTER_NEAREST_EXACT",
        "nearest": "INTER_NEAREST",
        "bilinear": "INTER_LINEAR",
        "area": "INTER_AREA",
        "bicubic": "INTER_CUBIC",
        "lanczos": "INTER_LANCZOS4",
    }

    WHITE_BYPASS_RATIO = 0.9999
    MAX_SAMPLES = 50000
    MIN_VALID_PIXELS = 10

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ref_resize_method": (
                    ["nearest-exact", "nearest", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "nearest-exact"},
                ),
                "order": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
                "blur_size": ("INT", {"default": 5, "min": 0, "max": 31, "step": 1}),
            },
            "optional": {
                "target_exclude_mask": ("MASK",),
                "ref_exclude_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "JT Tools/Tools"

    def _resolve_resize_interpolation(self, resize_method):
        cv2_name = self.RESIZE_MODE_TO_CV2.get(resize_method, "INTER_NEAREST")
        if cv2_name == "INTER_NEAREST_EXACT":
            return getattr(cv2, cv2_name, cv2.INTER_NEAREST)
        return getattr(cv2, cv2_name, cv2.INTER_NEAREST)

    def _resolve_warp_interpolation(self, resize_interpolation):
        # OpenCV warpAffine/remap does not accept INTER_NEAREST_EXACT.
        if resize_interpolation == getattr(cv2, "INTER_NEAREST_EXACT", -1):
            return cv2.INTER_NEAREST
        return resize_interpolation

    def _resize_image_if_needed(self, image_np, target_w, target_h, interpolation):
        h, w = image_np.shape[:2]
        if h == target_h and w == target_w:
            return image_np
        return cv2.resize(image_np, (target_w, target_h), interpolation=interpolation)

    def _resize_mask_if_needed(self, mask_np, target_w, target_h):
        h, w = mask_np.shape[:2]
        if h == target_h and w == target_w:
            return mask_np
        return cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def _get_ref_batch_index(self, batch_index, ref_batch_size):
        return batch_index if batch_index < ref_batch_size else 0

    def _get_mask_batch(self, mask_tensor, batch_index):
        if mask_tensor is None:
            return None
        if len(mask_tensor.shape) == 3:
            use_index = batch_index if batch_index < mask_tensor.shape[0] else 0
            return mask_tensor[use_index]
        return mask_tensor

    def _mask_to_valid_indices(self, mask_np):
        mask_np = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
        white_ratio = np.mean(mask_np >= 0.999)
        if white_ratio >= self.WHITE_BYPASS_RATIO:
            return None
        return mask_np <= 0.5

    def _build_valid_mask(self, target_h, target_w, target_mask_np, ref_mask_np):
        target_valid = np.ones((target_h, target_w), dtype=bool)
        ref_valid = np.ones((target_h, target_w), dtype=bool)

        if target_mask_np is not None:
            converted = self._mask_to_valid_indices(target_mask_np)
            if converted is not None:
                target_valid = converted

        if ref_mask_np is not None:
            converted = self._mask_to_valid_indices(ref_mask_np)
            if converted is not None:
                ref_valid = converted

        return target_valid & ref_valid

    def _estimate_shift(self, target_gray, ref_gray):
        try:
            shift, response = cv2.phaseCorrelate(target_gray.astype(np.float32), ref_gray.astype(np.float32))
            if not np.isfinite(response) or response <= 1e-6:
                return (0.0, 0.0)
            return (float(shift[0]), float(shift[1]))
        except Exception:
            return (0.0, 0.0)

    def _warp_image(self, image_np, dx, dy, interpolation):
        h, w = image_np.shape[:2]
        matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        warp_interpolation = self._resolve_warp_interpolation(interpolation)
        return cv2.warpAffine(
            image_np,
            matrix,
            (w, h),
            flags=warp_interpolation,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _warp_mask(self, mask_np, dx, dy):
        h, w = mask_np.shape[:2]
        matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        warped = cv2.warpAffine(
            mask_np.astype(np.float32),
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1.0,
        )
        return np.clip(warped, 0.0, 1.0)

    def _score_shift(self, target_gray, ref_gray, valid_mask):
        if not np.any(valid_mask):
            return np.inf
        diff = np.abs(target_gray[valid_mask] - ref_gray[valid_mask])
        if diff.size == 0:
            return np.inf
        return float(np.mean(diff))

    def _align_ref_to_target(self, target_np, ref_np, joint_valid, interpolation):
        target_gray = cv2.cvtColor(target_np.astype(np.float32), cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_np.astype(np.float32), cv2.COLOR_RGB2GRAY)

        raw_dx, raw_dy = self._estimate_shift(target_gray, ref_gray)
        candidates = [(0.0, 0.0), (raw_dx, raw_dy), (-raw_dx, -raw_dy)]
        best_ref = ref_np
        best_shift = (0.0, 0.0)
        best_score = self._score_shift(target_gray, ref_gray, joint_valid)

        for dx, dy in candidates[1:]:
            warped_ref = self._warp_image(ref_np, dx, dy, interpolation)
            warped_ref_gray = cv2.cvtColor(warped_ref.astype(np.float32), cv2.COLOR_RGB2GRAY)
            score = self._score_shift(target_gray, warped_ref_gray, joint_valid)
            if score < best_score:
                best_ref = warped_ref
                best_shift = (dx, dy)
                best_score = score

        return best_ref, best_shift

    def get_polynomial_features(self, img_flat, order):
        r = img_flat[:, 0]
        g = img_flat[:, 1]
        b = img_flat[:, 2]

        features = [r, g, b, np.ones_like(r)]
        if order >= 2:
            features.extend([r**2, g**2, b**2, r * g, r * b, g * b])
        if order >= 3:
            features.extend(
                [r**3, g**3, b**3, r**2 * g, r**2 * b, g**2 * r, g**2 * b, b**2 * r, b**2 * g, r * g * b]
            )
        return np.column_stack(features)

    def match_color(
        self,
        image_target,
        image_ref,
        strength=1.0,
        order=2,
        blur_size=5,
        ref_resize_method="nearest-exact",
        target_exclude_mask=None,
        ref_exclude_mask=None,
    ):
        if cv2 is None:
            print("Warning: OpenCV (cv2) not found, return original target image.")
            return (image_target,)

        batch_size, target_h, target_w, _ = image_target.shape
        ref_batch_size, _, _, _ = image_ref.shape
        interpolation = self._resolve_resize_interpolation(ref_resize_method)
        result_images = []

        for b in range(batch_size):
            target_np = image_target[b].cpu().numpy().astype(np.float32)
            ref_idx = self._get_ref_batch_index(b, ref_batch_size)
            ref_np = image_ref[ref_idx].cpu().numpy().astype(np.float32)
            ref_np = self._resize_image_if_needed(ref_np, target_w, target_h, interpolation)

            target_mask_t = self._get_mask_batch(target_exclude_mask, b)
            ref_mask_t = self._get_mask_batch(ref_exclude_mask, b)
            target_mask_np = None
            ref_mask_np = None

            if target_mask_t is not None:
                target_mask_np = self._resize_mask_if_needed(target_mask_t.cpu().numpy(), target_w, target_h)
            if ref_mask_t is not None:
                ref_mask_np = self._resize_mask_if_needed(ref_mask_t.cpu().numpy(), target_w, target_h)

            joint_valid = self._build_valid_mask(target_h, target_w, target_mask_np, ref_mask_np)
            if np.sum(joint_valid) < self.MIN_VALID_PIXELS:
                print("Warning: Too few valid pixels after mask filtering, skip color match.")
                result_images.append(image_target[b].unsqueeze(0))
                continue

            aligned_ref, shift = self._align_ref_to_target(target_np, ref_np, joint_valid, interpolation)

            if ref_mask_np is not None:
                # Keep target/ref mask correspondence after ref translation.
                ref_mask_np = self._warp_mask(ref_mask_np, shift[0], shift[1])
                joint_valid = self._build_valid_mask(target_h, target_w, target_mask_np, ref_mask_np)
                if np.sum(joint_valid) < self.MIN_VALID_PIXELS:
                    print("Warning: Too few valid pixels after structural alignment, skip color match.")
                    result_images.append(image_target[b].unsqueeze(0))
                    continue

            kernel = max(0, int(blur_size))
            if kernel % 2 == 0 and kernel > 0:
                kernel += 1
            if kernel > 0:
                target_for_fit = cv2.GaussianBlur(target_np, (kernel, kernel), 0)
                ref_for_fit = cv2.GaussianBlur(aligned_ref, (kernel, kernel), 0)
            else:
                target_for_fit = target_np
                ref_for_fit = aligned_ref

            target_flat = target_for_fit.reshape(-1, 3)
            ref_flat = ref_for_fit.reshape(-1, 3)
            valid_flat = joint_valid.flatten()
            target_pixels = target_flat[valid_flat]
            ref_pixels = ref_flat[valid_flat]

            if target_pixels.shape[0] < self.MIN_VALID_PIXELS:
                print("Warning: Not enough samples for polynomial fit, skip color match.")
                result_images.append(image_target[b].unsqueeze(0))
                continue

            if target_pixels.shape[0] > self.MAX_SAMPLES:
                indices = np.random.choice(target_pixels.shape[0], self.MAX_SAMPLES, replace=False)
                target_pixels = target_pixels[indices]
                ref_pixels = ref_pixels[indices]

            x = self.get_polynomial_features(target_pixels, order).astype(np.float32)
            y = ref_pixels.astype(np.float32)

            try:
                w, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            except Exception:
                print("Warning: Polynomial solve failed, skip color match.")
                result_images.append(image_target[b].unsqueeze(0))
                continue

            if not np.all(np.isfinite(w)):
                print("Warning: Non-finite polynomial coefficients, skip color match.")
                result_images.append(image_target[b].unsqueeze(0))
                continue

            full_target_flat = target_np.reshape(-1, 3)
            x_full = self.get_polynomial_features(full_target_flat, order).astype(np.float32)
            corrected_flat = np.dot(x_full, w)

            if not np.all(np.isfinite(corrected_flat)):
                print("Warning: Non-finite corrected pixels, skip color match.")
                result_images.append(image_target[b].unsqueeze(0))
                continue

            corrected = corrected_flat.reshape(target_h, target_w, 3).astype(np.float32)
            final = (1.0 - float(strength)) * target_np + float(strength) * corrected
            final = np.clip(final, 0.0, 1.0)

            result_images.append(
                torch.from_numpy(final).to(dtype=image_target.dtype, device=image_target.device).unsqueeze(0)
            )

        return (torch.cat(result_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "PolynomialColorMatch": PolynomialColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PolynomialColorMatch": "Polynomial color match",
}
