import os
import cv2
import numpy as np


class PhotoFinisher:
    """
    Fidelity-preserving VR panorama finisher.

    Goal:
      - improve perceived VR clarity without hallucinating new texture/content
      - avoid ESRGAN/AI reconstruction artifacts
      - preserve original structure, colors, and photographic realism

    Pipeline:
      1) Lanczos resize to target scale / target size
      2) optional light denoise
      3) CLAHE only on L channel, very conservative
      4) edge-aware unsharp mask
      5) optional tiny saturation restraint

    Input/output image format:
      - OpenCV BGR uint8
    """

    def __init__(
        self,
        enabled=True,
        scale=1,
        target_width=4096,
        target_height=2048,
        resize_interpolation="lanczos",
        denoise=False,
        denoise_h=1.2,
        denoise_h_color=1.2,
        clahe=True,
        clahe_clip=1.35,
        clahe_tile_grid=8,
        sharpen=True,
        sharpen_radius=1.15,
        sharpen_amount=0.40,
        sharpen_threshold=3.0,
        local_contrast=True,
        local_contrast_amount=0.025,
        saturation_scale=1.0,
        png_compression=1,
        debug=False,
    ):
        self.enabled = bool(enabled)
        self.scale = int(scale)
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.resize_interpolation = str(resize_interpolation).lower().strip()
        self.denoise = bool(denoise)
        self.denoise_h = float(denoise_h)
        self.denoise_h_color = float(denoise_h_color)
        self.clahe = bool(clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_tile_grid = int(clahe_tile_grid)
        self.sharpen = bool(sharpen)
        self.sharpen_radius = float(sharpen_radius)
        self.sharpen_amount = float(sharpen_amount)
        self.sharpen_threshold = float(sharpen_threshold)
        self.local_contrast = bool(local_contrast)
        self.local_contrast_amount = float(local_contrast_amount)
        self.saturation_scale = float(saturation_scale)
        self.png_compression = int(png_compression)
        self.debug = bool(debug)

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("PHOTO_FINISHER_ENABLED", "1") == "1",
            scale=int(os.getenv("PHOTO_FINISHER_SCALE", os.getenv("AUTO_UPSCALE_SCALE", "1"))),
            target_width=int(os.getenv("PHOTO_FINISHER_TARGET_WIDTH", "4096")),
            target_height=int(os.getenv("PHOTO_FINISHER_TARGET_HEIGHT", "2048")),
            resize_interpolation=os.getenv("PHOTO_FINISHER_RESIZE", "lanczos"),
            denoise=os.getenv("PHOTO_FINISHER_DENOISE", "0") == "1",
            denoise_h=float(os.getenv("PHOTO_FINISHER_DENOISE_H", "1.2")),
            denoise_h_color=float(os.getenv("PHOTO_FINISHER_DENOISE_H_COLOR", "1.2")),
            clahe=os.getenv("PHOTO_FINISHER_CLAHE", "1") == "1",
            clahe_clip=float(os.getenv("PHOTO_FINISHER_CLAHE_CLIP", "1.35")),
            clahe_tile_grid=int(os.getenv("PHOTO_FINISHER_CLAHE_TILE", "8")),
            sharpen=os.getenv("PHOTO_FINISHER_SHARPEN", "1") == "1",
            sharpen_radius=float(os.getenv("PHOTO_FINISHER_SHARPEN_RADIUS", "1.15")),
            sharpen_amount=float(os.getenv("PHOTO_FINISHER_SHARPEN_AMOUNT", "0.40")),
            sharpen_threshold=float(os.getenv("PHOTO_FINISHER_SHARPEN_THRESHOLD", "3")),
            local_contrast=os.getenv("PHOTO_FINISHER_LOCAL_CONTRAST", "1") == "1",
            local_contrast_amount=float(os.getenv("PHOTO_FINISHER_LOCAL_CONTRAST_AMOUNT", "0.025")),
            saturation_scale=float(os.getenv("PHOTO_FINISHER_SATURATION_SCALE", "1.0")),
            png_compression=int(os.getenv("PHOTO_FINISHER_PNG_COMPRESSION", "1")),
            debug=os.getenv("PHOTO_FINISHER_DEBUG", "0") == "1",
        )

    @staticmethod
    def _to_uint8(img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        return np.clip(img, 0, 255).astype(np.uint8)

    def _resize(self, img_bgr, scale=None):
        h, w = img_bgr.shape[:2]
        if self.target_width > 0 and self.target_height > 0:
            out_w, out_h = self.target_width, self.target_height
        else:
            s = int(scale or self.scale or 2)
            out_w, out_h = w * s, h * s

        if self.resize_interpolation in ["bicubic", "cubic"]:
            interp = cv2.INTER_CUBIC
        elif self.resize_interpolation in ["area"]:
            interp = cv2.INTER_AREA
        elif self.resize_interpolation in ["linear", "bilinear"]:
            interp = cv2.INTER_LINEAR
        else:
            interp = cv2.INTER_LANCZOS4

        if self.debug:
            print(f"📷 [FidelityFinisher] resize {w}x{h} -> {out_w}x{out_h}, interp={self.resize_interpolation}")
        return cv2.resize(img_bgr, (out_w, out_h), interpolation=interp)

    def _light_denoise(self, img_bgr):
        if not self.denoise:
            return img_bgr
        return cv2.fastNlMeansDenoisingColored(
            img_bgr,
            None,
            h=self.denoise_h,
            hColor=self.denoise_h_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _clahe_luma_only(self, img_bgr):
        if not self.clahe or self.clahe_clip <= 0:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        tile = max(2, int(self.clahe_tile_grid))
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(tile, tile))
        enhanced_lightness = clahe.apply(lightness)

        # Blend conservatively to avoid HDR / synthetic look.
        blend = 0.38
        final_lightness = cv2.addWeighted(lightness, 1.0 - blend, enhanced_lightness, blend, 0)
        return cv2.cvtColor(cv2.merge([final_lightness, a, b]), cv2.COLOR_LAB2BGR)

    def _unsharp_mask_edge_aware(self, img_bgr):
        if not self.sharpen or self.sharpen_amount <= 0:
            return img_bgr

        radius = max(0.1, float(self.sharpen_radius))
        amount = float(self.sharpen_amount)
        threshold = float(self.sharpen_threshold)

        src = img_bgr.astype(np.float32)
        blurred = cv2.GaussianBlur(src, (0, 0), radius)
        detail = src - blurred

        if threshold > 0:
            gray_detail = cv2.cvtColor(np.abs(detail).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = np.clip((gray_detail - threshold) / max(1.0, threshold * 4.0), 0.0, 1.0)
            # Avoid amplifying very flat sky/water noise.
            mask = cv2.GaussianBlur(mask, (0, 0), 0.8)[:, :, None]
            detail *= mask

        out = src + detail * amount
        return self._to_uint8(out)

    def _gentle_local_contrast(self, img_bgr):
        if not self.local_contrast or self.local_contrast_amount <= 0:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness, a, b = cv2.split(lab)
        blur = cv2.GaussianBlur(lightness, (0, 0), 8.0)
        lightness = np.clip(lightness + (lightness - blur) * self.local_contrast_amount, 0, 255)
        return cv2.cvtColor(cv2.merge([lightness, a, b]).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _saturation_control(self, img_bgr):
        if abs(self.saturation_scale - 1.0) < 1e-4:
            return img_bgr
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def enhance(self, img_bgr, scale=None):
        if img_bgr is None:
            return None
        img_bgr = self._to_uint8(img_bgr)
        if not self.enabled:
            return img_bgr

        out = self._resize(img_bgr, scale=scale)
        out = self._light_denoise(out)
        out = self._clahe_luma_only(out)
        out = self._unsharp_mask_edge_aware(out)
        out = self._gentle_local_contrast(out)
        out = self._saturation_control(out)
        return self._to_uint8(out)

    # Backward compatible with old call sites that used PhotoFinisher.apply().
    def apply(self, img_bgr, scene_metrics=None):
        return self.enhance(img_bgr, scale=1)

    def process_file(self, input_path, output_path, scale=None):
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ [FidelityFinisher] Failed to read image: {input_path}")
            return False
        out = self.enhance(img, scale=scale)
        ok = self.save(output_path, out)
        if self.debug:
            print(f"📷 [FidelityFinisher] saved={ok}: {output_path}")
        return bool(ok)

    def save(self, output_path, img_bgr):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            return cv2.imwrite(
                output_path,
                img_bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
            )
        return cv2.imwrite(output_path, img_bgr)
