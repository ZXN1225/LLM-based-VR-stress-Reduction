import os
import torch
import numpy as np
import cv2
from Extraction_Database import get_lighting_stats
from openai import OpenAI
from PIL import Image
from diffusers import AutoencoderKL

class FilterAgent:
    def __init__(self, api_key, hf_token, device = "cuda"):
        self.client = OpenAI(api_key=api_key)
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        self.vqgan = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            use_auth_token=hf_token,
            torch_dtype=self.dtype
        ).to(self.device)
        self.vqgan.eval()

        mu_path = "models/real_pano_mu.pt"
        if os.path.exists(mu_path):
            self.real_pano_mu = torch.load(mu_path).to(self.device, dtype=self.dtype)
        else:
            print("⚠️ Warning: real_pano_mu.pt not found, using zeros.")
            self.real_pano_mu = torch.zeros((1, 4, 64, 128)).to(self.device, dtype=self.dtype)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def _calculate_ds_score(self, image):
        """ Calculate the discontinuity score at the seam (DS Score) """
        img_np = np.array(image.convert("L")).astype(np.float32)
        h, w = img_np.shape

        # RMSE
        left_col = img_np[:, 0]
        right_col = img_np[:, -1]
        rmse = np.sqrt(np.mean((left_col - right_col) ** 2))

        color_score = max(0, 100 - (rmse * 4))

        edge_w = 5
        seam_patch = np.hstack([img_np[:, -edge_w:], img_np[:, :edge_w]])

        grad_x = cv2.Sobel(seam_patch, cv2.CV_64F, 1, 0, ksize=3)

        center_grad_variance = np.var(grad_x[:, edge_w - 1:edge_w + 1])

        structure_score = max(0, 100 - (center_grad_variance * 0.5))
        final_ds = (color_score * 0.7) + (structure_score * 0.3)

        return float(np.clip(final_ds, 0, 100))

    def _calculate_faed(self, image):
        """ Panoramic geometric feature scoring based on VQGAN """
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device,
                                                                                        dtype=self.dtype) / 255.0
        with torch.no_grad():
            latent_gen = self.vqgan.encode(img_tensor).latent_dist.mode()
            diff = latent_gen - self.real_pano_mu
            dist = torch.sqrt(torch.mean(diff ** 2)).item()
            max_dist_threshold = 2.0
            score = max(0, 100 * (1 - dist / max_dist_threshold))
            return float(score)

    def get_physical_report(self, image_path):
        """
        Main Entrance: Generates a comprehensive physical report of the image.
        Returns a dictionary for the TherapistAgent to review.
        """
        try:
            ori_img = Image.open(image_path).convert("RGB")
            # Downsample for faster CV processing
            img_low = ori_img.resize((1024, 512), Image.Resampling.LANCZOS)

            # Extract Metrics
            ds_score = self._calculate_ds_score(img_low)
            faed_score = self._calculate_faed(img_low)

            # Using your provided local extraction script
            bright, contrast, sharp, color, estimated_k, complexity = get_lighting_stats(image_path)

            return {
                "estimated_kelvin": float(estimated_k),
                "ds_score": float(ds_score),
                "faed_score": float(faed_score),
                "brightness": float(bright),
                "contrast": float(contrast),
                "sharpness": float(sharp),
                "complexity": float(complexity)
            }
        except Exception as e:
            print(f"❌ Physical Extraction Failed: {e}")
            return None
