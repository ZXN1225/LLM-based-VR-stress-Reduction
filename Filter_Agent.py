import os
import torch
import numpy as np
import cv2
from Extraction_Database import get_lighting_stats
from PIL import Image
from diffusers import AutoencoderKL
import pickle

class FilterAgent:
    def __init__(self, hf_token, device = "cuda"):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        self.vqgan = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            use_auth_token=hf_token,
            torch_dtype=self.dtype
        ).to(self.device)
        self.vqgan.eval()

        mu_np = np.load("models/real_pano_mu.npy")
        inv_cov_np = np.load("models/real_pano_inv_cov.npy")

        self.real_pano_mu = torch.from_numpy(mu_np).to(self.device).float()
        self.real_pano_inv_cov = torch.from_numpy(inv_cov_np).to(self.device).float()

        with open("models/pca.pkl", "rb") as f:
            pca_obj = pickle.load(f)
            self.pca_components = torch.from_numpy(pca_obj.components_).to(self.device).float()
            self.pca_mean = torch.from_numpy(pca_obj.mean_).to(self.device).float()

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def _calculate_ds_score(self, image):
        """ Calculate the discontinuity score at the seam (DS Score) """
        img = np.array(image).astype(np.float32)  # RGB
        h, w, c = img.shape

        edge_w = 5

        left = img[:, :edge_w, :]
        right = img[:, -edge_w:, :]

        color_diff = left - right
        rmse = np.sqrt(np.mean(color_diff ** 2))

        rmse_norm = rmse / 255.0

        color_score = 100 * np.exp(-rmse_norm * 5)  # ⭐ 更平滑

        seam_patch = np.concatenate([right, left], axis=1)

        gray = cv2.cvtColor(seam_patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        left_grad = grad_mag[:, :edge_w]
        right_grad = grad_mag[:, edge_w:]

        grad_diff = left_grad - right_grad
        grad_rmse = np.sqrt(np.mean(grad_diff ** 2))

        grad_norm = grad_rmse / 255.0

        structure_score = 100 * np.exp(-grad_norm * 5)
        final_score = 0.6 * color_score + 0.4 * structure_score

        return float(np.clip(final_score, 0, 100))

    def _calculate_md(self, image):
        """Mahalanobis distance score"""

        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        img_tensor = (img_tensor / 127.5) - 1.0

        with torch.no_grad():
            latent = self.vqgan.encode(img_tensor).latent_dist.mode()

            z_flat = latent.reshape(1, -1).float()
            z_pca = torch.mm(z_flat - self.pca_mean, self.pca_components.t())

            diff = z_pca - self.real_pano_mu
            dist_sq = torch.mm(torch.mm(diff, self.real_pano_inv_cov), diff.t())
            dist = torch.sqrt(torch.clamp(dist_sq, min=0)).item()

            tau = 10.0
            score = 100 * np.exp(-dist / tau)

            return score

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
            md_score = self._calculate_md(img_low)

            # Using your provided local extraction script
            bright, contrast, greenery, sky, estimated_k, complexity, fd = get_lighting_stats(image_path)

            return {
                "estimated_kelvin": float(estimated_k),
                "ds_score": float(ds_score),
                "md_score": float(md_score),
                "brightness": float(bright),
                "contrast": float(contrast),
                "greenery_ratio": float(greenery),
                "sky_ratio": float(sky),
                "complexity": float(complexity),
                "fractal_dimension": float(fd)
            }
        except Exception as e:
            print(f"❌ Physical Extraction Failed: {e}")
            return None
