import json
import os
import base64
import shutil
import io
import torch
import numpy as np
import cv2
import time
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
            self.real_pano_mu = torch.zeros((1, 4, 64, 128)).to(self.device, dtype=self.dtype)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def _get_current_kelvin(self, image):
        """ Extraction of light source (3% high-brightness sampling) """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2Lab)
        l_channel = lab[:, :, 0]

        threshold = np.percentile(l_channel, 97)
        mask = l_channel >= threshold
        bright_pixels = img_cv[mask]

        if len(bright_pixels) > 0:
            avg_bgr = np.mean(bright_pixels, axis=0)
            return self._estimate_kelvin_logic(avg_bgr[2], avg_bgr[1], avg_bgr[0])
        return 6500

    def _estimate_kelvin_logic(self, r, g, b):
        """ McCamy formula """
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        X = 0.4124 * r_n + 0.3576 * g_n + 0.1804 * b_n
        Y = 0.2126 * r_n + 0.7152 * g_n + 0.0722 * b_n
        Z = 0.0193 * r_n + 0.1192 * g_n + 0.9505 * b_n
        x = X / (X + Y + Z + 1e-6)
        y = Y / (X + Y + Z + 1e-6)
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * (n ** 3) + 3525 * (n ** 2) + 6823.3 * n + 5524.33
        return float(np.clip(cct, 1500, 12000))

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
            dist = torch.norm(latent_gen - self.real_pano_mu).item()
            max_dist_threshold = 200.0
            score = max(0, 100 * (1 - dist / max_dist_threshold))
            return float(score)

    def evaluate_scenes(self, user_input, generated_scenes):
        """
        1. Multidimensional indicator calculation
        2. Weighted scoring matrix summary
        3. Sorting and filtering Top 10
        """
        scored_scenes = []
        final_dir = "static/final_images"
        os.makedirs(final_dir, exist_ok=True)

        for scene in generated_scenes:
            print(f"🧐 Multidimensional evaluation scenario {scene['step']}...")
            try:
                img = Image.open(scene['image_path']).convert("RGB")

                # DS Score (50%)
                ds_score = self._calculate_ds_score(img)

                # FAED (20%)
                faed_score = self._calculate_faed(img)

                # Kelvin Offset (10%)
                current_k = self._get_current_kelvin(img)
                target_k = scene['unity_config']['kelvin']
                k_diff = abs(current_k - target_k)
                kelvin_score = max(0, 100 - (k_diff / 10))

                # GPT-4o Vision (20%)
                gpt_report = self._ask_gpt4o_vision(img, scene['image_prompt'], user_input)

                final_score = (ds_score * 0.50 +
                               faed_score * 0.20 +
                               kelvin_score * 0.10 +
                               gpt_report['score'] * 10 * 0.20)

                scene['filter_score'] = final_score
                scene['current_kelvin'] = current_k

                scored_scenes.append(scene)


            except Exception as e:
                print(f"⚠️ Filter Error for Scene {scene['step']}: {e}")
                continue

        # Sort to top10
        sorted_scenes = sorted(scored_scenes, key=lambda x: x['filter_score'], reverse=True)[:10]

        final_output_scenes = []
        for i, scene in enumerate(sorted_scenes):
            new_step = i + 1
            old_path = scene['image_path']
            new_filename = f"scene_{new_step}.png"
            new_path = os.path.join(final_dir, new_filename)

            shutil.copy2(old_path, new_path)

            scene.update({
                "step": new_step,
                "image_url": f"http://localhost:8000/{new_path.replace(os.sep, '/')}",
                "image_path": new_path
            })
            final_output_scenes.append(scene)

        print(f"✅ Evaluation complete. The top 10 treatment scenarios have been saved to {final_dir}")
        return final_output_scenes

    def _ask_gpt4o_vision(self, image, target_prompt, user_input):
        """ GPT-4o Vision """

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

        max_retries = 5
        retry_delay = 2

        system_prompt = f"""
                You are a professional VR Therapy Quality Inspector. 
                User Input: {user_input}

                Evaluate the image based on:
                1. Semantic Alignment: Does it match the healing prompt?
                2. Technical Quality: Are there 360-degree distortions (broken seams, tiling.etc.)? Are these images seamless?
                3. Therapeutic Value: Is the lighting (Kelvin) soothing? Avoid visual chaos.

                Return JSON ONLY:
                {{ "score": 0-10 }}
                """

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{system_prompt}\nTarget Prompt: {target_prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }],
                    response_format={"type": "json_object"},
                    timeout=30
                )
                content = response.choices[0].message.content
                if content:
                    return json.loads(content)
                else:
                    raise ValueError("Empty response from GPT")

            except Exception as e:
                print(f"⚠️ GPT Vision Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print("❌ All 5 GPT retries failed. Using fallback score.")

        return {"score": 5}