import os
import gc
import time
import cv2
import numpy as np
import requests
import torch
import random
import subprocess
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler
)


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


class ProductionAgent:
    """
    Production agent upgraded with:
    1) RAG reference panorama conditioning through SDXL IP-Adapter.
    2) SUPIR-based final enhancement instead of Real-ESRGAN.

    Expected env variables for SUPIR:
        SUPIR_COMMAND_TEMPLATE

    Example:
        export SUPIR_COMMAND_TEMPLATE='python /path/to/supir_single.py --input {input} --output {output} --scale {scale}'

    The placeholder tokens {input}, {output}, and {scale} will be replaced automatically.
    If SUPIR_COMMAND_TEMPLATE is not set or SUPIR fails, the function safely returns the seam-fixed image.
    """

    def __init__(
        self,
        suno_api_key,
        suno_base_url,
        template_path="PictureData/control.jpg",
        ip_adapter_repo="h94/IP-Adapter",
        ip_adapter_subfolder="sdxl_models",
        ip_adapter_weight_name="ip-adapter_sdxl.bin",
        ip_adapter_scale=0.45,
        supir_scale=4,
    ):
        self.suno_key = suno_api_key
        self.suno_base = suno_base_url
        self.template_path = template_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.custom_lora_path = "./models/lora/Custome.safetensors"
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight_name = ip_adapter_weight_name
        self.ip_adapter_scale = float(ip_adapter_scale)
        self.ip_adapter_ready = False

        self.supir_scale = int(supir_scale)
        self.supir_command_template = os.getenv("SUPIR_COMMAND_TEMPLATE", "").strip()
        self.supir_ready = bool(self.supir_command_template)

        self.cached_control_img = self._prepare_control_image(template_path)
        self.cached_canny_img = self._prepare_canny_image()

        self._init_models()
        self._init_supir()

    def _init_models(self):
        depth_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=self.dtype
        )

        canny_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=self.dtype
        )

        controlnets = [depth_controlnet, canny_controlnet]

        self.control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnets,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        )

        self.control_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.control_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        self.inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        )

        lora_dir = "./models/lora"
        os.makedirs(lora_dir, exist_ok=True)

        try:
            if os.path.exists(self.custom_lora_path):
                self.control_pipe.unload_lora_weights()
                self.control_pipe.load_lora_weights(self.custom_lora_path, adapter_name="pano_lora")
                self.control_pipe.set_adapters(["pano_lora"], adapter_weights=[0.45])
                print(f"✅ LoRA Loaded: {self.custom_lora_path}")
            else:
                print(f"⚠️ LoRA not found: {self.custom_lora_path}")
        except Exception as e:
            print(f"❌ LoRA Loading Error: {e}")

        try:
            raw_e_path = hf_hub_download(
                repo_id="jbilcke-hf/sdxl-panorama",
                filename="embeddings.pti",
                local_dir=lora_dir
            )
            state_dict = load_file(str(raw_e_path).strip())
            tokens = ["<s0>", "<s1>"]
            for i, (tokenizer, encoder) in enumerate([
                (self.control_pipe.tokenizer, self.control_pipe.text_encoder),
                (self.control_pipe.tokenizer_2, self.control_pipe.text_encoder_2)
            ]):
                key = f"text_encoders_{i}"
                if key in state_dict:
                    emb_weights = state_dict[key]
                    tokenizer.add_tokens(tokens)
                    encoder.resize_token_embeddings(len(tokenizer))
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    with torch.no_grad():
                        for idx, t_id in enumerate(token_ids):
                            encoder.get_input_embeddings().weight.data[t_id] = emb_weights[idx].to(
                                device=self.device,
                                dtype=self.dtype
                            )
            print("✅ Panorama embeddings ready.")
        except Exception as e:
            print(f"❌ Panorama embedding error: {e}")

        self._init_ip_adapter()

        if self.device == "cuda":
            self.control_pipe.to(device=self.device, dtype=self.dtype)

            if hasattr(self.control_pipe, "unet") and self.control_pipe.unet is not None:
                self.control_pipe.unet.to(device=self.device, dtype=self.dtype)
            if hasattr(self.control_pipe, "vae") and self.control_pipe.vae is not None:
                self.control_pipe.vae.to(device=self.device, dtype=self.dtype)
            if hasattr(self.control_pipe, "image_encoder") and self.control_pipe.image_encoder is not None:
                self.control_pipe.image_encoder.to(device=self.device, dtype=self.dtype)

            self.inpaint_pipe.enable_model_cpu_offload()
            self.inpaint_pipe.enable_attention_slicing()

        else:
            self.control_pipe.enable_model_cpu_offload()
            self.control_pipe.enable_attention_slicing()
            self.inpaint_pipe.enable_model_cpu_offload()
            self.inpaint_pipe.enable_attention_slicing()

        if self.ip_adapter_ready:
            print(
                "✅ IP-Adapter mode: control_pipe stays on CUDA; CPU offload and attention slicing are disabled for control_pipe.")

        self.control_pipe.vae.enable_tiling()
        self.inpaint_pipe.vae.enable_tiling()

    def _init_ip_adapter(self):
        try:
            self.control_pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder=self.ip_adapter_subfolder,
                weight_name=self.ip_adapter_weight_name
            )
            self.control_pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            self.ip_adapter_ready = True

            print(
                f"✅ IP-Adapter ready: {self.ip_adapter_repo}/"
                f"{self.ip_adapter_subfolder}/{self.ip_adapter_weight_name}, "
                f"scale={self.ip_adapter_scale}"
            )

        except Exception as e:
            self.ip_adapter_ready = False
            print(f"❌ IP-Adapter loading failed: {e}")

    def _init_supir(self):
        if self.supir_ready:
            print("✅ SUPIR command template detected. Final enhancement will use SUPIR.")
        else:
            print("⚠️ SUPIR_COMMAND_TEMPLATE is not set. Final enhancement will return seam-fixed image.")

    def _prepare_control_image(self, image_path):
        width, height = 1024, 512
        mask = np.zeros((height, width), dtype=np.uint8)

        horizon_y = int(height * 0.45)
        buffer_zone = int(height * 0.05)
        start_y = horizon_y + buffer_zone

        for y in range(start_y, height):
            color = int(255 * (y - start_y) / (height - start_y))
            mask[y, :] = color

        mask[:int(height * 0.12), :] = 0
        mask[int(height * 0.92):, :] = 0

        depth_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)

    def _prepare_canny_image(self):
        width, height = 1024, 512
        layout = np.array(self.cached_control_img.convert("RGB"))
        gray = cv2.cvtColor(layout, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        horizon_y = int(height * 0.50)
        cv2.line(edges, (0, horizon_y), (width - 1, horizon_y), 255, 1)

        cv2.ellipse(edges, (width // 2, int(height * 1.05)),
                    (int(width * 0.55), int(height * 0.32)),
                    0, 200, 340, 180, 1)
        cv2.ellipse(edges, (width // 2, int(height * 1.08)),
                    (int(width * 0.85), int(height * 0.45)),
                    0, 200, 340, 120, 1)

        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        canny_rgb = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
        return Image.fromarray(canny_rgb)

    def _load_reference_image(self, ref_image_path):
        if not ref_image_path:
            return None
        if not os.path.exists(ref_image_path):
            print(f"⚠️ Reference image not found: {ref_image_path}")
            return None
        try:
            ref = Image.open(ref_image_path).convert("RGB")
            return ref.resize((1024, 512), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"⚠️ Failed to load reference image {ref_image_path}: {e}")
            return None

    def _load_reference_images(self, scene_data=None, ref_image_path=None, max_refs=3):
        refs = []

        if scene_data and isinstance(scene_data, dict):
            for item in scene_data.get("reference_images", []):
                p = item.get("reference_image_path")
                img = self._load_reference_image(p)
                if img is not None:
                    refs.append(img)

        if not refs and ref_image_path:
            img = self._load_reference_image(ref_image_path)
            if img is not None:
                refs.append(img)

        return refs[:max_refs]

    def _refine_seams(self, image, prompt):
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        rolled_np = np.roll(img_np, shift=w // 2, axis=1)
        rolled_img = Image.fromarray(rolled_np)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask_width = int(w * 0.15)
        center_start = (w // 2) - (mask_width // 2)
        center_end = (w // 2) + (mask_width // 2)
        mask[:, center_start:center_end] = 255

        mask_pil = Image.fromarray(mask)
        mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=20))

        refine_prompt = (
            f"{prompt}, seamless 360 view, ultra-detailed landscape, "
            f"continuous horizontal line, no seams, cinematic photorealistic"
        )

        fixed_rolled_img = self.inpaint_pipe(
            prompt=refine_prompt,
            negative_prompt="seam, vertical line, split, gap, sharp edge, distorted horizon",
            image=rolled_img,
            mask_image=mask_blurred,
            width=1024,
            height=512,
            num_inference_steps=14,
            strength=0.35,
        ).images[0]

        final_np = np.array(fixed_rolled_img)
        final_np = np.roll(final_np, shift=-(w // 2), axis=1)
        return Image.fromarray(final_np)

    async def generate_image(self, prompt, step_id, folder=None, filename=None,  ref_image_path=None, scene_data=None, seed=None):
        print(f"Starting Production for Scene {step_id}...")
        if ref_image_path:
            print(f"🖼️ RAG reference conditioning: {ref_image_path}")

        clean_prompt = prompt.replace("(", "").replace(")", "").strip()
        full_prompt = (
            f"mypano_style, <s0><s1>, ({clean_prompt}:1.25), "
            f"360 panorama, high definition photography, 8K, photorealistic, seamless"
        )
        neg_prompt = (
            "grid, tiling, blocky, distorted, deformed horizon, vertical lines, watermark, "
            "blurry, ghosting, grey sky, flat color, color overflow, banding, compression artifacts, "
            "nadir, zenith, polar distortion, pinched top, swirl, person, crowd, buildings, indoor"
        )

        flush()

        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        try:
            folder = folder or "static/results/tmp"
            os.makedirs(folder, exist_ok=True)
            if not filename:
                filename = f"scene_{step_id}.png"
            file_path = os.path.join(folder, filename)

            ip_images = self._load_reference_images(
                scene_data=scene_data,
                ref_image_path=ref_image_path,
                max_refs=3
            )
            depth_img = self.cached_control_img
            canny_img = self.cached_canny_img

            pipe_kwargs = {
                "prompt": full_prompt,
                "negative_prompt": neg_prompt,
                "image": [depth_img, canny_img],
                "controlnet_conditioning_scale": [0.30, 0.08],
                "width": 1024,
                "height": 512,
                "num_inference_steps": 35,
                "guidance_scale": 6.5,
                "output_type": "latent",
                "generator": generator,
            }
            if self.ip_adapter_ready and ip_images:
                pipe_kwargs["ip_adapter_image"] = ip_images if len(ip_images) > 1 else ip_images[0]
            elif ref_image_path:
                print("⚠️ Reference path was provided, but IP-Adapter is unavailable or image loading failed.")

            output = self.control_pipe(**pipe_kwargs)
            latents = output.images
            del output

            latents = latents.to(dtype=self.dtype)

            with torch.no_grad():
                scaled_latents = latents / self.control_pipe.vae.config.scaling_factor
                image_tensor = self.control_pipe.vae.decode(scaled_latents, return_dict=False)[0]

                image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                base_img = self.control_pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
                del image_tensor, scaled_latents, latents

            flush()

            refined_img = self._refine_seams(base_img, prompt)
            del base_img

            refined_img.save(file_path)
            print(f"Scene {step_id} completed: {file_path} | seed={seed}")

            del refined_img
            flush()
            return f"http://localhost:8000/{file_path}", file_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            try:
                self.control_pipe.vae.to(dtype=self.dtype)
            except Exception:
                pass
            flush()
            return None, None

    def _run_supir(self, input_path, output_path, scale=None):
        if not self.supir_ready:
            return False
        scale = int(scale or self.supir_scale)
        command = self.supir_command_template.format(
            input=input_path,
            output=output_path,
            scale=scale
        )
        print(f"🚀 Running SUPIR: {command}")
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            print("❌ SUPIR failed.")
            print(result.stderr[-2000:])
            return False
        if not os.path.exists(output_path):
            print(f"❌ SUPIR command finished, but output not found: {output_path}")
            return False
        return True

    def upscale_image(self, input_path, output_path, scale=2):
        import os
        import subprocess

        command_template = os.getenv("SUPIR_COMMAND_TEMPLATE")

        if not command_template:
            print("⚠️ SUPIR_COMMAND_TEMPLATE is not set.")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        command = command_template.format(
            input=os.path.abspath(input_path).replace("\\", "/"),
            output=os.path.abspath(output_path).replace("\\", "/"),
            scale=scale
        )

        print("\n🚀 [SUPIR] Running command:")
        print(command)

        try:
            subprocess.run(command, shell=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"❌ [SUPIR] Failed: {e}")
            return False

    def generate_music(self, prompt, style, title):
        base_url = (self.suno_base or "").rstrip('/')
        if not base_url:
            print("❌ [Suno API] Base URL is missing!")
            return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        headers = {
            "Authorization": f"Bearer {self.suno_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "customMode": True,
            "instrumental": True,
            "model": "V4_5ALL",
            "callBackUrl": "https://api.example.com/callback",
            "prompt": prompt,
            "style": style,
            "title": title,
        }

        try:
            gen_endpoint = f"{base_url}/generat"
            print(f"🎵 [Suno API] Calling: {gen_endpoint}")
            response = requests.post(gen_endpoint, json=payload, headers=headers, timeout=30)
            print(f" Status Code: {response.status_code}")
            print(f" Raw Response: '{response.text}'")

            if response.status_code != 200:
                print(f"❌ [Suno API] Error {response.status_code}")
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

            try:
                res_json = response.json()
            except Exception:
                print("❌ [Suno API] Failed to parse JSON from response.")
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

            task_id = res_json.get("data", {}).get("taskId") if isinstance(res_json, dict) else None
            if not task_id:
                print("❌ [Suno API] No TaskID in response.")
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

            poll_endpoint = f"{base_url}/generate/record-info"
            for i in range(50):
                time.sleep(10)
                status_res = requests.get(poll_endpoint, params={"taskId": task_id}, headers=headers)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    data_obj = status_data.get("data", {})
                    status = data_obj.get("status")
                    print(f"   [Polling {i + 1}] {status}")

                    if status in ["SUCCESS", "FIRST_SUCCESS"]:
                        suno_data = data_obj.get("response", {}).get("sunoData", [])
                        if suno_data:
                            url = suno_data[0].get("audioUrl")
                            if url:
                                return url

        except Exception as e:
            print(f"❌ [Suno API] Exception: {e}")

        return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
