import os
import gc
import time
import cv2
import numpy as np
import requests
import torch
from safetensors.torch import load_file
#from transformers import pipeline
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class ProductionAgent:
    def __init__(self, suno_api_key, suno_base_url,
                 template_path="PictureData/control.jpg"):
        self.suno_key = suno_api_key
        self.suno_base = suno_base_url
        self.template_path = template_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        #self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=self.device)

        self.custom_lora_path = "./models/lora/MyHighResPano_SDXL-000005.safetensors"
        self.upscaler_model_path = './models/realesrgan/RealESRGAN_x4plus.pth'

        self._init_models()
        self._init_upscaler()

    def _init_models(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=self.dtype
        )

        self.control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
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

        # LoRA
        lora_dir = "./models/lora"
        os.makedirs(lora_dir, exist_ok=True)

        try:
            if os.path.exists(self.custom_lora_path):
                self.control_pipe.unload_lora_weights()
                self.control_pipe.load_lora_weights(self.custom_lora_path, adapter_name="pano_lora")
                self.control_pipe.set_adapters(["pano_lora"], adapter_weights=[0.5])
                print(f"✅ LoRA Loaded: {self.custom_lora_path}")
            else:
                print(f"❌ Error: Not found {self.custom_lora_path}")
        except Exception as e:
            print(f"❌ LoRA Loading Error: {e}")

        # Embedding
        try:
            raw_e_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti",
                                         local_dir=lora_dir)
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
                            encoder.get_input_embeddings().weight.data[t_id] = emb_weights[idx].to(device=self.device,
                                                                                                   dtype=self.dtype)
            print("✅ Panorama Embeddings ready.")
        except Exception as e:
            print(f"❌ Embedding Error: {e}")

        self.control_pipe.vae.to(dtype=torch.bfloat16)
        self.inpaint_pipe.vae.to(dtype=torch.bfloat16)

        self.control_pipe.enable_sequential_cpu_offload()
        self.inpaint_pipe.enable_sequential_cpu_offload()

        self.control_pipe.enable_attention_slicing()
        self.inpaint_pipe.enable_attention_slicing()

        self.control_pipe.vae.enable_tiling()
        self.inpaint_pipe.vae.enable_tiling()

    def _init_upscaler(self):
        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=4,
                model_path=self.upscaler_model_path,
                model=model,
                tile=512,
                tile_pad=32,
                pre_pad=0,
                half=True,
                device=self.device
            )
            print("✅ Real-ESRGAN Ready.")
        except Exception as e:
            print(f"❌ Upscaler Init Error (Check .pth path): {e}")
            self.upscaler = None

    def _prepare_control_image(self, image_path):

        width, height = 1024, 512
        mask = np.zeros((height, width), dtype=np.uint8)

        horizon_y = int(height * 0.5)
        buffer_zone = int(height * 0.05)
        start_y = horizon_y + buffer_zone

        for y in range(start_y, height):
            color = int(255 * (y - start_y) / (height - start_y))
            mask[y, :] = color

        mask[:int(height * 0.1), :] = 0

        depth_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)


    def _refine_seams(self, image, prompt):
        """ Smoothing of extreme regions in ERP images  """
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # roll 50%
        rolled_np = np.roll(img_np, shift=w // 2, axis=1)
        rolled_img = Image.fromarray(rolled_np)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask_width = int(w * 0.15)
        center_start = (w // 2) - (mask_width // 2)
        center_end = (w // 2) + (mask_width // 2)
        mask[:, center_start:center_end] = 255

        mask_pil = Image.fromarray(mask)
        mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=18))

        refine_prompt = (f"{prompt}, seamless 360 view, ultra-detailed landscape, "
                         f"continuous horizontal line, no seams, cinematic photorealistic")

        fixed_rolled_img = self.inpaint_pipe(
            prompt=refine_prompt,
            negative_prompt="seam, vertical line, split, gap, sharp edge, distorted horizon",
            image=rolled_img,
            mask_image=mask_blurred,
            width=1024,
            height=512,
            num_inference_steps=30,
            strength=0.45,
        ).images[0]

        # rollback
        final_np = np.array(fixed_rolled_img)
        final_np = np.roll(final_np, shift=-(w // 2), axis=1)

        return Image.fromarray(final_np)

    def generate_image(self, prompt, step_id):
        """ Generate custom 360 pictures with the entire pipeline """
        print(f"Starting Production for Scene {step_id}...")

        boosted_prompt = ", ".join([f"({word.strip()}:1.5)" for word in prompt.split(',') if word.strip()])
        full_prompt = (f"mypano_style, {boosted_prompt}, 360 panorama, ultra-high definition photography, "
                       f"8K, seamless, HDR, photorealistic, <s0><s1>")
        neg_prompt = ("grid, tiling, blocky, distorted, deformed horizon, vertical lines, tiling, watermark, "
                      "blurry, ghosting, grey sky, flat color, color overflow, banding, compression artifacts,"
                      "grass on zenith, trees in sky, person, crowd")

        control_img = self._prepare_control_image(self.template_path)

        flush()

        try:
            latents = self.control_pipe(
                prompt=full_prompt,
                negative_prompt= neg_prompt,
                image=control_img,
                controlnet_conditioning_scale=0.25,
                width=1024,
                height=512,
                num_inference_steps=35,
                guidance_scale=6.5,
                output_type = "latent",
            ).images

            latents = latents.to(dtype=torch.bfloat16)

            with torch.no_grad():
                scaled_latents = latents / self.control_pipe.vae.config.scaling_factor
                image_tensor = self.control_pipe.vae.decode(scaled_latents, return_dict=False)[0]

                image_tensor[:, :, :, 0] = image_tensor[:, :, :, 2]
                image_tensor[:, :, :, -1] = image_tensor[:, :, :, -3]

                base_img = self.control_pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

            os.makedirs("static/base", exist_ok=True)
            base_img.save(f"static/base/scene_{step_id}.png")

            flush()

            # Refining
            refined_img = self._refine_seams(base_img, prompt)

            if self.upscaler:
                img_cv = cv2.cvtColor(np.array(refined_img), cv2.COLOR_RGB2BGR)
                del refined_img
                output_cv, _ = self.upscaler.enhance(img_cv, outscale=4)
                del img_cv
                candidate_img = Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
                del output_cv
            else:
                candidate_img = refined_img

            os.makedirs("static/candidate", exist_ok=True)
            file_path = f"static/candidate/scene_{step_id}.png"
            candidate_img.save(file_path)

            print(f"Scene {step_id} completed: {file_path}")

            try:
                if 'img_cv' in locals(): del img_cv
                if 'output_cv' in locals(): del output_cv
                if 'candidate_img' in locals(): del candidate_img
                if 'refined_img' in locals(): del refined_img
            except:
                pass

            flush()
            return f"http://localhost:8000/{file_path}", file_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            self.control_pipe.vae.to(dtype=self.dtype)
            return None, None

    def generate_music(self, prompt, style, title):

        base_url = self.suno_base.rstrip('/')
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

            if res_json is None:
                print("❌ [Suno API] res_json is None.")
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

            task_id = res_json.get("data", {}).get("taskId") if isinstance(res_json, dict) else None

            if not task_id:
                print(f"❌ [Suno API] No TaskID in response.")
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

            poll_endpoint = f"{base_url}/generate/record-info"
            for i in range(20):
                time.sleep(10)
                status_res = requests.get(poll_endpoint, params={"taskId": task_id}, headers=headers)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    data_obj = status_data.get("data", {})
                    status = data_obj.get("status")
                    print(f"   [Polling {i + 1}] {status}")

                    if status == "SUCCESS":
                        suno_data = data_obj.get("response", {}).get("sunoData", [])
                        if suno_data:
                            return suno_data[0].get("audioUrl")

        except Exception as e:
            print(f"❌ [Suno API] Exception: {e}")

        return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"