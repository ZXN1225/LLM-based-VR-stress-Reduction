import os
import gc
import time
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import types
import random
from PipelineGuardrails import public_static_url
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from PhotoFinisher import PhotoFinisher

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

try:
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
except Exception:
    StableDiffusionXLControlNetImg2ImgPipeline = None

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except Exception:
    RealESRGANer = None
    RRDBNet = None

try:
    from spandrel import ImageModelDescriptor, ModelLoader
except Exception:
    ImageModelDescriptor = None
    ModelLoader = None


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


class ProductionAgent:
    """
    Dual-generation-pipeline ProductionAgent.

    Build note:
      - Generation stage: horizontal latent circular padding is enabled.
      - Seam refine stage: reverted to the original roll-to-center inpaint method,
        without circular padding in the seam inpaint pipeline. This avoids the
        large vertical seam artifact observed after refinement.


    Architecture:
      Final enhancer uses fidelity-preserving PhotoFinisher instead of AI super-resolution.
      1) plain_pipe: SDXL + ControlNet + LoRA, WITHOUT IP-Adapter.
         Used by DIRECT_GENERATION / no-RAG baselines.
      2) rag_pipe: SDXL + ControlNet + LoRA + IP-Adapter.
         Used by FULL / AUDIT_ONE / RAG_ONLY when reference images exist.
      3) persistent seam_pipe: SDXL inpaint loaded once and reused.
      4) Fidelity PhotoFinisher for final 4x clarity enhancement.

    This avoids the Direct Generation image_embeds error because the direct baseline never
    touches a pipeline that has IP-Adapter attached.
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
        upscale_scale=1,
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

        self.plain_pipe = None
        self.rag_pipe = None
        self.rag_ip_adapter_ready = False
        self.active_generation_pipe_name = None
        self.generation_device_mode = os.getenv("GENERATION_PIPE_DEVICE_MODE", "swap").lower().strip()
        # swap: keep inactive generation pipes on CPU, move selected one to CUDA.
        # keep_cuda: leave generation pipes on CUDA after first use. Faster but higher VRAM.
        # cpu: run generation on CPU fallback.

        self.enable_latent_circular_padding = os.getenv("ENABLE_LATENT_CIRCULAR_PADDING", "1") == "1"
        self.circular_padding_vertical_mode = os.getenv("CIRCULAR_PADDING_VERTICAL_MODE", "constant").lower().strip()
        self.circular_padding_modules = {
            x.strip().lower() for x in os.getenv(
                "PANORAMA_CIRCULAR_MODULES", "unet,controlnet,vae"
            ).split(",") if x.strip()
        }
        self.seam_use_latent_circular_padding = os.getenv("SEAM_USE_LATENT_CIRCULAR_PADDING", "0") == "1"

        self.upscale_scale = int(os.getenv("PHOTO_FINISHER_SCALE", os.getenv("AUTO_UPSCALE_SCALE", str(upscale_scale))))
        self.upscale_backend = "fidelity_finisher"
        self.upscale_ready = True
        self.photo_finisher = PhotoFinisher.from_env()

        # Progressive latent/high-resolution generation.
        # Stage 1 still generates a 1024x512 latent panorama. Optional img2img stages then
        # upsample the decoded image and re-denoise it with the same ControlNet pipeline.
        # This is a practical SDXL "hires-fix" implementation for the current project: it
        # adds real diffusion detail before the final PhotoFinisher, without using ESRGAN.
        self.progressive_hires_enabled = os.getenv("PROGRESSIVE_HIRES_ENABLED", "1") == "1"
        self.progressive_hires_stages = self._parse_hires_stages(
            os.getenv("PROGRESSIVE_HIRES_STAGES", "2048x1024,3072x1536")
        )
        self.progressive_hires_steps = int(os.getenv("PROGRESSIVE_HIRES_STEPS", "20"))
        self.progressive_hires_strengths = self._parse_float_list(
            os.getenv("PROGRESSIVE_HIRES_STRENGTHS", "0.40,0.28")
        )
        self.progressive_hires_guidance_scale = float(os.getenv("PROGRESSIVE_HIRES_GUIDANCE_SCALE", "5.0"))
        self.progressive_hires_controlnet_scale = float(os.getenv("PROGRESSIVE_HIRES_CONTROLNET_SCALE", "0.15"))
        self.progressive_hires_use_ip_adapter = os.getenv("PROGRESSIVE_HIRES_USE_IP_ADAPTER", "1") == "1"

        self.seam_refiner_enabled = os.getenv("SEAM_REFINER_ENABLED", "1") == "1"
        self.seam_steps = int(os.getenv("SEAM_REFINE_STEPS", "8"))
        self.seam_strength = float(os.getenv("SEAM_REFINE_STRENGTH", "0.28"))
        self.seam_device_mode = os.getenv("SEAM_REFINER_DEVICE_MODE", "swap").lower().strip()
        self.seam_move_generation_to_cpu = os.getenv("SEAM_MOVE_GENERATION_TO_CPU", "1") == "1"
        self.seam_pipe = None
        self.seam_ready = False

        self.cached_control_img = self._prepare_control_image(template_path)

        self._init_generation_pipelines()
        self._init_realesrgan()
        self._init_seam_refiner()

    # ------------------------------------------------------------------
    # Panorama latent circular padding
    # ------------------------------------------------------------------
    def _patch_conv2d_horizontal_circular(self, root_module, label="module"):
        """
        Patch Conv2d layers to use horizontal circular padding only.

        """
        if root_module is None:
            return 0

        patched = 0
        for module in root_module.modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            if getattr(module, "_horizontal_circular_patched", False):
                continue

            padding = module.padding
            if isinstance(padding, int):
                pad_h, pad_w = padding, padding
            else:
                pad_h, pad_w = int(padding[0]), int(padding[1])

            if pad_h == 0 and pad_w == 0:
                continue

            module._original_padding_for_horizontal_circular = (pad_h, pad_w)
            module.padding = (0, 0)
            module._horizontal_circular_patched = True

            vertical_mode = self.circular_padding_vertical_mode

            def _horizontal_circular_forward(conv_self, input_tensor):
                ph, pw = conv_self._original_padding_for_horizontal_circular
                x = input_tensor

                # Wrap only the left/right boundary.
                if pw > 0:
                    x = F.pad(x, (pw, pw, 0, 0), mode="circular")

                if ph > 0:
                    if vertical_mode == "replicate":
                        x = F.pad(x, (0, 0, ph, ph), mode="replicate")
                    else:
                        x = F.pad(x, (0, 0, ph, ph), mode="constant", value=0.0)

                return F.conv2d(
                    x,
                    conv_self.weight,
                    conv_self.bias,
                    conv_self.stride,
                    (0, 0),
                    conv_self.dilation,
                    conv_self.groups,
                )

            module.forward = types.MethodType(_horizontal_circular_forward, module)
            patched += 1

        if patched > 0:
            print(f"🌀 Enabled horizontal latent circular padding for {label}: patched {patched} Conv2d layers.")
        else:
            print(f"⚠️ No Conv2d layers patched for {label}.")
        return patched

    def _enable_panorama_circular_padding_for_pipe(self, pipe, label="pipe", include_modules=None):
        if not self.enable_latent_circular_padding or pipe is None:
            return

        include_modules = include_modules or self.circular_padding_modules

        try:
            if "unet" in include_modules and hasattr(pipe, "unet") and pipe.unet is not None:
                self._patch_conv2d_horizontal_circular(pipe.unet, f"{label}.unet")
            if "controlnet" in include_modules and hasattr(pipe, "controlnet") and pipe.controlnet is not None:
                self._patch_conv2d_horizontal_circular(pipe.controlnet, f"{label}.controlnet")
            if "vae" in include_modules and hasattr(pipe, "vae") and pipe.vae is not None:
                self._patch_conv2d_horizontal_circular(pipe.vae, f"{label}.vae")
        except Exception as e:
            print(f"⚠️ Failed to enable horizontal circular padding for {label}: {e}")

    # ------------------------------------------------------------------
    # Generation pipelines
    # ------------------------------------------------------------------
    def _build_generation_pipe(self, label: str, with_ip_adapter: bool):
        print(f"🚀 Loading {label} generation pipeline | IP-Adapter={with_ip_adapter}")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=self.dtype,
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

        # Generation-stage panorama periodic boundary handling.
        # This is applied before moving the model to CPU/GPU and remains active.
        self._enable_panorama_circular_padding_for_pipe(pipe, label=label)

        self._load_lora_and_pano_embeddings(pipe, label)

        ip_ready = False
        if with_ip_adapter:
            try:
                pipe.load_ip_adapter(
                    self.ip_adapter_repo,
                    subfolder=self.ip_adapter_subfolder,
                    weight_name=self.ip_adapter_weight_name,
                )
                pipe.set_ip_adapter_scale(self.ip_adapter_scale)
                ip_ready = True
                print(
                    f"✅ {label} IP-Adapter ready: {self.ip_adapter_repo}/"
                    f"{self.ip_adapter_subfolder}/{self.ip_adapter_weight_name}, scale={self.ip_adapter_scale}"
                )
            except Exception as e:
                ip_ready = False
                print(f"❌ {label} IP-Adapter loading failed: {e}")

        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

        # Keep initialized on CPU by default. generate_image() moves the selected pipeline to GPU.
        if self.device == "cuda" and self.generation_device_mode == "keep_cuda":
            pipe.to(device=self.device, dtype=self.dtype)
            self._ensure_pipe_modules_dtype(pipe)
            print(f"✅ {label} generation pipeline kept on CUDA.")
        elif self.device == "cuda":
            pipe.to("cpu")
            print(f"✅ {label} generation pipeline loaded once; resident on CPU until used.")
        else:
            pipe.enable_attention_slicing()
            pipe.to("cpu")
            print(f"✅ {label} generation pipeline loaded on CPU.")

        return pipe, ip_ready

    def _load_lora_and_pano_embeddings(self, pipe, label: str):
        lora_dir = "./models/lora"
        os.makedirs(lora_dir, exist_ok=True)

        try:
            if os.path.exists(self.custom_lora_path):
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass
                pipe.load_lora_weights(self.custom_lora_path, adapter_name="pano_lora")
                pipe.set_adapters(["pano_lora"], adapter_weights=[0.45])
                print(f"✅ {label} LoRA Loaded: {self.custom_lora_path}")
            else:
                print(f"⚠️ {label} LoRA not found: {self.custom_lora_path}")
        except Exception as e:
            print(f"❌ {label} LoRA Loading Error: {e}")

        try:
            raw_e_path = hf_hub_download(
                repo_id="jbilcke-hf/sdxl-panorama",
                filename="embeddings.pti",
                local_dir=lora_dir,
            )
            state_dict = load_file(str(raw_e_path).strip())
            tokens = ["<s0>", "<s1>"]
            for i, (tokenizer, encoder) in enumerate([
                (pipe.tokenizer, pipe.text_encoder),
                (pipe.tokenizer_2, pipe.text_encoder_2),
            ]):
                key = f"text_encoders_{i}"
                if key in state_dict:
                    emb_weights = state_dict[key]
                    existing_vocab = tokenizer.get_vocab()
                    tokens_to_add = [t for t in tokens if t not in existing_vocab]
                    if tokens_to_add:
                        tokenizer.add_tokens(tokens_to_add)
                        encoder.resize_token_embeddings(len(tokenizer))
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    with torch.no_grad():
                        for idx, t_id in enumerate(token_ids):
                            encoder.get_input_embeddings().weight.data[t_id] = emb_weights[idx].to(
                                device=encoder.device,
                                dtype=self.dtype,
                            )
            print(f"✅ {label} panorama embeddings ready.")
        except Exception as e:
            print(f"❌ {label} panorama embedding error: {e}")

    def _init_generation_pipelines(self):
        self.plain_pipe, _ = self._build_generation_pipe("Plain/no-RAG", with_ip_adapter=False)
        self.rag_pipe, self.rag_ip_adapter_ready = self._build_generation_pipe("RAG/reference", with_ip_adapter=True)
        flush()

    def _ensure_pipe_modules_dtype(self, pipe):
        if self.device != "cuda" or pipe is None:
            return
        try:
            if hasattr(pipe, "unet") and pipe.unet is not None:
                pipe.unet.to(device=self.device, dtype=self.dtype)
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.to(device=self.device, dtype=self.dtype)
            if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
                pipe.image_encoder.to(device=self.device, dtype=self.dtype)
        except Exception as e:
            print(f"⚠️ Failed to enforce generation pipe dtype: {e}")

    def _move_generation_pipes_to_cpu(self):
        for name, pipe in [("plain", self.plain_pipe), ("rag", self.rag_pipe)]:
            if pipe is None:
                continue
            try:
                print(f"🧹 Moving {name}_pipe to CPU...")
                pipe.to("cpu")
            except Exception as e:
                print(f"⚠️ Failed to move {name}_pipe to CPU: {e}")
        self.active_generation_pipe_name = None
        flush()

    def _select_generation_pipe(self, use_rag_pipe: bool):
        name = "rag" if use_rag_pipe else "plain"
        pipe = self.rag_pipe if use_rag_pipe else self.plain_pipe
        if pipe is None:
            raise RuntimeError(f"Generation pipeline '{name}' is not initialized.")

        if self.device == "cuda" and self.generation_device_mode != "cpu":
            if self.generation_device_mode == "swap" and self.active_generation_pipe_name != name:
                # Move only the inactive generation pipe to CPU before moving selected one to CUDA.
                other_name = "plain" if name == "rag" else "rag"
                other_pipe = self.plain_pipe if name == "rag" else self.rag_pipe
                if other_pipe is not None:
                    try:
                        print(f"🧹 Moving inactive {other_name}_pipe to CPU...")
                        other_pipe.to("cpu")
                    except Exception as e:
                        print(f"⚠️ Failed to move inactive {other_name}_pipe to CPU: {e}")
                flush()
            if self.active_generation_pipe_name != name or self.generation_device_mode == "keep_cuda":
                try:
                    print(f"🚀 Moving selected {name}_pipe to GPU...")
                    pipe.to(device=self.device, dtype=self.dtype)
                    self._ensure_pipe_modules_dtype(pipe)
                    self.active_generation_pipe_name = name
                except Exception as e:
                    raise RuntimeError(f"Failed to move {name}_pipe to GPU: {e}")
                flush()
        return pipe, name

    # Backward-compatible names used by older code.
    def _move_control_pipe_to_cpu(self):
        self._move_generation_pipes_to_cpu()

    def _move_control_pipe_to_gpu(self):
        # Restore the previously selected pipe only when needed by generate_image().
        return

    # ------------------------------------------------------------------
    # Fidelity-preserving final enhancement
    # ------------------------------------------------------------------
    def _init_realesrgan(self):

        self.upscale_backend = "fidelity_finisher"
        self.upscale_ready = self.photo_finisher is not None and self.photo_finisher.enabled
        if self.upscale_ready:
            print(
                f"✅ Fidelity PhotoFinisher ready: scale={self.upscale_scale}, "
            )
        else:
            print("⚠️ Fidelity PhotoFinisher disabled. Final images will be copied without enhancement.")

    # ------------------------------------------------------------------
    # Control/reference helpers
    # ------------------------------------------------------------------
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
                if not isinstance(item, dict):
                    continue
                img = self._load_reference_image(item.get("reference_image_path"))
                if img is not None:
                    refs.append(img)
        if not refs and ref_image_path:
            img = self._load_reference_image(ref_image_path)
            if img is not None:
                refs.append(img)
        return refs[:max_refs]

    # ------------------------------------------------------------------
    # Persistent seam refinement
    # ------------------------------------------------------------------
    def _get_seam_dtype(self):
        name = os.getenv("SEAM_REFINE_DTYPE", "bf16").lower().strip()
        if self.device != "cuda":
            return torch.float32
        if name in ["bf16", "bfloat16"]:
            return torch.bfloat16
        if name in ["fp16", "float16", "half"]:
            return torch.float16
        return torch.float32

    def _init_seam_refiner(self):
        if not self.seam_refiner_enabled:
            print("⚠️ Persistent seam refiner disabled by SEAM_REFINER_ENABLED=0.")
            self.seam_ready = False
            return
        try:
            seam_dtype = self._get_seam_dtype()
            print(
                f"🧵 Loading persistent SDXL seam refiner once: "
                f"dtype={seam_dtype}, mode={self.seam_device_mode}, steps={self.seam_steps}, strength={self.seam_strength}"
            )
            self.seam_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=seam_dtype,
                variant="fp16" if seam_dtype in [torch.float16, torch.bfloat16] else None,
                use_safetensors=True,
            )
            # IMPORTANT:
            # Do NOT apply latent horizontal circular padding to the seam inpaint/refine pipeline.
            # Generation uses circular padding to improve left/right periodic consistency,
            # but the seam refiner still uses the older roll-to-center method below.
            # Enabling circular padding inside the inpaint pipeline can create a visible vertical
            # artifact near the rolled center seam.
            if self.seam_use_latent_circular_padding:
                print("⚠️ SEAM_USE_LATENT_CIRCULAR_PADDING is ignored in this build; seam refine uses roll-to-center only.")
            self.seam_pipe.vae.enable_tiling()
            if self.device == "cuda" and self.seam_device_mode == "keep_cuda":
                self.seam_pipe.to(device=self.device, dtype=seam_dtype)
                print("✅ Persistent seam refiner loaded and kept on CUDA.")
            else:
                self.seam_pipe.to("cpu")
                print("✅ Persistent seam refiner loaded once; resident on CPU until seam phase.")
            self.seam_ready = True
            flush()
        except Exception as e:
            self.seam_pipe = None
            self.seam_ready = False
            print(f"❌ Persistent seam refiner init failed: {e}")
            flush()

    def _move_seam_pipe_to_gpu(self):
        if self.device != "cuda" or self.seam_pipe is None:
            return
        try:
            seam_dtype = self._get_seam_dtype()
            print("🚀 Moving persistent seam refiner to GPU...")
            self.seam_pipe.to(device=self.device, dtype=seam_dtype)
        except Exception as e:
            print(f"⚠️ Failed to move seam refiner to GPU: {e}")
        flush()

    def _move_seam_pipe_to_cpu(self):
        if self.seam_pipe is None or self.seam_device_mode == "keep_cuda":
            return
        try:
            print("🧹 Moving persistent seam refiner back to CPU...")
            self.seam_pipe.to("cpu")
        except Exception as e:
            print(f"⚠️ Failed to move seam refiner to CPU: {e}")
        flush()

    def _refine_single_seam_persistent(self, image, prompt):
        if self.seam_pipe is None:
            return image
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        rolled_np = np.roll(img_np, shift=w // 2, axis=1)
        rolled_img = Image.fromarray(rolled_np)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask_width = int(w * float(os.getenv("SEAM_REFINE_MASK_RATIO", "0.15")))
        center_start = (w // 2) - (mask_width // 2)
        center_end = (w // 2) + (mask_width // 2)
        mask[:, center_start:center_end] = 255
        blur_radius = float(os.getenv("SEAM_REFINE_MASK_BLUR", "20"))
        mask_blurred = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=blur_radius))

        refine_prompt = (
            f"{prompt}, seamless 360 view, ultra-detailed landscape, "
            f"continuous horizontal line, no seams, cinematic photorealistic"
        )
        negative_prompt = (
            "seam, vertical line, split, gap, sharp edge, distorted horizon, artifacts, "
            "over-sharpened texture, unnatural contrast, painterly, cartoon"
        )

        with torch.no_grad():
            result = self.seam_pipe(
                prompt=refine_prompt,
                negative_prompt=negative_prompt,
                image=rolled_img,
                mask_image=mask_blurred,
                width=w,
                height=h,
                num_inference_steps=int(self.seam_steps),
                strength=float(self.seam_strength),
            ).images[0]
        final_np = np.array(result)
        final_np = np.roll(final_np, shift=-(w // 2), axis=1)
        return Image.fromarray(final_np)

    def refine_seams_for_paths(self, scene_items, output_folder="static/results/seam_refined", overwrite=False):
        if not scene_items:
            return {}
        os.makedirs(output_folder, exist_ok=True)
        fallback = {item.get("step", idx + 1): item.get("image_path") for idx, item in enumerate(scene_items)}
        if not self.seam_ready or self.seam_pipe is None:
            print("⚠️ Persistent seam refiner is not ready. Keeping base images.")
            return {k: v for k, v in fallback.items() if v}

        if self.device == "cuda" and self.seam_move_generation_to_cpu:
            self._move_generation_pipes_to_cpu()
        if self.device == "cuda" and self.seam_device_mode != "cpu":
            self._move_seam_pipe_to_gpu()

        refined_paths = {}
        try:
            for idx, item in enumerate(scene_items, start=1):
                step_id = item.get("step", idx)
                image_path = item.get("image_path")
                prompt = item.get("prompt", "seamless restorative natural landscape")
                if not image_path or not os.path.exists(image_path):
                    print(f"⚠️ [PersistentSeam] skip step {step_id}: missing image {image_path}")
                    continue
                print(f"🧵 [PersistentSeam] refining step {step_id}: {image_path}")
                try:
                    base_img = Image.open(image_path).convert("RGB")
                    refined_img = self._refine_single_seam_persistent(base_img, prompt)
                    if overwrite:
                        out_path = image_path
                    else:
                        # IMPORTANT: in batched experiments many case images are named identically
                        # inside different folders, e.g. case_01/base/init.png, case_02/base/init.png.
                        # If we save all seam outputs as output_folder/init_seam.png, every case will
                        # point to the last refined image. Use explicit unique output_filename when provided,
                        # otherwise include step_id to avoid overwriting.
                        output_filename = item.get("output_filename") or item.get("seam_filename")
                        if output_filename:
                            out_name = str(output_filename)
                            if not out_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                out_name += ".png"
                        else:
                            name = os.path.splitext(os.path.basename(image_path))[0]
                            out_name = f"step_{step_id}_{name}_seam.png"
                        out_path = os.path.join(output_folder, out_name)
                    refined_img.save(out_path)
                    refined_paths[step_id] = out_path
                    print(f"✅ [PersistentSeam] saved step {step_id}: {out_path}")
                    del base_img, refined_img
                    flush()
                except Exception as e:
                    print(f"⚠️ [PersistentSeam] failed step {step_id}: {e}. Keeping base image.")
                    refined_paths[step_id] = image_path
                    flush()
        finally:
            if self.device == "cuda" and self.seam_device_mode == "swap":
                self._move_seam_pipe_to_cpu()
        return refined_paths

    # ------------------------------------------------------------------
    # Progressive latent/high-resolution generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_hires_stages(value):
        stages = []
        for item in str(value or "").replace(";", ",").split(","):
            item = item.strip().lower()
            if not item:
                continue
            try:
                if "x" in item:
                    w_s, h_s = item.split("x", 1)
                elif ":" in item:
                    w_s, h_s = item.split(":", 1)
                else:
                    continue
                w, h = int(w_s), int(h_s)
                if w > 0 and h > 0:
                    stages.append((w, h))
            except Exception:
                continue
        return stages

    @staticmethod
    def _parse_float_list(value):
        vals = []
        for item in str(value or "").replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                vals.append(float(item))
            except Exception:
                pass
        return vals

    def _get_hires_strength(self, stage_idx):
        if self.progressive_hires_strengths:
            idx = min(stage_idx, len(self.progressive_hires_strengths) - 1)
            return float(self.progressive_hires_strengths[idx])
        return 0.22 if stage_idx == 0 else 0.16

    def _get_hires_img2img_pipe(self, pipe, pipe_name="generation"):
        if StableDiffusionXLControlNetImg2ImgPipeline is None:
            print("⚠️ Progressive hires disabled: StableDiffusionXLControlNetImg2ImgPipeline is unavailable in this diffusers install.")
            return None
        if pipe is None:
            return None
        cached = getattr(pipe, "_progressive_hires_img2img_pipe", None)
        if cached is not None:
            return cached
        try:
            print(f"🔍 Building progressive hires img2img pipe for {pipe_name} from existing SDXL components...")
            img2img_pipe = StableDiffusionXLControlNetImg2ImgPipeline(**pipe.components)
            img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            # Components are shared with the generation pipe. The selected generation pipe is
            # already on the target device, so this does not duplicate the full model in VRAM.
            setattr(pipe, "_progressive_hires_img2img_pipe", img2img_pipe)
            return img2img_pipe
        except Exception as e:
            print(f"⚠️ Failed to build progressive hires img2img pipe: {e}")
            return None

    def _run_progressive_hires(self, pipe, pipe_name, base_img, full_prompt, neg_prompt, generator, use_rag_pipe=False, ip_images=None):
        if not self.progressive_hires_enabled:
            return base_img
        if not self.progressive_hires_stages:
            return base_img

        img2img_pipe = self._get_hires_img2img_pipe(pipe, pipe_name=pipe_name)
        if img2img_pipe is None:
            print("⚠️ Progressive hires unavailable. Keeping 1024x512 base image.")
            return base_img

        current_img = base_img.convert("RGB")
        ip_images = ip_images or []
        for stage_idx, (target_w, target_h) in enumerate(self.progressive_hires_stages):
            strength = float(self._get_hires_strength(stage_idx))
            steps = int(self.progressive_hires_steps)
            print(
                f"🔎 [ProgressiveHires] stage {stage_idx + 1}/{len(self.progressive_hires_stages)}: "
                f"{current_img.size[0]}x{current_img.size[1]} -> {target_w}x{target_h}, "
                f"strength={strength}, steps={steps}"
            )

            init_img = current_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            control_img = self.cached_control_img.resize((target_w, target_h), Image.Resampling.BICUBIC)

            kwargs = {
                "prompt": full_prompt,
                "negative_prompt": neg_prompt,
                "image": init_img,
                "control_image": control_img,
                "strength": strength,
                "width": target_w,
                "height": target_h,
                "num_inference_steps": steps,
                "guidance_scale": float(self.progressive_hires_guidance_scale),
                "controlnet_conditioning_scale": float(self.progressive_hires_controlnet_scale),
                "generator": generator,
            }
            if use_rag_pipe and self.progressive_hires_use_ip_adapter and ip_images:
                kwargs["ip_adapter_image"] = ip_images[0]

            try:
                with torch.no_grad():
                    out = img2img_pipe(**kwargs)
                    next_img = out.images[0].convert("RGB")
                del out, init_img, control_img
                current_img = next_img
                flush()
            except TypeError as e:
                # Some older diffusers versions use image=control and init_image=image for
                # ControlNet img2img. Retry once with the alternate naming.
                if "control_image" in str(e) or "unexpected keyword" in str(e):
                    try:
                        init_img = current_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        control_img = self.cached_control_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
                        kwargs.pop("control_image", None)
                        kwargs["image"] = control_img
                        kwargs["init_image"] = init_img
                        with torch.no_grad():
                            out = img2img_pipe(**kwargs)
                            next_img = out.images[0].convert("RGB")
                        del out, init_img, control_img
                        current_img = next_img
                        flush()
                    except Exception as e2:
                        print(f"⚠️ [ProgressiveHires] stage {stage_idx + 1} failed after fallback: {e2}. Keeping previous image.")
                        break
                else:
                    print(f"⚠️ [ProgressiveHires] stage {stage_idx + 1} failed: {e}. Keeping previous image.")
                    break
            except Exception as e:
                print(f"⚠️ [ProgressiveHires] stage {stage_idx + 1} failed: {e}. Keeping previous image.")
                break

        return current_img

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    async def generate_image(self, prompt, step_id, folder=None, filename=None, ref_image_path=None, scene_data=None, seed=None):
        print(f"Starting Production for Scene {step_id}...")
        print(f"🚀 Ultra-quality settings: base={os.getenv('GENERATION_BASE_WIDTH', '1536')}x{os.getenv('GENERATION_BASE_HEIGHT', '768')}, hires={self.progressive_hires_stages}, steps={self.progressive_hires_steps}, strengths={self.progressive_hires_strengths}")
        clean_prompt = prompt.replace("(", "").replace(")", "").strip()
        full_prompt = (
            f"mypano_style, <s0><s1>, ({clean_prompt}:1.25), realistic 360 panorama,"
            f"high definition photography, 8K, photorealistic, seamless, "
        )
        neg_prompt = (
            "grid, tiling, blocky, distorted, deformed horizon, vertical lines, watermark, "
            "blurry, ghosting, grey sky, flat color, color overflow, banding, compression artifacts, "
            "nadir, zenith, polar distortion, pinched top, swirl, person, crowd, buildings, indoor, "
            "oil painting, painterly, plastic texture, fake detail, oversharpened"
        )

        scene_data = scene_data or {}
        explicit_no_ip = isinstance(scene_data, dict) and scene_data.get("use_ip_adapter") is False
        ip_images = []
        if not explicit_no_ip:
            ip_images = self._load_reference_images(scene_data=scene_data, ref_image_path=ref_image_path, max_refs=3)
        use_rag_pipe = bool((not explicit_no_ip) and ip_images and self.rag_ip_adapter_ready)

        if use_rag_pipe:
            print(f"🖼️ RAG reference conditioning enabled. ref={ref_image_path or scene_data.get('reference_filename')}")
            pipe, pipe_name = self._select_generation_pipe(use_rag_pipe=True)
        else:
            if explicit_no_ip:
                print("🧪 Direct/no-RAG generation: using plain pipeline without IP-Adapter.")
            elif not ip_images:
                print("⚠️ No valid reference image found. Falling back to plain pipeline without IP-Adapter.")
            elif not self.rag_ip_adapter_ready:
                print("⚠️ RAG IP-Adapter unavailable. Falling back to plain pipeline.")
            pipe, pipe_name = self._select_generation_pipe(use_rag_pipe=False)

        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        generator_device = self.device if self.device == "cuda" and self.generation_device_mode != "cpu" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        try:
            folder = folder or "static/results/tmp"
            os.makedirs(folder, exist_ok=True)
            if not filename:
                filename = f"scene_{step_id}.png"
            file_path = os.path.join(folder, filename)

            base_width = int(os.getenv("GENERATION_BASE_WIDTH", "1536"))
            base_height = int(os.getenv("GENERATION_BASE_HEIGHT", "768"))

            pipe_kwargs = {
                "prompt": full_prompt,
                "negative_prompt": neg_prompt,
                "image": self.cached_control_img.resize((base_width, base_height), Image.Resampling.BICUBIC),
                "controlnet_conditioning_scale": float(os.getenv("GENERATION_CONTROLNET_SCALE", "0.25")),
                "width": base_width,
                "height": base_height,
                "num_inference_steps": int(os.getenv("GENERATION_STEPS", "36")),
                "guidance_scale": float(os.getenv("GENERATION_GUIDANCE_SCALE", "6.0")),
                "output_type": "latent",
                "generator": generator,
            }
            if use_rag_pipe:
                pipe_kwargs["ip_adapter_image"] = ip_images[0]

            output = pipe(**pipe_kwargs)
            latents = output.images
            del output
            latents = latents.to(dtype=self.dtype)

            with torch.no_grad():
                scaled_latents = latents / pipe.vae.config.scaling_factor
                image_tensor = pipe.vae.decode(scaled_latents, return_dict=False)[0]
                image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                base_img = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0].convert("RGB")
                del image_tensor, scaled_latents, latents

            # Progressive high-resolution diffusion refinement. This is the main quality upgrade:
            # it asks SDXL to add/detail natural textures at 1536x768 and 2048x1024 before saving.
            final_img = self._run_progressive_hires(
                pipe=pipe,
                pipe_name=pipe_name,
                base_img=base_img,
                full_prompt=full_prompt,
                neg_prompt=neg_prompt,
                generator=generator,
                use_rag_pipe=use_rag_pipe,
                ip_images=ip_images,
            )
            del base_img

            final_img.save(file_path)
            print(
                f"Scene {step_id} completed: {file_path} | size={final_img.size[0]}x{final_img.size[1]} "
                f"| seed={seed} | pipeline={pipe_name} | progressive_hires={self.progressive_hires_enabled}"
            )
            del final_img
            flush()
            return public_static_url(file_path), file_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            try:
                pipe.vae.to(dtype=self.dtype)
            except Exception:
                pass
            flush()
            return None, None

    # ------------------------------------------------------------------
    # Final fidelity enhancement
    # ------------------------------------------------------------------
    def upscale_image(self, input_path, output_path, scale=4):
        """
        Backward-compatible method name used by the rest of the pipeline.

        This no longer performs AI super-resolution. It performs fidelity-preserving
        enhancement only:
          - Lanczos resize to 4x by default
          - conservative luma-only CLAHE
          - edge-aware unsharp mask
        """
        if not input_path or not os.path.exists(input_path):
            print(f"⚠️ [FidelityFinisher] Input missing: {input_path}")
            return False
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not self.photo_finisher or not self.photo_finisher.enabled:
            print("⚠️ [FidelityFinisher] Disabled. Copying original image.")
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return os.path.exists(output_path)
            except Exception as e:
                print(f"❌ [FidelityFinisher] Copy failed: {e}")
                return False

        scale = int(scale or self.upscale_scale or 4)
        try:
            print(f"🚀 [FidelityFinisher] Enhancing: {input_path} -> {output_path}, scale={scale}")
            ok = self.photo_finisher.process_file(input_path, output_path, scale=scale)
            flush()
            if not ok or not os.path.exists(output_path):
                print(f"❌ [FidelityFinisher] Failed to write output: {output_path}")
                return False
            try:
                img = cv2.imread(output_path, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"✅ [FidelityFinisher] Saved: {output_path} | {w}x{h}")
                else:
                    print(f"✅ [FidelityFinisher] Saved: {output_path}")
            except Exception:
                print(f"✅ [FidelityFinisher] Saved: {output_path}")
            return True
        except Exception as e:
            print(f"❌ [FidelityFinisher] Failed: {e}")
            flush()
            return False

    def upscale_images_batch(self, input_paths, output_paths, scale=4):
        input_paths = [p for p in (input_paths or []) if p]
        output_paths = [p for p in (output_paths or []) if p]
        if not input_paths or not output_paths or len(input_paths) != len(output_paths):
            print("⚠️ [FidelityFinisher batch] Invalid input/output path lists.")
            return {}
        self._move_generation_pipes_to_cpu()
        if self.device == "cuda" and self.seam_pipe is not None and self.seam_device_mode == "swap":
            self._move_seam_pipe_to_cpu()
        results = {}
        scale = int(scale or self.upscale_scale or 1)
        for input_path, output_path in zip(input_paths, output_paths):
            ok = self.upscale_image(input_path, output_path, scale=scale)
            results[output_path] = output_path if ok and os.path.exists(output_path) else None
        return results

    # ------------------------------------------------------------------
    # Optional music generation, unchanged fallback behavior
    # ------------------------------------------------------------------
    def generate_music(self, prompt, style, title):
        base_url = (self.suno_base or "").rstrip('/')
        if not base_url:
            print("❌ [Suno API] Base URL is missing!")
            return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        headers = {
            "Authorization": f"Bearer {self.suno_key}",
            "Content-Type": "application/json",
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
            gen_endpoint = f"{base_url}/generate"
            print(f"🎵 [Suno API] Calling: {gen_endpoint}")
            response = requests.post(gen_endpoint, json=payload, headers=headers, timeout=30)
            print(f" Status Code: {response.status_code}")
            print(f" Raw Response: '{response.text}'")
            if response.status_code != 200:
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            try:
                res_json = response.json()
            except Exception:
                return "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            task_id = res_json.get("data", {}).get("taskId") if isinstance(res_json, dict) else None
            if not task_id:
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
