import torch
import os
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm


def calculate_real_pano_mu(image_folder, hf_token, device="cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        use_auth_token=hf_token,
        torch_dtype=dtype
    ).to(device)
    vae.eval()

    latent_vectors = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    print(f"Processing {len(image_files)} pictures...")

    for filename in tqdm(image_files):
        try:
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("RGB").resize((1024, 512))

            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
            img_tensor = (img_tensor / 127.5) - 1.0

            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.mode()
                latent_vectors.append(latent.cpu())

        except Exception as e:
            print(f"Jump {filename}: {e}")

    if not latent_vectors:
        return None

    all_latents = torch.cat(latent_vectors, dim=0)
    real_pano_mu = torch.mean(all_latents, dim=0, keepdim=True)

    return real_pano_mu


mu = calculate_real_pano_mu("./PictureData", "Your-hf-token")
torch.save(mu, "models/real_pano_mu.pt")