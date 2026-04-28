import torch
import os
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle


def calculate_real_pano_stats(
    image_folder,
    hf_token,
    device="cuda",
    pca_dim=128
):
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        use_auth_token=hf_token,
        torch_dtype=dtype
    ).to(device)
    vae.eval()

    latent_vectors = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    print(f"Processing {len(image_files)} images...")

    for filename in tqdm(image_files):
        try:
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("RGB").resize((1024, 512))

            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
            img_tensor = (img_tensor / 127.5) - 1.0

            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.mode()
                latent = latent.view(-1).cpu().numpy()
                latent_vectors.append(latent)

        except Exception as e:
            print(f"Skip {filename}: {e}")

    if len(latent_vectors) < 10:
        raise ValueError("Too few images to estimate distribution")

    latents = np.stack(latent_vectors, axis=0)  # [N, 32768]

    print("Fitting PCA...")
    pca = PCA(n_components=pca_dim)
    latents_reduced = pca.fit_transform(latents)  # [N, D]

    print("Computing statistics...")

    mu = np.mean(latents_reduced, axis=0)
    cov = np.cov(latents_reduced, rowvar=False)

    cov += 1e-6 * np.eye(cov.shape[0])

    inv_cov = np.linalg.pinv(cov)

    return mu, cov, inv_cov, pca

mu, cov, inv_cov, pca = calculate_real_pano_stats(
    "./PictureData",
    "hf_....",
    pca_dim=128
)

np.save("models/real_pano_mu.npy", mu)
np.save("models/real_pano_cov.npy", cov)
np.save("models/real_pano_inv_cov.npy", inv_cov)

with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)
