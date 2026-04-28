import os
from dotenv import load_dotenv
import base64
import cv2
import json
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image
import io
import torch
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT').to(device)
seg_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.224, 0.224, 0.225]),
])

Image.MAX_IMAGE_PIXELS = None

load_dotenv()

API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)
db_client = chromadb.PersistentClient(path="./PictureBase")
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-small"
)
collection = db_client.get_or_create_collection(name="nature_environments", embedding_function=emb_fn)


def encode_image_to_base64(image_path):
    """ Encoding images of different formats """

    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")

            max_dimension = 2048
            if max(rgb_img.size) > max_dimension:
                rgb_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def estimate_kelvin(r, g, b):
    """ Estimate kelvin using McCamy's Formula """

    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    if (r_n + g_n + b_n) == 0: return 6500

    # Gamma correction
    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = to_linear(r_n)
    g_lin = to_linear(g_n)
    b_lin = to_linear(b_n)

    # Linear RGB to CIE XYZ
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    # Calculate chromaticity coordinates
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    # McCamy's Formula
    try:
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * (n**3) + 3525 * (n**2) + 6823.3 * n + 5524.33
        return float(np.clip(cct, 1500, 12000))
    except:
        return 6500


def calculate_visual_complexity(image):

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    r = 30
    magnitude_spectrum[cy - r:cy + r, cx - r:cx + r] = 0

    high_freq_score = np.mean(magnitude_spectrum)
    return float(high_freq_score)


def calculate_fractal_dimension(image):

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 250)

    def count_boxes(img, k):
        h, w = img.shape
        h_new, w_new = (h // k) * k, (w // k) * k
        img_cropped = img[:h_new, :w_new]
        boxes = img_cropped.reshape(h_new // k, k, w_new // k, k)
        return np.sum(np.any(boxes > 0, axis=(1, 3)))

    p = min(edges.shape)
    n = int(np.floor(np.log2(p)))
    scales = 2 ** np.arange(n, 1, -1)
    counts = [count_boxes(edges, s) for s in scales]

    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return float(round(-coeffs[0], 3))


def get_semantic_segmentation_stats(image_path):
    """
    Get greenery_radio by semantic segmentation
    """
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]

    seg_map = output.argmax(0).cpu().numpy()
    total_pixels = seg_map.size
    img_cv = cv2.imread(image_path)
    img_resized = cv2.resize(img_cv, (520, 520))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([95, 255, 255]))

    greenery_mask = ((seg_map == 0) | (seg_map == 15)) & (color_mask > 0)
    greenery_ratio = np.sum(greenery_mask) / total_pixels

    return float(greenery_ratio)


def get_sky_mask_stats(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]

    mask_sky = cv2.inRange(hsv, np.array([90, 15, 100]), np.array([140, 255, 255]))
    sky_ratio = np.sum(mask_sky > 0) / total_pixels

    return sky_ratio


def get_lighting_stats(image_path):
    """
    Advanced Physical Feature Extraction:
    Based on HSV Space and Highlight Region Analysis
    """

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)

    if img is None:
        return 0, "unknown"

    # --- Basic physical quantities ---
    h, w, _ = img.shape
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    cut = img[y_start:y_end, :, :]
    lab = cv2.cvtColor(cut, cv2.COLOR_BGR2Lab)
    l, a, b_channel = cv2.split(lab)
    brightness = np.mean(l)
    contrast = np.std(l)
    complexity = calculate_visual_complexity(cut)
    fd = calculate_fractal_dimension(cut)

    # --- Kelvin's estimation ---
    mask = (l> np.percentile(l, 20)) & (l < np.percentile(l, 90))
    if not np.any(mask):
        return 6500.0

    valid_pixels = cut[mask]
    median_bgr = np.median(valid_pixels, axis=0)
    kelvin = estimate_kelvin(median_bgr[2], median_bgr[1], median_bgr[0])

    # --- Other ratio indicators ---
    greenery_ratio = get_semantic_segmentation_stats(image_path)
    sky_ratio = get_sky_mask_stats(img)

    return float(brightness), float(contrast), float(greenery_ratio), float(sky_ratio), float(kelvin), float(complexity), float(fd)


def process_and_store_dataset(folder_path):
    """
    Using AI agent to process data of example pictures and store them in a database
    """

    valid_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")

    system_prompt = """
         You are a Clinical Environment Analyst specializing in Environmental Psychology.
         Analyze the image and provide a JSON description for future therapy
         Focus on: 
         1. lighting: Subjective warmth and intensity. (You can decide based on sunlight condition, any light source or anything that can affect the lighting tendency of the picture.)
         2. environment: Type of nature.
         3. mood: 1-3 emotional tags.
         4. objects: List 3-5 key natural elements.
         5. psychological elements or analysis: Your perspective regarding psychological knowledge (e.g, SRT and ART), such as Prospect, Refuge, Complexity, Soft_Fascination, Being Away, .etc.
         Output ONLY a JSON object: "
         {
          "lighting": "..."
          "environment": "...",
          "mood": "...",
          "objects": "...","
          "psychological": "{"prospect": "...", .....}",        
}
    """

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(folder_path, filename)
            print(f"Analyzing: {filename}...")

            base64_image = encode_image_to_base64(path)
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": system_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }],
                        max_tokens=250,
                        timeout=60.0,
                        response_format = {"type": "json_object"}
                    )
                    semantic = json.loads(response.choices[0].message.content)

                    bright, contrast, greenery, sky, kelvin, complexity, fd = get_lighting_stats(path)

                    # Store to vector database
                    collection.add(
                        documents=[json.dumps(semantic)],
                        metadatas=[{
                            "filename": filename,
                            "brightness": bright,
                            "contrast": contrast,
                            "greenery_ratio": greenery,
                            "sky_ratio": sky,
                            "complexity": complexity,
                            "fractal_dimension": fd,
                            "lighting": semantic['lighting'].lower(),
                            "environment": semantic['environment'],
                            "mood": semantic['mood'],
                            "objects": semantic['objects'],
                            "psychological": json.dumps(semantic['psychological'], ensure_ascii=False),
                            "estimated_kelvin": kelvin,
                        }],
                        ids=[filename]
                    )
                    print(f"✅ Processed: {filename}")
                    break

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")


if __name__ == "__main__":
    process_and_store_dataset("./PictureData")
