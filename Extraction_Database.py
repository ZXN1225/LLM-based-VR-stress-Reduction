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
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b_channel = cv2.split(lab)
    brightness = np.mean(l)
    contrast = np.std(l)

    # --- Kelvin's estimation ---
    # Extract the brightest 3% of pixels (simulating the search for the sun or core light source)
    threshold = np.percentile(l, 97)
    mask = l >= threshold
    bright_pixels = img[mask]

    if len(bright_pixels) > 0:
        avg_bgr = np.mean(bright_pixels, axis=0)
        kelvin = estimate_kelvin(avg_bgr[2], avg_bgr[1], avg_bgr[0])
    else:
        kelvin = 6500

    # --- Other quantitative indicators ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    (B, G, R) = cv2.split(img.astype("float"))
    colorfulness = np.sqrt(np.var(R - G) + np.var(0.5 * (R + G) - B)) + 0.3 * np.sqrt(
    np.mean(R - G) ** 2 + np.mean(0.5 * (R + G) - B) ** 2)

    return float(brightness), float(contrast), float(sharpness), float(colorfulness), float(kelvin)


def process_and_store_dataset(folder_path):
    """
    Using AI agent to process data of example pictures and store them in a database
    """

    valid_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")

    system_prompt = (
        "You are a professional image analyst. Analyze this 360-degree VR therapy image. "
        "Focus on: 1. Lighting_Tendency (Warm/Cool/Neutral/.... You can decide based on sunlight condition, any light source or anything that can affect the lighting tendency of the picture "
        "2. Environment type. 3. Emotional Mood Tags(1-3). 4. Objects:'List 3-5 key natural elements'"
        "Output ONLY a JSON object: "
        "{'lighting': 'Warm/Cool/Neutral', 'environment': '...', 'mood': '...', 'simple description': '...', 'objects': ''}"
    )

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

                    bright, contrast, sharp, color_score, kelvin = get_lighting_stats(path)

                    # Store to vector database
                    collection.add(
                        documents=[json.dumps(semantic)],
                        metadatas=[{
                            "filename": filename,
                            "temp_tendency": semantic['lighting'].lower(),
                            "brightness": bright,
                            "contrast": contrast,
                            "sharpness": sharp,
                            "colorfulness": color_score,
                            "environment": semantic['environment'],
                            "mood": semantic['mood'],
                            "objects": semantic['objects'],
                            "estimated_kelvin": kelvin,
                            "is_optimized_target": True if 3000 <= kelvin <= 4500 else False
                        }],
                        ids=[filename]
                    )
                    print(f"✅ Processed: {filename}")
                    break

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")


process_and_store_dataset("./PictureData")