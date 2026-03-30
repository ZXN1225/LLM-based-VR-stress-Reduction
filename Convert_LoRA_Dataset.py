import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_DIR = "./PictureData"
OUTPUT_DIR = "./train_data"
TRIGGER_WORD = "mypano_style"
TARGET_WIDTH = 1024
TARGET_HEIGHT = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading BLIP...")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True
    ).to(device)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🚀 Start processing the number of  {len(image_files)} images...")

    for filename in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        try:
            with Image.open(img_path).convert("RGB") as img:

                img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

                file_base = os.path.splitext(filename)[0]
                img_resized.save(os.path.join(OUTPUT_DIR, f"{file_base}.jpg"), "JPEG", quality=95)

                # recognize contents(caption)
                inputs = processor(img, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                auto_caption = processor.decode(out[0], skip_special_tokens=True)

                final_caption = (f"{TRIGGER_WORD}, {auto_caption}, a high quality 360 panorama, "
                                 f"equirectangular, seamless, photorealistic, highly detailed.")

                with open(os.path.join(OUTPUT_DIR, f"{file_base}.txt"), "w") as f:
                    f.write(final_caption)

        except Exception as e:
            print(f"❌ {filename} fail: {e}")

    del model, processor
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    process_dataset()
    print("✅ Data caption and convertion succeed!")