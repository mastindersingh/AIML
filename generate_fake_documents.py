"""
Script to generate simple fake/tampered versions of document images for testing.
This script overlays text, adds noise, and blurs images to simulate tampering.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np

def tamper_image(input_path, output_path, tamper_type="text_overlay"):
    img = Image.open(input_path).convert("RGB")
    if tamper_type == "text_overlay":
        draw = ImageDraw.Draw(img)
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), "FAKE", fill=(255, 0, 0), font=font)
    elif tamper_type == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=5))
    elif tamper_type == "noise":
        arr = np.array(img)
        noise = np.random.randint(0, 50, arr.shape, dtype='uint8')
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype('uint8'))
    img.save(output_path)
    print(f"Saved tampered image: {output_path}")

def generate_fakes(source_dir, output_dir, tamper_types=None):
    if tamper_types is None:
        tamper_types = ["text_overlay", "blur", "noise"]
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            in_path = os.path.join(source_dir, fname)
            for ttype in tamper_types:
                out_name = f"tampered_{ttype}_" + fname
                out_path = os.path.join(output_dir, out_name)
                tamper_image(in_path, out_path, ttype)

if __name__ == "__main__":
    # Example usage: generate fakes for all images in document_verification/
    src = os.path.dirname(__file__)
    out = os.path.join(src, "tampered")
    generate_fakes(src, out)
