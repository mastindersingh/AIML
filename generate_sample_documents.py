"""
Generate sample original and fake document images for testing document verification.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_original(path):
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 40), "Name: John Doe", fill=(0, 0, 0))
    d.text((20, 80), "ID: 1234567890", fill=(0, 0, 0))
    d.text((20, 120), "Valid: 2025", fill=(0, 0, 0))
    img.save(path)

def create_fake(path):
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 40), "Name: Jane Roe", fill=(0, 0, 0))  # Changed name
    d.text((20, 80), "ID: 1234567890", fill=(0, 0, 0))
    d.text((20, 120), "Valid: 2025", fill=(0, 0, 0))
    # Add a red rectangle to simulate tampering
    d.rectangle([250, 30, 390, 70], fill=(255, 0, 0, 128))
    img.save(path)

if __name__ == "__main__":
    os.makedirs("document_verification", exist_ok=True)
    create_original("document_verification/original_id_card.png")
    create_fake("document_verification/fake_id_card.png")
    print("Sample documents generated: original_id_card.png, fake_id_card.png")
