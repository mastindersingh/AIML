
"""
Sample script for document verification using DocumentVerifier.
Detects if a document image is likely to be fake or altered.

Optionally, downloads a sample document dataset from Hugging Face using the datasets library.
"""
import os
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def download_sample_documents():
    if load_dataset is None:
        print("Install the 'datasets' library to download sample datasets: pip install datasets")
        return
    # Example: Replace with a real dataset name if available
    dataset_name = "nielsr/funsd"
    print(f"Downloading sample dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    # Save a few images to the document_verification folder
    save_dir = os.getcwd()
    count = 0
    for item in dataset['train']:
        if 'image' in item:
            img = item['image']
            img_path = os.path.join(save_dir, f"sample_hf_{count}.png")
            img.save(img_path)
            print(f"Saved {img_path}")
            count += 1
        if count >= 3:
            break

from document_verifier import DocumentVerifier

if __name__ == "__main__":
    # Uncomment to download sample documents from Hugging Face
    download_sample_documents()
    verifier = DocumentVerifier()
    images = [
        ("original_id_card.png", "Original (Genuine) Document"),
        ("fake_id_card.png", "Fake (Tampered) Document"),
        ("passport1.png", "Passport Sample 1"),
        ("passport2.png", "Passport Sample 2"),
        ("passport3.png", "Passport Sample 3"),
        ("sample_hf_0.png", "HuggingFace Sample 0"),
        ("sample_hf_1.png", "HuggingFace Sample 1"),
        ("sample_hf_2.png", "HuggingFace Sample 2")
    ]
    # Add all tampered images
    tampered_dir = os.path.join(os.getcwd(), "tampered")
    if os.path.exists(tampered_dir):
        for fname in os.listdir(tampered_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append((f"tampered/{fname}", f"Tampered: {fname}"))
    for img, desc in images:
    result = verifier.verify(img)
        print(f"{desc} - {img}: {result}")
