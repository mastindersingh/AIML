"""
DocumentVerifier class: loads a model and verifies document images.
"""
import os
import numpy as np
import cv2
from PIL import Image
from counterfeit.model import siamese_model
from counterfeit.predict import ImageEncoder


class DocumentVerifier:
    def __init__(self, model_path=None):
        # Use Siamese model for document verification
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "..", "counterfeit", "siamese_model_weights.h5")
        self.target_shape = (224, 224)  # Default input size for ResNet50
        # Load Siamese embedding model
        self.encoder = None
        try:
            model = siamese_model(self.target_shape, pre_weights="imagenet", trainable=False)
            self.encoder = ImageEncoder(model, self.model_path, self.target_shape, full_trainable=False)
        except Exception as e:
            print(f"Warning: Could not load Siamese model. Falling back to placeholder. Error: {e}")
            self.encoder = None

    def verify(self, image_path):
        if not os.path.exists(image_path):
            return {'error': 'Image not found'}
        # List of reference (genuine) images
        originals = [
            os.path.join(os.path.dirname(__file__), 'original_id_card.png'),
            os.path.join(os.path.dirname(__file__), 'passport1.png'),
            os.path.join(os.path.dirname(__file__), 'passport2.png'),
            os.path.join(os.path.dirname(__file__), 'passport3.png'),
            os.path.join(os.path.dirname(__file__), 'sample_hf_0.png'),
            os.path.join(os.path.dirname(__file__), 'sample_hf_1.png'),
            os.path.join(os.path.dirname(__file__), 'sample_hf_2.png'),
        ]
        # Use Siamese model if available
        if self.encoder is not None:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    return {'error': 'Could not read image'}
                max_sim = -1.0
                for orig_path in originals:
                    if os.path.exists(orig_path):
                        anchor = cv2.imread(orig_path)
                        if anchor is not None:
                            sim = self.encoder.predict(anchor, img, img_aligned=False)
                            if sim > max_sim:
                                max_sim = sim
                # Threshold for genuine/fake (tune as needed)
                if max_sim > 0.85:
                    return {'document_status': 'genuine', 'confidence': float(max_sim)}
                else:
                    return {'document_status': 'fake', 'confidence': float(max_sim)}
            except Exception as e:
                return {'error': f'Verification failed: {e}'}
        # Fallback: use simple image difference
        try:
            from PIL import ImageChops
            img = Image.open(image_path).convert('RGB').resize((256, 256))
        except Exception as e:
            return {'error': f'Could not open image: {e}'}
        min_diff = 1.0
        for orig_path in originals:
            if os.path.exists(orig_path):
                try:
                    orig = Image.open(orig_path).convert('RGB').resize((256, 256))
                    diff = ImageChops.difference(img, orig)
                    diff_arr = np.array(diff)
                    norm_diff = np.mean(diff_arr) / 255.0
                    if norm_diff < min_diff:
                        min_diff = norm_diff
                except Exception:
                    continue
        if min_diff < 0.08:
            return {'document_status': 'genuine', 'confidence': float(1.0 - min_diff)}
        else:
            return {'document_status': 'fake', 'confidence': float(min_diff)}
