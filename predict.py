import logging
import cv2
import numpy as np
from scipy.spatial import distance
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from image_utils import align_imgs

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ImageEncoder:
    embedding_model = None
    def __init__(self, model: Model, model_path: str, target_shape: tuple[int, int], full_trainable: bool) -> None:
        self.model_path = model_path
        self.target_shape = target_shape
        self.model = model(target_shape, None, full_trainable)
        if model_path:
            self.load_model(self.model_path)
        else:
            logger.info("Deferring model load!")
            self.model = None
    def load_model(self, model_path: str) -> None:
        self.model.load_weights(model_path)
        self.embedding_model = self.model.get_layer("Embedding")
    def preprocess(self, img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        target_h, target_w = target_shape[0], target_shape[1]
        scale_h = min(target_w / img_w, target_h / img_h)
        img = cv2.resize(img, (0, 0), fx=scale_h, fy=scale_h)
        img = cv2.resize(img, target_shape[::-1], interpolation=cv2.INTER_AREA)
        return img
    def img2vec(self, img: np.ndarray) -> np.ndarray:
        img = self.preprocess(img, self.target_shape)
        img = np.expand_dims(img, axis=0)
        return self.embedding_model(resnet.preprocess_input(img))
    def predict(self, anchor_img: np.ndarray, input_img: np.ndarray, img_aligned: bool) -> float:
        anchor_img = self.preprocess(anchor_img, self.target_shape)
        input_img = self.preprocess(input_img, self.target_shape)
        if img_aligned:
            input_img, _ = align_imgs(input_img, anchor_img, min_match_counts=10)
        anchor_embeddings = self.img2vec(anchor_img)
        input_embeddings = self.img2vec(input_img)
        cos_sim = 1.0 - distance.cosine(anchor_embeddings[0], input_embeddings[0])
        return cos_sim
