from config import COCO_CLASSES, GLOVE_PATH, IMAGE_PATH
from detector import Detector
from glove_utils import load_needed_glove
from features import prepare_object_features
from ART import ARTEncoder
import torch.nn as nn

def main():
    try:
        # Load detector
        print("Initializing detector...")
        detector = Detector()
        print("Detector initialized.")

        # Prepare needed words for GloVe
        print("Loading GloVe embeddings...")
        needed_words = set([c.lower() for c in COCO_CLASSES if c != "__background__"])
        glove = load_needed_glove(GLOVE_PATH, needed_words)
        print("Loaded embeddings:", len(glove))

        # Extract features from image
        print(f"Extracting features from image: {IMAGE_PATH}")
        out = detector.extract_features(IMAGE_PATH, top_k=10)
        print("Extracted classes are", out["class_names"])

        # Prepare final object features
        print("Preparing final object features...")
        h = prepare_object_features(out, glove)
        boxes = out["boxes"]
        print("Final object feature shape:", h.shape)

        # --- ART Contextualization ---
        
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format for ART
        print("Converting box format...")
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
        # x and y remain the same (top-left corner)
        
        # Project input features to hidden dimension before ART
        print("Projecting features to hidden dimension...")
        input_proj = nn.Linear(h.shape[1], 512)
        h_projected = input_proj(h)
        
        print("Passing features through ART Encoder...")
        encoder = ARTEncoder(input_dim=512, hidden_dim=512)
        contextualized_h = encoder(h_projected, boxes_xywh)
        print("ART output shape:", contextualized_h.shape)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()