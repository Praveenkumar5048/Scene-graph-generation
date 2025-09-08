# main.py (refactored)
from config import COCO_CLASSES, GLOVE_PATH, IMAGE_PATH
from detector import Detector
from glove_utils import load_needed_glove
from features import prepare_object_features

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
        print("Feature extraction complete.")

        # Prepare final object features
        print("Preparing final object features...")
        h = prepare_object_features(out, glove)
        print("Final object feature shape:", h.shape)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()