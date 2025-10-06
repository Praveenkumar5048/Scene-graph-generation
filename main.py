from config import VG_CLASSES, GLOVE_PATH, IMAGE_PATH
from detector import Detector
from glove_utils import load_needed_glove
from features import prepare_object_features
from sgg_model import SceneGraphGenerator
from improve_model import initialize_sgg_model_weights, enhance_scene_graph_with_rules
import torch.nn as nn
import torch
import os

def main():
    try:
        print("=== Faster R-CNN with ResNet-50-FPN Scene Graph Generation ===")
        print("Following research paper methodology with exact architecture")
        
        detector = FasterRCNNDetector(confidence_threshold=0.5)  # Faster R-CNN as per paper
        predicates = VG_OFFICIAL_PREDICATES
        
        # Get the class labels from the detector  
        object_classes = detector.class_labels
        print(f"Using {len(object_classes)} VG object classes from pretrained model")
        print(f"Using {len(predicates)} VG predicates")

        # Prepare needed words for GloVe embeddings
        print("Loading GloVe embeddings...")
        needed_words = set([c.lower() for c in VG_CLASSES if c != "__background__"])
        glove = load_needed_glove(GLOVE_PATH, needed_words)
        print("Loaded embeddings:", len(glove))

        # Extract features using Faster R-CNN with ResNet-50-FPN
        print(f"Extracting features from image: {IMAGE_PATH}")
        out = detector.extract_features(IMAGE_PATH, top_k=10)
        print("Detected objects:", out["class_names"])

        # Check if any objects were detected
        if len(out["class_names"]) == 0:
            print("⚠️ No objects detected!")
            print("Try lowering the confidence threshold or check the image.")
            return

        # Prepare object features for attention transformer SGG
        print("Preparing object features for attention transformer...")
        h = prepare_object_features(out, glove)
        boxes = out["boxes"]
        class_ids = out["class_labels"]
        print("Final object feature shape:", h.shape)
        print("Detected class IDs:", class_ids)

        # Check if any objects were detected
        if h.shape[0] == 0:
            print("No objects detected in the image. Cannot proceed with ART processing.")
            print("This might be due to:")
            print("1. Architecture mismatch between pretrained model and current model")
            print("2. Model weights not loading correctly")
            print("3. Confidence threshold too high")
            print("Consider using the COCO pretrained model as fallback or checking model architecture.")
            return

        # --- ART Contextualization ---
        
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format for ART
        print("Converting box format...")
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1

        # Initialize complete SGG model
        print("Initializing Scene Graph Generator...")
        # Use only the number of detected objects, not all VG classes
        num_detected = len(out["class_names"])
        sgg_model = SceneGraphGenerator(
            input_dim=h.shape[1],    # 1328
            art_hidden_dim=512,
            obj_dim=512,
            num_classes=num_detected,  # Only detected objects
            num_predicates=len(predicates),  # VG predicates
            use_sol=True,
            use_dtm=True
        )
        print(f"SGG model initialized for {num_detected} detected objects and {len(predicates)} predicates")
        
        # Check if trained model exists, otherwise use better initialization
        if os.path.exists('sgg_model_trained.pth'):
            print("Loading pre-trained model...")
            sgg_model.load_state_dict(torch.load('sgg_model_trained.pth'))
        else:
            print("No trained model found. Using improved initialization...")
            initialize_sgg_model_weights(sgg_model)
        
        print("SGG model initialized.")

        # Create mapping from detected objects to indices 0, 1, 2, ...
        # This ensures the SGG model works with consecutive indices
        detected_to_sequential = {i: i for i in range(len(out["class_names"]))}
        sequential_class_ids = torch.arange(len(out["class_names"]), dtype=torch.long)
        
        print(f"Using sequential class IDs: {sequential_class_ids.tolist()}")
        print(f"For detected objects: {out['class_names']}")

        # Run inference with sequential indices
        print("Running Scene Graph Generation...")
        scene_graph = sgg_model.predict(h, boxes_xywh, detected_classes=sequential_class_ids)
        
        # Enhance with rule-based relationships (using sequential indices)
        print("Enhancing with rule-based relationships...")
        scene_graph = enhance_scene_graph_with_rules(scene_graph, sequential_class_ids, boxes_xywh)
        
        print("\n=== SCENE GRAPH RESULTS ===")
        print(f"Objects detected: {len(scene_graph['objects'])}")
        print(f"Relations detected: {len(scene_graph['relations'])}")
        print(f"Triplets found: {len(scene_graph['triplets'])}")
        
        # Display triplets with readable names
        if scene_graph['triplets']:
            print("\nTop Scene Graph Triplets:")
            print(f"DEBUG: Detected objects: {out['class_names']}")
            print(f"DEBUG: COCO names: {out.get('coco_names', 'N/A')}")
            
            for i, triplet in enumerate(scene_graph['triplets'][:10]):  # Show top 10
                # Convert indices to readable names using the actual detected objects
                subj_idx = triplet['subject']
                obj_idx = triplet['object']
                
                # Use the detected object names directly
                if subj_idx < len(out['class_names']):
                    subj_name = out['class_names'][subj_idx]
                else:
                    subj_name = f"object_{subj_idx}"
                    
                if obj_idx < len(out['class_names']):
                    obj_name = out['class_names'][obj_idx]
                else:
                    obj_name = f"object_{obj_idx}"
                    
                pred_name = predicates[triplet['predicate']] if triplet['predicate'] < len(predicates) else f"pred_{triplet['predicate']}"
                
                print(f"{i+1}. {subj_name} -> {pred_name} -> {obj_name} (score: {triplet['score']:.3f})")
                
            # Also show detected objects with their names
            print("\nDetected Objects:")
            for i, obj_class in enumerate(scene_graph['objects'][:10]):  # Show first 10 objects
                obj_name = object_classes[obj_class.item()] if obj_class.item() < len(object_classes) else f"class_{obj_class.item()}"
                score = scene_graph['object_scores'][i].item() if i < len(scene_graph['object_scores']) else 1.0
                print(f"{i+1}. {obj_name} (confidence: {score:.3f})")
        else:
            print("No relationship triplets detected.")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()