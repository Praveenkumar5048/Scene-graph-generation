# features.py
import torch
import numpy as np

def prepare_object_features(detector_output, glove, img_w=640, img_h=480):
    """
    Prepare object features as per the paper:
    - Visual features (from ROI head): 1024-dim
    - Spatial features (normalized bbox): 4-dim  
    - Semantic features (GloVe): 300-dim
    Total: 1328-dim per object
    """
    features = detector_output["features"]  # [K, 1024]
    boxes = detector_output["boxes"]        # [K, 4] in [x1,y1,x2,y2]
    labels = detector_output["class_names"] # [K]
    K = features.size(0)
    
    # Handle case when no objects are detected
    if K == 0:
        print("No objects detected, returning empty feature tensor")
        return torch.empty(0, 1328)  # Return empty tensor with correct feature dimension
    
    # Spatial features: normalized bounding box coordinates
    norm_boxes = boxes.clone().float()
    norm_boxes[:, [0, 2]] /= img_w  # normalize x coordinates
    norm_boxes[:, [1, 3]] /= img_h  # normalize y coordinates
    
    # Semantic features: GloVe word embeddings
    emb_dim = 300
    word_embeddings = []
    for label in labels:
        vec = glove.get(label, np.zeros(emb_dim))
        word_embeddings.append(torch.tensor(vec, dtype=torch.float32))
    
    if word_embeddings:
        word_embeddings = torch.stack(word_embeddings)  # [K, 300]
    else:
        # Create empty embedding tensor if no labels
        word_embeddings = torch.zeros(0, emb_dim)
    
    # Concatenate: visual + spatial + semantic
    h = torch.cat([features, norm_boxes, word_embeddings], dim=1)  # [K, 1024+4+300=1328]
    return h
