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
    
    # Convert boxes from [x1,y1,x2,y2] to normalized [x_center, y_center, w, h]
    boxes = boxes.clone().float()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = ((x2 - x1) / img_w).clamp(min=0.0)
    h = ((y2 - y1) / img_h).clamp(min=0.0)
    norm_boxes = torch.stack([cx, cy, w, h], dim=1)
    
    # Semantic features: GloVe word embeddings
    emb_dim = 300
    word_embeddings = []
    for label in labels:
        key = label.lower() if isinstance(label, str) else str(label)
        vec = glove.get(key, np.zeros(emb_dim))
        word_embeddings.append(torch.tensor(vec, dtype=torch.float32))
    word_embeddings = torch.stack(word_embeddings)  # [K, 300]
    
    # Concatenate: visual + spatial + semantic
    h = torch.cat([features, norm_boxes, word_embeddings], dim=1)  # [K, 1024+4+300=1328]
    # return both feature vectors and the normalized boxes in [cx,cy,w,h] form (for ART pair encoder)
    return h, norm_boxes
