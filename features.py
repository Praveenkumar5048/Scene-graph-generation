# features.py
import torch
import numpy as np

def prepare_object_features(detector_output, glove, img_w=640, img_h=480):
    features = detector_output["features"]
    boxes = detector_output["boxes"]
    labels = detector_output["class_names"]
    K = features.size(0)
    norm_boxes = boxes.clone()
    norm_boxes[:, [0, 2]] /= img_w
    norm_boxes[:, [1, 3]] /= img_h
    wh = (norm_boxes[:, 2] - norm_boxes[:, 0]) * (norm_boxes[:, 3] - norm_boxes[:, 1])
    box_feats = torch.cat([norm_boxes, wh.unsqueeze(1)], dim=1)
    emb_dim = 300
    word_embeddings = []
    for label in labels:
        vec = glove.get(label, np.zeros(emb_dim))
        word_embeddings.append(torch.tensor(vec))
    word_embeddings = torch.stack(word_embeddings)
    h = torch.cat([features, box_feats, word_embeddings], dim=1)
    return h
