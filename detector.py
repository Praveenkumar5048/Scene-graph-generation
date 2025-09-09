# detector.py
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from config import COCO_CLASSES

class Detector:
    def __init__(self):
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False
        self.backbone = self.detector.backbone
        self.roi_heads = self.detector.roi_heads

    def extract_features(self, img_path, top_k=36, score_thresh=0.2):
        print(f"Loading image: {img_path}")
        image = Image.open(img_path).convert("RGB")
        print("Image loaded and converted to RGB.")
        img_tensor = F.to_tensor(image).unsqueeze(0)
        print(f"Image tensor shape: {img_tensor.shape}")
        with torch.no_grad():
            print("Running detector...")
            outputs = self.detector(img_tensor)
            print("Detector output received.")
            boxes = outputs[0]['boxes']
            labels = outputs[0]['labels']
            scores = outputs[0]['scores']
            print(f"Raw detections: {boxes.shape[0]} boxes")
            keep = scores >= score_thresh
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            print(f"Filtered detections: {boxes.shape[0]} boxes with score >= {score_thresh}")
            topk = min(top_k, boxes.size(0))
            boxes, labels, scores = boxes[:topk], labels[:topk], scores[:topk]
            print(f"Top-K selected: {topk} boxes")
            features = self.backbone(img_tensor)
            print("Backbone features extracted.")
            proposals = [boxes]
            box_features = self.roi_heads.box_roi_pool(features, proposals, img_tensor.shape[-2:])
            print("ROI pooled features extracted.")
            box_features = self.roi_heads.box_head(box_features)
            print("Box head features extracted.")
            result = {
                "features": box_features,
                "boxes": boxes,
                "class_labels": labels,
                "class_names": [COCO_CLASSES[i] if i < len(COCO_CLASSES) else "unknown" for i in labels],
                "scores": scores
            }
        print("Feature extraction complete.")
        return result
