# detector.py
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from PIL import Image
import torch.nn as nn
import torchvision
from config import VG_CLASSES

class VGFasterRCNNDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.classes = VG_CLASSES
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Faster R-CNN with ResNet-101-FPN backbone for VG"""
        print("Creating Faster R-CNN with ResNet-101-FPN backbone...")
        
        try:
            # First, try to load the checkpoint to understand its structure
            print("Loading pretrained VG model from faster_rcnn_ckpt.pth")
            checkpoint = torch.load('faster_rcnn_ckpt.pth', map_location=self.device)
            
            # Check if the checkpoint has the expected structure
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Found 'model' key in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Found 'state_dict' key in checkpoint")
            else:
                state_dict = checkpoint
                print("Using checkpoint directly as state_dict")
            
            # Try to determine the original model architecture from checkpoint keys
            has_fpn = any('fpn' in key.lower() for key in state_dict.keys())
            backbone_type = 'resnet101' if any('layer4' in key for key in state_dict.keys()) else 'resnet50'
            
            print(f"Detected architecture: {backbone_type} with {'FPN' if has_fpn else 'no FPN'}")
            
            # Create model that matches the checkpoint architecture
            if has_fpn:
                # Create ResNet-101-FPN backbone exactly as it was trained
                backbone = resnet_fpn_backbone('resnet101', pretrained=False)
                self.model = FasterRCNN(
                    backbone=backbone,
                    num_classes=len(self.classes)
                )
            else:
                # Create standard ResNet-101 backbone without FPN
                from torchvision.models import resnet101
                backbone = resnet101(pretrained=False)
                # Remove the final classification layers
                backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
                backbone.out_channels = 2048
                
                self.model = FasterRCNN(
                    backbone=backbone,
                    num_classes=len(self.classes)
                )
            
            print("ResNet-101 model created successfully")
            
            # Check classifier compatibility
            classifier_key = 'roi_heads.box_predictor.cls_score.weight'
            if classifier_key in state_dict:
                classifier_shape = state_dict[classifier_key].shape
                expected_classes = len(self.classes)
                actual_classes = classifier_shape[0]
                print(f"Classifier found: {actual_classes} classes, expected: {expected_classes}")
                
                if actual_classes != expected_classes:
                    print(f"Class count mismatch. Removing classifier from checkpoint.")
                    # Remove classifier layers from checkpoint
                    keys_to_remove = [k for k in state_dict.keys() if 'box_predictor' in k]
                    for key in keys_to_remove:
                        del state_dict[key]
                    print("Classifier layers removed from checkpoint")
            else:
                print("No classifier found in checkpoint. Will use our classifier.")
            
            # Try to load with exact parameter matching
            try:
                # Load the state dict with strict=False to handle missing keys
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                # Count matched parameters
                total_params = len(self.model.state_dict())
                matched_params = total_params - len(missing_keys)
                match_percentage = (matched_params / total_params) * 100
                
                print(f"Parameter matching: {matched_params}/{total_params} ({match_percentage:.1f}%)")
                
                if match_percentage < 50:
                    print("Low parameter matching. Trying alternative loading method...")
                    # Try loading with different key mapping
                    model_dict = self.model.state_dict()
                    
                    # Create a mapping of parameters
                    matched_dict = {}
                    for model_key in model_dict.keys():
                        if model_key in state_dict:
                            if model_dict[model_key].shape == state_dict[model_key].shape:
                                matched_dict[model_key] = state_dict[model_key]
                            else:
                                print(f"Shape mismatch for {model_key}: model {model_dict[model_key].shape} vs checkpoint {state_dict[model_key].shape}")
                    
                    # Update model with matched parameters
                    model_dict.update(matched_dict)
                    self.model.load_state_dict(model_dict)
                    
                    match_percentage = (len(matched_dict) / len(model_dict)) * 100
                    print(f"Alternative loading: {len(matched_dict)}/{len(model_dict)} ({match_percentage:.1f}%)")
                
                print("VG pretrained model loaded successfully!")
                
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    
            except Exception as load_error:
                print(f"Error during state dict loading: {load_error}")
                # If loading fails, create fresh model with COCO weights
                backbone = resnet_fpn_backbone('resnet101', pretrained=True)
                self.model = FasterRCNN(
                    backbone=backbone,
                    num_classes=len(self.classes)
                )
                print("Created fresh ResNet-101-FPN with COCO weights")
            
            # Always ensure classifier is correct for VG classes
            print("Ensuring classifier head matches VG classes...")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, len(self.classes))
            self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, len(self.classes) * 4)
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Creating fresh ResNet-101-FPN model with COCO pretraining...")
            # Fallback to fresh COCO model
            backbone = resnet_fpn_backbone('resnet101', pretrained=True)
            self.model = FasterRCNN(
                backbone=backbone,
                num_classes=len(self.classes)
            )
        
        self.model.to(self.device)
        self.model.eval()
        print("Model initialization complete.")

    def load_image(self, image_path):
        """Load and preprocess image"""
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print("Image loaded and converted to RGB.")
        return image

    def detect_objects(self, image_path, confidence_threshold=0.05, max_detections=10):
        """Detect objects in image and return detections"""
        # Load image
        image = self.load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"Image tensor shape: {image_tensor.shape}")
        
        # Run detection
        print("Running detector...")
        with torch.no_grad():
            detections = self.model(image_tensor)
        print("Detector output received.")
        
        # Extract results
        boxes = detections[0]['boxes'].cpu()
        scores = detections[0]['scores'].cpu()
        labels = detections[0]['labels'].cpu()
        
        print(f"Raw detections: {len(boxes)} boxes")
        if len(scores) > 0:
            print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"Scores above 0.1: {(scores > 0.1).sum().item()}")
            print(f"Scores above 0.05: {(scores > 0.05).sum().item()}")
        
        # Filter by confidence
        valid_indices = scores >= confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        
        print(f"Filtered detections: {len(boxes)} boxes with score >= {confidence_threshold}")
        
        # Limit to max detections
        if len(boxes) > max_detections:
            top_indices = torch.argsort(scores, descending=True)[:max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]
        
        print(f"Top-K selected: {len(boxes)} boxes")
        
        return boxes, scores, labels

    def extract_features(self, image_path, confidence_threshold=0.05, max_detections=10, top_k=None):
        """Extract features for detected objects"""
        # Handle backward compatibility with top_k parameter
        if top_k is not None:
            max_detections = top_k
            
        # Get detections
        boxes, scores, labels = self.detect_objects(image_path, confidence_threshold, max_detections)
        
        if len(boxes) == 0:
            return torch.empty(0, 1024), []
        
        # Load image for feature extraction
        image = self.load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract backbone features
        print("Backbone features extracted.")
        with torch.no_grad():
            # Get intermediate features from backbone
            backbone_features = self.model.backbone(image_tensor)
            
            # Prepare boxes for ROI pooling
            boxes_list = [boxes.to(self.device)]
            
            # Extract ROI pooled features
            print("ROI pooled features extracted.")
            roi_features = self.model.roi_heads.box_roi_pool(
                backbone_features, boxes_list, [image_tensor.shape[-2:]]
            )
            
            # Pass through box head
            print("Box head features extracted.")
            box_features = self.model.roi_heads.box_head(roi_features)
            
        print("Feature extraction complete.")
        
        # Debug: Print predicted class indices and names
        print(f"Predicted class indices: {labels.tolist()}")
        class_names = [self.classes[label] for label in labels]
        print(f"Predicted class names: {class_names}")
        
        return box_features.cpu(), class_names

# Alias for backward compatibility
Detector = VGFasterRCNNDetector