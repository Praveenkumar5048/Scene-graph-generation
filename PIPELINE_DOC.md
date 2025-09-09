# Scene Graph Generation Pipeline

## Step 1: Object Detection

**In the paper:**
- Faster R-CNN with ResNeXt-101-FPN backbone
- Pretrained on Visual Genome (VG)
- Detector is frozen (not updated during SGG training)
- Outputs: bounding boxes, ROI appearance features, class scores

**In our implementation:**
- Used Faster R-CNN with ResNet-50-FPN backbone
- Pretrained on COCO (since it's lightweight & available in torchvision)
- We froze its parameters
- We extracted bounding boxes, ROI features (1024-dim), and scores

## Step 2: Object Feature Construction

Each object representation `hi` is a concatenation of:
- **Appearance feature** `vi` (1024-dim from detector ROI head)
- **Spatial feature** `bi` (normalized bbox coordinates [x1,y1,x2,y2], 4-dim)
- **Semantic embedding** `ei` (GloVe word embedding of detected class label, 300-dim)

→ **Final: 1328-dim vector per object** (1024 + 4 + 300)

## Step 3: Object Encoder (ART)

After constructing object features, we add contextual information using the **Attention Redirection Transformer (ART)** as the object encoder:

### ART Architecture:
- **Pairwise Spatial Encoder**: Computes geometric relationships between object pairs (relative positions, size ratios, proper IoU calculation)
- **Stage 1 (Attention Distraction)**: Attends over all object pairs, capturing broad contextual cues
- **Stage 2 (Attention Integration)**: Refines attention by focusing on filtered pairs using mask

### Key Features:
- Two-stage attention mechanism with residual connections
- LayerNorm and FFN after each attention stage  
- Projects 1328-dim input to 512-dim hidden representation
- Outputs contextualized object embeddings where each object vector carries information about its interactions with other objects

### Implementation Details:
- **Mask Support**: Stage 2 can filter object pairs based on provided mask
- **Proper IoU**: Fixed intersection-over-union calculation for spatial relationships
- **Dimension Matching**: Ensures consistent tensor dimensions throughout processing

### Differences from Paper:
- **Mask A Strategy**: Paper uses learned sampling for reliable pairs; our implementation supports masking but uses simpler heuristics (top-K by detector confidence)
- **Multi-head Attention**: Paper suggests multi-head attention; our implementation uses scalar attention weights
- **Training**: Paper trains ART end-to-end with SGG objectives; our current implementation is inference-only

## Current Status
✅ **Completed**: Object detection, feature construction, ART encoder architecture  
⏳ **Next**: Optimised ART + SOL + DTM + relation decoder.