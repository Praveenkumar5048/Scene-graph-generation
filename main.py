from config import COCO_CLASSES, GLOVE_PATH, IMAGE_PATH
from detector import Detector
from glove_utils import load_needed_glove
from features import prepare_object_features
from ART import ARTEncoder, PairSpatialEncoder
import torch.nn as nn
from SOL import SemanticOrientedLearning, build_predicate_semantics
from DTM import DualTransH
from losses import FocalLoss
import torch

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
    contextualized_h, pair_feats, pair_msgs = encoder(h_projected, boxes_xywh)
    print("ART output shape:", contextualized_h.shape)
    print("Pair spatial features shape:", pair_feats.shape)
    print("Pairwise messages shape:", pair_msgs.shape)

        # --- Semantic Oriented Learning (SOL) demo ---
        # create a small set of example predicate labels (replace with actual VG predicates when training)
        pred_list = ["on", "in", "holding", "wearing", "near"]
        glove_for_preds = glove  # glove is a dict mapping words -> vectors
        pred_sem = build_predicate_semantics(pred_list, glove_for_preds, emb_dim=200)

        # Use pair-wise messages from ART as pair-level contexts for SOL
        # pair_msgs: [M, hidden_dim], pair_feats: [M, pair_dim]
        if pair_msgs.size(0) == 0:
            print("No pairs to process (K<=1). Exiting demo.")
            return

        # prepare a tiny SOL module
    sol = SemanticOrientedLearning(glove_dim=200, pair_ctx_dim=pair_msgs.size(1), pair_pos_dim=pair_feats.size(1), global_feat_dim=contextualized_h.size(1), num_predicates=len(pred_list), out_dim=256)

        # student (without predicate semantics)
        fused_student, logits_student = sol(pair_msgs, pair_feats, pair_feats, global_feat=contextualized_h.mean(dim=0, keepdim=True), pred_semantic=None)
        # teacher (with predicate semantics) - we pass class-level semantics
        fused_teacher, logits_teacher = sol(pair_msgs, pair_feats, pair_feats, global_feat=contextualized_h.mean(dim=0, keepdim=True), pred_semantic=pred_sem)

        print("SOL student logits shape:", logits_student.shape)
        print("SOL teacher logits shape:", logits_teacher.shape)

        # --- Dual TransH Module (DTM) demo ---
        # Rel encoder mock: use a linear projection to 2*r and pass through DTM
        rel_enc_proj = nn.Linear(pair_ctxs.size(1), 512)
        rel_enc_out = rel_enc_proj(pair_ctxs)  # [M, 512]
        dtm = DualTransH(in_dim=512, split_dim=256)
        rel_repr = dtm(rel_enc_out)
        print("DTM relation repr shape:", rel_repr.shape)

        # --- Example: replace cross-entropy with focal loss for predicate classification ---
        criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
        # build fake targets for demo (random) - in actual training use ground-truth predicate ids
        M = logits_student.size(0)
        targets = torch.randint(low=0, high=len(pred_list), size=(M,))
        loss = criterion(logits_student, targets)
        print("Demo focal loss on student logits:", loss.item())

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()