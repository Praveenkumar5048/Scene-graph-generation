import os
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from detector import Detector
from features import prepare_object_features
from ART import ARTEncoder
from SOL import SemanticOrientedLearning, build_predicate_semantics
from DTM import DualTransH
from losses import FocalLoss
from dataset import get_dataloaders


def load_predicates(pred_path: str = None, num_default: int = 51):
    if pred_path and os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            preds = [l.strip() for l in f if l.strip()]
        return preds
    # fallback: create default predicate names
    return [f'pred_{i}' for i in range(num_default)]


def map_relations_to_pair_labels(K, relations, num_predicates):
    """Map GT relations (list of dicts subj_idx, obj_idx, pred_idx) to pair-index labeling
    Pair ordering matches ART.PairSpatialEncoder which iterates i over objects then j over objects (skip i==j).
    Returns a list of length M with class indices (default 0)
    """
    M = K * (K - 1)
    labels = [0] * M
    # build mapping from (i,j) to index
    idx = 0
    pair_map = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            pair_map[(i, j)] = idx
            idx += 1
    if relations is None:
        return labels
    for r in relations:
        si = r.get('subj_idx')
        oi = r.get('obj_idx')
        pi = r.get('pred_idx', 0)
        if (si, oi) in pair_map:
            labels[pair_map[(si, oi)]] = int(pi) if pi < num_predicates else 0
    return labels


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds = get_dataloaders(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # build modules
    detector = Detector() if args.use_detector else None

    # features -> ART expected input dim = 1328 (visual 1024 + spatial 4 + glove 300)
    art = ARTEncoder(input_dim=1328, hidden_dim=512, pair_dim=128).to(device)
    # SOL expects pair_ctx_dim matching pair messages dim (ART returns pair_messages with dim hidden_dim)
    predicates = load_predicates(args.predicates)
    glove_dict = None
    if args.glove and os.path.exists(args.glove):
        # lazy load when building predicate semantics
        import numpy as np
        glove_dict = {}
        with open(args.glove, 'r', encoding='utf8') as f:
            for line in f:
                toks = line.strip().split()
                glove_dict[toks[0]] = np.array(toks[1:], dtype='float32')

    pred_sem = build_predicate_semantics(predicates, glove_dict) if glove_dict is not None else build_predicate_semantics(predicates, {})
    pred_sem = pred_sem.to(device)

    sol = SemanticOrientedLearning(glove_dim=pred_sem.size(1), pair_ctx_dim=512, pair_pos_dim=128, global_feat_dim=512, num_predicates=len(predicates)).to(device)
    dtm = DualTransH(in_dim=512*2, split_dim=512).to(device)

    focal = FocalLoss(gamma=args.gamma, alpha=args.alpha).to(device)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    params = list(art.parameters()) + list(sol.parameters()) + list(dtm.parameters()) + list(input_proj.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    # training loop (simplified): iterate over train images and run forward pass
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        art.train(); sol.train(); dtm.train(); input_proj.train()
        for item in train_loader:
            img_path = item['image_path'][0]
            ann = item['annotation'][0]

            # get detector features
            if detector is not None:
                det_out = detector.extract_features(img_path)
            else:
                # Without detector, we can't extract RoI features; create a small synthetic placeholder
                # Create K random proposals
                import torch
                K = 6
                det_out = {
                    'features': torch.randn(K, 1024),
                    'boxes': torch.rand(K, 4) * 200.0,
                    'class_names': [str(i) for i in range(K)],
                    'scores': torch.rand(K)
                }

            # prepare object features and normalized boxes (cx,cy,w,h)
            h, norm_boxes = prepare_object_features(det_out)
            h = h.to(device)
            boxes = norm_boxes.to(device)

            # ART forward
            h2, pair_feats, pair_msgs = art(h, boxes)

            # pair_feats: [M, pair_dim], pair_msgs: [M, hidden_dim]
            M = pair_feats.size(0)

            # build per-image global feat: simple pooling of h2
            global_feat = h2.mean(dim=0, keepdim=True)

            # build labels mapping from annotation if present
            if ann is not None and 'relations' in ann and 'boxes' in ann:
                K_gt = len(ann['boxes'])
                pair_labels = map_relations_to_pair_labels(K_gt, ann['relations'], len(predicates))
                # If detector proposals differ from GT count, fallback to default
                if K_gt != h.size(0):
                    pair_labels = [0] * M
            else:
                # fallback to random labels (demo only)
                pair_labels = [random.randint(0, len(predicates)-1) for _ in range(M)]

            if M == 0:
                continue

            targets = torch.tensor(pair_labels, dtype=torch.long, device=device)

            # SOL forward (teacher uses pred_semantics)
            fused_student, logits_student = sol(pair_msgs, pair_feats, pair_feats, global_feat=global_feat, pred_semantic=None)
            fused_teacher, logits_teacher = sol(pair_msgs, pair_feats, pair_feats, global_feat=global_feat, pred_semantic=pred_sem)

            # focal loss on student logits
            loss_focal = focal(logits_student, targets)

            # teacher-student KL (align student to teacher)
            loss_kl = kl_loss(F.log_softmax(logits_student, dim=1), F.softmax(logits_teacher.detach(), dim=1))

            loss = loss_focal + args.kl_weight * loss_kl

            opt.zero_grad()
            loss.backward()
            opt.step()

        # checkpoint
        ckpt_path = Path(args.ckpt_dir) / f'checkpoint_epoch{epoch+1}.pt'
        torch.save({'art': art.state_dict(), 'sol': sol.state_dict(), 'dtm': dtm.state_dict(), 'input_proj': input_proj.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset root folder (images/annotations)')
    parser.add_argument('--use-detector', action='store_true', help='use detector to extract features (slow, requires torchvision pretrained model)')
    parser.add_argument('--glove', type=str, default=None, help='path to glove txt file (optional)')
    parser.add_argument('--predicates', type=str, default=None, help='path to predicate list (one per line)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--kl-weight', type=float, default=0.5)
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
