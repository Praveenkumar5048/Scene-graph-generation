"""Smoke test: run a single forward pass (synthetic tensors) to validate model wiring."""
import torch
from ART import ARTEncoder
from SOL import SemanticOrientedLearning, build_predicate_semantics
from DTM import DualTransH
from losses import FocalLoss


def run_smoke():
    device = torch.device('cpu')
    # synthetic K objects
    K = 5
    # synthetic features: visual 1024 + spatial 4 + glove 300 = 1328
    h = torch.randn(K, 1328)
    # convert boxes to [x,y,w,h] expected by ART.PairSpatialEncoder
    boxes = torch.rand(K, 4) * 100.0
    # project to ART input dim (ART expects 1329 in repo); create simple linear layer here
    proj = torch.nn.Linear(1328, 1329)
    h_proj = proj(h)

    art = ARTEncoder(input_dim=1329, hidden_dim=512, pair_dim=128)
    h2, pair_feats, pair_msgs = art(h_proj, boxes)

    # build predicate semantics (dummy)
    preds = [f'pred_{i}' for i in range(10)]
    pred_sem = build_predicate_semantics(preds, {})

    sol = SemanticOrientedLearning(glove_dim=pred_sem.size(1), pair_ctx_dim=pair_msgs.size(1), pair_pos_dim=pair_feats.size(1), global_feat_dim=h2.size(1), num_predicates=len(preds))
    fused_s, logits_s = sol(pair_msgs, pair_feats, pair_feats, global_feat=h2.mean(dim=0, keepdim=True), pred_semantic=None)
    fused_t, logits_t = sol(pair_msgs, pair_feats, pair_feats, global_feat=h2.mean(dim=0, keepdim=True), pred_semantic=pred_sem)

    # focal loss on student logits with random targets
    M = logits_s.size(0)
    targets = torch.randint(0, logits_s.size(1), (M,))
    focal = FocalLoss()
    loss = focal(logits_s, targets)
    print('Smoke test shapes:')
    print('h2', h2.shape)
    print('pair_feats', pair_feats.shape)
    print('pair_msgs', pair_msgs.shape)
    print('logits_student', logits_s.shape)
    print('loss', loss.item())


if __name__ == '__main__':
    run_smoke()
