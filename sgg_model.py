import torch
import torch.nn as nn
from ART import ARTEncoder
from relation_decoder import RelationDecoder
from sol import subject_object_localization
from dtm import dynamic_triplet_mining


class SceneGraphGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int = 1328,
        art_hidden_dim: int = 512,
        obj_dim: int = 512,
        num_classes: int = 150,
        num_predicates: int = 50,
        use_sol: bool = True,
        use_dtm: bool = True,
    ) -> None:
        super().__init__()
        # Project input features to ART input (optionally with class score)
        # Note: ART in this repo expects input_dim+1 (we'll append score channel); ensure compatibility
        self.append_score = True
        art_in = input_dim + (1 if self.append_score else 0)
        self.art = ARTEncoder(input_dim=art_in, hidden_dim=art_hidden_dim, pair_dim=128)

        # Object head: classify objects from contextualized embeddings (optional in inference-only)
        self.obj_head = nn.Sequential(
            nn.Linear(art_hidden_dim, obj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obj_dim, num_classes)
        )
        self.obj_proj = nn.Linear(art_hidden_dim, obj_dim)

        # Relation decoder
        self.relation = RelationDecoder(obj_dim=obj_dim, pair_geom_dim=128, num_predicates=num_predicates)

        self.use_sol = use_sol
        self.use_dtm = use_dtm

    @torch.no_grad()
    def predict(self, obj_feats_1328: torch.Tensor, boxes_xywh: torch.Tensor, detected_classes: torch.Tensor | None = None,
                scores: torch.Tensor | None = None):
        """
        Inference-only forward pass.
        obj_feats_1328: [K, 1328]
        boxes_xywh: [K, 4]
        detected_classes: [K] optional integer ids for pretty-printing
        scores: [K] detection scores (for SOL and to append to ART input)
        returns dict with objects, relations, triplets
        """
        device = obj_feats_1328.device
        K = obj_feats_1328.size(0)

        if K == 0:
            return {
                "objects": torch.empty(0, dtype=torch.long),
                "object_scores": torch.empty(0),
                "relations": torch.empty(0, 2, dtype=torch.long),
                "relation_scores": torch.empty(0),
                "triplets": []
            }

        if scores is None:
            scores = torch.ones(K, device=device)

        # Append score channel to features for ART input
        art_in = torch.cat([obj_feats_1328, scores.view(-1, 1)], dim=-1) if self.append_score else obj_feats_1328

        # ART contextualization
        ctx = self.art(art_in, boxes_xywh)

        # Object classification (optional)
        logits_obj = self.obj_head(ctx)
        obj_scores, obj_pred = logits_obj.softmax(dim=-1).max(dim=-1)

        # Project for relations
        obj_embed = self.obj_proj(ctx)

        # SOL: pick objects to keep
        keep_idx = torch.arange(K, device=device)
        if self.use_sol:
            keep_idx = subject_object_localization(scores=scores, boxes=boxes_xywh, topk=min(50, K))
        kept_embed = obj_embed[keep_idx]
        kept_boxes = boxes_xywh[keep_idx]

        # DTM: generate pairs
        pairs = torch.stack(torch.meshgrid(keep_idx, keep_idx, indexing="ij"), dim=-1).view(-1, 2)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if self.use_dtm:
            # Convert to local indices for geometry computation
            local_pairs = dynamic_triplet_mining(num_objects=kept_embed.size(0))
            # map local pairs back to global indices
            if local_pairs.numel() > 0:
                global_pairs = torch.stack([keep_idx[local_pairs[:, 0]], keep_idx[local_pairs[:, 1]]], dim=-1)
            else:
                global_pairs = torch.zeros(0, 2, dtype=torch.long, device=device)
        else:
            global_pairs = pairs

        # Relation classification
        # Use kept features but index geometry using kept boxes. Need local mapping for decoder.
        if global_pairs.numel() == 0:
            rel_logits = torch.zeros(0, self.relation.mlp[-1].out_features, device=device)
        else:
            # build a map from global idx -> local idx in kept set
            id_map = {int(k.item()): i for i, k in enumerate(keep_idx)}
            local_pairs = torch.tensor([[id_map[int(s.item())], id_map[int(o.item())]] for s, o in global_pairs],
                                       dtype=torch.long, device=device)
            rel_logits = self.relation(kept_embed, kept_boxes, local_pairs)

        if rel_logits.numel() == 0:
            triplets = []
            relation_scores = torch.empty(0, device=device)
        else:
            rel_scores, rel_pred = rel_logits.softmax(dim=-1).max(dim=-1)
            relation_scores = rel_scores
            # Build triplets structure
            triplets = []
            for (s, o), p, sc in zip(global_pairs.tolist(), rel_pred.tolist(), rel_scores.tolist()):
                triplets.append({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "score": float(sc)
                })

        return {
            "objects": obj_pred,
            "object_scores": obj_scores,
            "relations": global_pairs if global_pairs.numel() else torch.empty(0, 2, dtype=torch.long, device=device),
            "relation_scores": relation_scores,
            "triplets": triplets,
        }
