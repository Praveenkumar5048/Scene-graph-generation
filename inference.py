import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import build_model
import warnings

# Quiet noisy torchvision deprecation warnings that appear during model build/import.
# We only silence specific known messages to avoid hiding important warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*pretrained.*deprecated.*",
    category=UserWarning,
    module=r"torchvision.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Arguments other than a weight enum or `None` for 'weights'.*",
    category=UserWarning,
    module=r"torchvision.*",
)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output', default='', help='If set, save visualization to this path (png). If empty, show interactively')
    parser.add_argument('--print_results', action='store_true', default=True,
                        help='If set, print predicted subject-relation-object triplets to stdout')
    parser.add_argument('--save_json', default='', help='If set, save predicted triplets to this path (json)')
    parser.add_argument('--save_graph', default='', help='If set, save scene graph as JSON (nodes/edges) to this path')
    parser.add_argument('--graph_image', default='', help='If set, draw and save a simple graph image (requires networkx) to this path')
    parser.add_argument('--merge_iou', default=0.5, type=float, help='IoU threshold to merge nodes with same label')
    parser.add_argument('--graph_min_dist', default=0.14, type=float, help='Normalized minimum distance between nodes in graph layout (0-1)')
    parser.add_argument('--graph_iters', default=80, type=int, help='Number of repulsion iterations for graph layout')
    parser.add_argument('--graph_attraction', default=0.12, type=float, help='Attraction strength to original positions when laying out graph')
    parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)

    # load checkpoint with CPU safe mapping when requested
    map_location = 'cpu' if args.device == 'cpu' else None
    # torch.load in recent PyTorch versions may default to weights_only=True which
    # rejects certain pickled objects (raises UnpicklingError). Retry with
    # weights_only=False when needed (only for trusted checkpoints).
    try:
        ckpt = torch.load(args.resume, map_location=map_location)
    except Exception as e:
        # If the error indicates weights-only loading, retry with weights_only=False
        msg = str(e)
        if 'weights_only' in msg or 'Weights only load failed' in msg or 'WeightsUnpickler' in msg:
            try:
                # This argument exists on newer torch versions; pass it to allow full unpickling.
                ckpt = torch.load(args.resume, map_location=map_location, weights_only=False)
            except TypeError:
                # weights_only not supported (older torch); re-raise the original error
                raise
        else:
            raise
    model.load_state_dict(ckpt['model'])

    # move model to the requested device
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)
    # image size for later box rescaling and graph layout
    im_w, im_h = im.size

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        # After running the model we compute the predictions again here (based on the
        # outputs just produced) and optionally print/save them as a human readable list.
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                probas_obj.max(-1).values > 0.3))

        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = 10
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]

        # prepare a list of human-readable predictions
        pred_list = []
        for i, q in enumerate(keep_queries):
            sub_idx = int(probas_sub[q].argmax())
            rel_idx = int(probas[q].argmax())
            obj_idx = int(probas_obj[q].argmax())
            sub_score = float(probas_sub[q].max())
            rel_score = float(probas[q].max())
            obj_score = float(probas_obj[q].max())
            # bounding boxes corresponding to this (rescaled)
            sb = sub_bboxes_scaled[indices][i].tolist()
            ob = obj_bboxes_scaled[indices][i].tolist()
            pred = {
                'subject': CLASSES[sub_idx],
                'relation': REL_CLASSES[rel_idx],
                'object': CLASSES[obj_idx],
                'scores': {'subject': sub_score, 'relation': rel_score, 'object': obj_score},
                'subject_box': sb,
                'object_box': ob,
            }
            pred_list.append(pred)

        # print to stdout if requested
        if getattr(args, 'print_results', False):
            print('\nPredicted triplets:')
            for p in pred_list:
                print(f"{p['subject']} ({p['scores']['subject']:.2f}) -- {p['relation']} ({p['scores']['relation']:.2f}) --> {p['object']} ({p['scores']['object']:.2f})")

        # optionally save JSON
        if getattr(args, 'save_json', ''):
            import json
            with open(args.save_json, 'w') as jf:
                json.dump(pred_list, jf, indent=2)
            print(f"Saved predictions JSON to {args.save_json}")

        # Build a scene-graph style representation (nodes + edges)
        # Nodes are unique by (label, box) to avoid duplicates; edges reference node ids.
        if getattr(args, 'save_graph', '') or getattr(args, 'graph_image', ''):
            nodes = []
            edges = []
            # We'll merge nodes having the same label and IoU > merge_iou
            def iou(boxA, boxB):
                # boxes are [xmin, ymin, xmax, ymax]
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interW = max(0, xB - xA)
                interH = max(0, yB - yA)
                interArea = interW * interH
                boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
                boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
                unionArea = boxAArea + boxBArea - interArea
                if unionArea == 0:
                    return 0.0
                return interArea / unionArea

            merge_thr = float(getattr(args, 'merge_iou', 0.5))

            def find_or_add(label, box):
                # try to find existing node with same label and IoU > threshold
                for n in nodes:
                    if n['label'] == label:
                        if iou(n['box'], [int(round(x)) for x in box]) > merge_thr:
                            return n['id']
                # otherwise add new node
                nid = len(nodes)
                nodes.append({'id': nid, 'label': label, 'box': [int(round(x)) for x in box]})
                return nid

            for i, p in enumerate(pred_list):
                s_id = find_or_add(p['subject'], p['subject_box'])
                o_id = find_or_add(p['object'], p['object_box'])
                edges.append({'source': s_id, 'target': o_id, 'relation': p['relation'], 'scores': p['scores']})

            graph = {'nodes': nodes, 'edges': edges}

            if getattr(args, 'save_graph', ''):
                import json
                with open(args.save_graph, 'w') as gf:
                    json.dump(graph, gf, indent=2)
                print(f"Saved scene graph JSON to {args.save_graph}")

            # draw a graph image using node positions derived from box centers
            if getattr(args, 'graph_image', ''):
                try:
                    # compute normalized positions in [0,1] (flip y for display)
                    pos = {}
                    for n in nodes:
                        xmin, ymin, xmax, ymax = n['box']
                        cx = (xmin + xmax) / 2.0 / float(im_w)
                        cy = (ymin + ymax) / 2.0 / float(im_h)
                        # flip y so 0 is at bottom for matplotlib plotting
                        pos[n['id']] = (cx, 1.0 - cy)

                    # apply a small repulsive layout to reduce label/node overlap
                    # convert pos to mutable list
                    pid_list = list(pos.keys())
                    pcoords = {pid: list(pos[pid]) for pid in pid_list}
                    # repulsion parameters (in normalized [0,1] coord space)
                    min_dist = float(getattr(args, 'graph_min_dist', 0.14))
                    iterations = int(getattr(args, 'graph_iters', 80))
                    attraction = float(getattr(args, 'graph_attraction', 0.12))
                    damping = 0.65
                    # keep original positions to apply mild attraction to prevent runaway
                    orig = {pid: list(pos[pid]) for pid in pid_list}
                    for _ in range(iterations):
                        disp = {pid: [0.0, 0.0] for pid in pid_list}
                        for i in range(len(pid_list)):
                            for j in range(i + 1, len(pid_list)):
                                a = pid_list[i]
                                b = pid_list[j]
                                dx = pcoords[a][0] - pcoords[b][0]
                                dy = pcoords[a][1] - pcoords[b][1]
                                dist = (dx * dx + dy * dy) ** 0.5
                                if dist == 0:
                                    # small random jitter to break symmetry
                                    dx = (0.01 - 0.02 * (i % 2))
                                    dy = (0.01 - 0.02 * (j % 2))
                                    dist = (dx * dx + dy * dy) ** 0.5
                                if dist < min_dist:
                                    # repulsive force magnitude (quadratic near overlap)
                                    force = (min_dist - dist) / (dist + 1e-6)
                                    ux = dx / (dist + 1e-6)
                                    uy = dy / (dist + 1e-6)
                                    disp[a][0] += ux * force
                                    disp[a][1] += uy * force
                                    disp[b][0] -= ux * force
                                    disp[b][1] -= uy * force
                        # apply attraction back to original positions and damping
                        for pid in pid_list:
                            # attraction towards original position
                            disp[pid][0] += (orig[pid][0] - pcoords[pid][0]) * attraction
                            disp[pid][1] += (orig[pid][1] - pcoords[pid][1]) * attraction
                            # apply damping and move
                            pcoords[pid][0] += disp[pid][0] * damping
                            pcoords[pid][1] += disp[pid][1] * damping
                            # keep in bounds with margin
                            pcoords[pid][0] = min(0.98, max(0.02, pcoords[pid][0]))
                            pcoords[pid][1] = min(0.98, max(0.02, pcoords[pid][1]))

                    # update pos with the adjusted coordinates
                    for pid in pid_list:
                        pos[pid] = tuple(pcoords[pid])

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    # draw edges as arrows
                    for e in edges:
                        s = pos[e['source']]
                        t = pos[e['target']]
                        ax.annotate('', xy=t, xytext=s,
                                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0, shrinkA=10, shrinkB=10))
                        # label mid-point of edge
                        mx = (s[0] + t[0]) / 2.0
                        my = (s[1] + t[1]) / 2.0
                        ax.text(mx, my, e['relation'], fontsize=8, color='darkgreen', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

                    # draw nodes
                    xs = [pos[n['id']][0] for n in nodes]
                    ys = [pos[n['id']][1] for n in nodes]
                    labels = [n['label'] for n in nodes]
                    ax.scatter(xs, ys, s=600, c='lightblue', edgecolors='k')

                    # annotate node labels slightly above the node marker
                    for (x, y, lab) in zip(xs, ys, labels):
                        ax.text(x, y + 0.02, lab, fontsize=9, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(args.graph_image, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved scene graph image to {args.graph_image}")
                except Exception as e:
                    print('Could not draw graph image:', e)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

        fig.tight_layout()
        # save to file in headless / VM environments when --output is provided
        if args.output:
            fig.savefig(args.output)
            print(f"Saved visualization to {args.output}")
        else:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
