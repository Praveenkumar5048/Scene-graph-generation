import os
import json
import random
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset
from PIL import Image


def list_images(root: str, exts=('jpg', 'jpeg', 'png')):
    p = Path(root)
    imgs = [str(x) for x in p.rglob('*') if x.suffix.lower().lstrip('.') in exts]
    imgs.sort()
    return imgs


def create_splits(root: str, out_path: str = 'splits.json', seed: int = 42):
    """Create train/test splits 70/30 and reserve up to 5000 images as validation (from train).

    If the dataset is smaller than 5000 training images, validation size is set to 20% of train.
    Saves JSON with keys: train, val, test (each is list of image paths).
    """
    imgs = list_images(root)
    random.Random(seed).shuffle(imgs)
    N = len(imgs)
    test_n = int(0.30 * N)
    train_n = N - test_n
    train = imgs[:train_n]
    test = imgs[train_n:]

    # reserve validation images from training set
    if train_n >= 5000 + 1:
        val_n = 5000
    else:
        val_n = max(1, int(0.2 * train_n))
    val = train[:val_n]
    train = train[val_n:]

    splits = {'train': train, 'val': val, 'test': test}
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    return splits


class ImageDataset(Dataset):
    """Dataset that lists images under a folder. Optionally reads per-image annotation JSONs
    from a sibling folder `annotations/` with the same basename and `.json` extension.

    Each annotation JSON (if present) is expected to contain a list of relations:
      {
        "boxes": [[x1,y1,x2,y2], ...],
        "relations": [{"subj_idx":0, "obj_idx":1, "pred_idx":5}, ...]
      }

    If annotations are not present, the dataset will still return image paths and the
    loader/trainer should handle missing GT (e.g., use detector-based pseudo-labels).
    """

    def __init__(self, image_list: List[str], annotations_dir: str = None, transform=None):
        self.images = list(image_list)
        self.annotations_dir = annotations_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _load_ann(self, img_path: str):
        if self.annotations_dir is None:
            return None
        bn = Path(img_path).stem
        p = Path(self.annotations_dir) / (bn + '.json')
        if not p.exists():
            return None
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        ann = self._load_ann(img_path)
        item = {'image_path': img_path, 'annotation': ann}
        return item


def get_dataloaders(dataset_root: str, splits_json: str = 'splits.json'):
    if not os.path.exists(splits_json):
        create_splits(dataset_root, out_path=splits_json)
    with open(splits_json, 'r') as f:
        splits = json.load(f)
    train_ds = ImageDataset(splits['train'], annotations_dir=os.path.join(dataset_root, 'annotations'))
    val_ds = ImageDataset(splits['val'], annotations_dir=os.path.join(dataset_root, 'annotations'))
    test_ds = ImageDataset(splits['test'], annotations_dir=os.path.join(dataset_root, 'annotations'))
    return train_ds, val_ds, test_ds
