# config.py

# Visual Genome classes from pretrained_faster_rcnn/labels.json
VG_CLASSES = [
    "__background__", "roof", "kite", "pant", "bowl", "laptop", "paper", "shoe", "railing", "chair", "windshield",
    "ear", "tire", "cup", "bench", "tail", "bike", "board", "orange", "hat", "finger",
    "plate", "woman", "handle", "branch", "food", "elephant", "bear", "wave", "tile", "giraffe",
    "desk", "lady", "towel", "glove", "bag", "nose", "rock", "tower", "motorcycle", "sneaker",
    "fence", "people", "house", "sign", "hair", "street", "zebra", "racket", "logo", "girl",
    "arm", "wire", "leaf", "clock", "hill", "bird", "umbrella", "leg", "screen", "men",
    "sink", "trunk", "post", "sidewalk", "box", "boy", "cow", "skateboard", "plane", "stand",
    "pillow", "toilet", "pot", "number", "pole", "table", "boat", "sheep", "horse", "eye",
    "sock", "window", "vehicle", "curtain", "man", "banana", "fork", "head", "door", "shelf",
    "cabinet", "glass", "flag", "train", "child", "seat", "neck", "room", "player", "ski",
    "cap", "tree", "bed", "cat", "light", "skier", "engine", "drawer", "guy", "airplane",
    "car", "mountain", "shirt", "paw", "boot", "snow", "lamp", "book", "flower", "animal",
    "bus", "vegetable", "tie", "beach", "pizza", "wheel", "plant", "helmet", "track", "hand",
    "fruit", "mouth", "letter", "vase", "kid", "building", "short", "surfboard", "phone", "coat",
    "counter", "dog", "face", "jacket", "person", "truck", "bottle", "basket", "jean", "wing"
]

# Visual Genome predicates for scene graph relationships
VG_PREDICATES = [
    "and", "says", "belonging to", "over", "parked on", "growing on", "standing on", "made of",
    "attached to", "at", "in", "hanging from", "in front of", "from", "for", "lying on",
    "to", "behind", "flying in", "looking at", "on back of", "holding", "under", "laying on",
    "riding", "has", "across", "wearing", "walking on", "eating", "wears", "watching",
    "walking in", "sitting on", "between", "covered in", "carrying", "using", "along",
    "on", "with", "above", "part of", "covering", "of", "against", "playing", "near",
    "painted on", "mounted on"
]

# Aliases for compatibility
VG_OFFICIAL_CLASSES = VG_CLASSES
VG_OFFICIAL_PREDICATES = VG_PREDICATES

GLOVE_PATH = "glove.6B/glove.6B.300d.txt"
IMAGE_PATH = "test_image1.png"
